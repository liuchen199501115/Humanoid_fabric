"""Counterfactual saved-QP analysis only; no altered constraints are executed."""
from pathlib import Path
import json,numpy as np,osqp
from scipy import sparse
p=Path(__file__).parent;reports=[];history=np.load(p/'fixed_clock120_body10_overdue.npz')
for time in [3.9,4.,4.04,4.06]:
 d=np.load(p/f'fixed_clock120_body10_constraint_audit.snapshot_{time:.3f}.npz');H0=d['problem_H'][0].astype(float);g0=d['problem_h'][0].astype(float);G0=d['problem_G'][0].astype(float);h0=d['problem_b'][0].astype(float);A=d['problem_A_eq'][0].astype(float);b=d['problem_b_eq'][0].astype(float);lb0=d['problem_lb'][0].astype(float);ub0=d['problem_ub'][0].astype(float);n=len(g0);accrows=np.flatnonzero(abs(G0[:,:35]).max(1)>1e-12);tube_rows=3 if len(np.flatnonzero(G0[-3,:35]))==29 else 2;friction_rows=int(accrows[0]);cases=[]
 def task(name,select=None):
  J=d['task_'+name+'_A'][0].astype(float);M=d['task_'+name+'_metric'][0].astype(float);active=d['task_'+name+'_active'][0].copy()
  if select is not None:active[:select]=False
  M*=active[:,None]*active[None,:];return J.T@M@J,J.T@M@(d['task_'+name+'_bias'][0]-d['task_'+name+'_desired'][0])
 for case in ['baseline','com_feedback12_gain1','com_feedback12_gain3']:
  H=H0.copy();g=g0.copy();G=G0.copy();h=h0.copy();lb=lb0.copy();ub=ub0.copy();labels=[f'friction_{i}' if i<friction_rows else f'collision_{i-friction_rows}' if i<len(G)-tube_rows else f'tube_{i-len(G)+3}' for i in range(len(G))]
  if case=='without_collision_barriers':
   select=np.r_[np.arange(friction_rows),np.arange(len(G)-tube_rows,len(G))];G=G[select];h=h[select];labels=[labels[i] for i in select]
  if case=='without_tracking_tubes':G=G[:-tube_rows];h=h[:-tube_rows];labels=labels[:-tube_rows]
  if case=='unbounded_effort':lb[47:]=-1e6;ub[47:]=1e6
  if case in ('without_com_objective','without_body_objective'):
   th,tg=task('center_of_mass' if case=='without_com_objective' else 'body_orientation');H[:35,:35]-=th;g[:35]-=tg
  if case.startswith('com_feedback12_'):
   index=abs(history['history_time'].ravel()-time).argmin();gain=1. if case.endswith('gain1') else 3.
   position=history['history_com'][index].ravel()-history['history_com_target'][index].ravel();velocity=history['history_com_velocity'][index].ravel()-history['history_com_target_velocity'][index].ravel();ff=history['history_com_target_acceleration'][index].ravel();feedback=-gain*np.array([35,35,25])*position-np.sqrt(gain)*np.array([12,12,10])*velocity
   old=d['task_center_of_mass_desired'][0];desired=old.copy();desired[:2]=(ff+feedback.clip(-12,12))[:2];J=d['task_center_of_mass_A'][0];M=d['task_center_of_mass_metric'][0];g[:35]+=J.T@M@(old-desired)
  if case=='upper_posture4':th,tg=task('joint_posture',12);H[:35,:35]+=3*th;g[:35]+=3*tg
  As=np.maximum(np.linalg.norm(A,axis=1),1e-8);Gs=np.maximum(np.linalg.norm(G,axis=1),1e-8);C=sparse.csc_matrix(np.vstack((A/As[:,None],G/Gs[:,None],np.eye(n))));lower=np.r_[b/As,np.full(len(G),-np.inf),lb];upper=np.r_[b/As,h/Gs,ub];solver=osqp.OSQP();solver.setup(P=sparse.triu(sparse.csc_matrix(H)),q=g,A=C,l=lower,u=upper,verbose=False,eps_abs=1e-8,eps_rel=1e-8,max_iter=100000,polish=True);sol=solver.solve();x=sol.x
  if x is None or sol.info.status_val not in [1,2]:cases.append(dict(case=case,status=sol.info.status));continue
  violation=float(max(abs(A@x-b).max(),(G@x-h).max(),(lb-x).max(),(x-ub).max(),0));active_ineq=sol.y[len(A):len(A)+len(G)]/Gs;top=np.argsort(abs(active_ineq))[-8:][::-1];bounddual=sol.y[-n:];topbounds=np.argsort(abs(bounddual))[-8:][::-1]
  cases.append(dict(case=case,status=sol.info.status,residual=violation,root_angular_acceleration=x[3:6].tolist(),upper_selected_acceleration=x[[19,20,29]].tolist(),com_acceleration=(d['task_center_of_mass_A'][0]@x[:35]+d['task_center_of_mass_bias'][0]).tolist(),largest_inequality_duals=[dict(row=labels[i],dual=float(active_ineq[i]),slack=float(h[i]-G[i]@x)) for i in top if abs(active_ineq[i])>1e-5],largest_bound_duals=[dict(variable=int(i),dual=float(bounddual[i]),value=float(x[i]),bounds=[float(lb[i]),float(ub[i])]) for i in topbounds if abs(bounddual[i])>1e-5],recorded_acceleration_max_difference=float(abs(x[:35]-d['predicted_acc'][0]).max()) if case=='baseline' else None))
 reports.append(dict(time=time,contact_mode=d['contact_mode'].tolist(),physical_force=d['forces'].tolist(),cases=cases));print(time,[(c['case'],c.get('upper_selected_acceleration')) for c in cases],flush=True)
(p/'qp_com_feedback_audit.json').write_text(json.dumps(reports,indent=2)+'\n')
