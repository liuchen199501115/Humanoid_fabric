from pathlib import Path
import numpy as np,json,osqp
from scipy import sparse
p=Path(__file__).parent;run='fixed_clock120_body10_full_body_preview42_collision_surface28'
a=np.load(p/(run+'.full_body_preview.first_plan.npz'),allow_pickle=True);d=np.load(p/(run+'.replan_states/3.686.npz'),allow_pickle=True)
k=np.searchsorted(a['frames'],d['frame']+1e-8,side='right')-1
want=a['controls'][k,:35];A=d['problem_A_eq'];b=d['problem_b_eq'];G=d['problem_G'];h=d['problem_b'];lb=d['problem_lb'];ub=d['problem_ub'];n=len(lb)
only_acc=np.max(abs(G[:,35:]),axis=1)<1e-12
res=G[:,:35]@want-h;indices=np.argsort(np.where(only_acc,res,-np.inf))[-8:][::-1]
report=dict(stage=int(k),rows=len(G),planned_acceleration=want.tolist(),acceleration_inequality_violations=[dict(row=int(i),residual=float(res[i]),nonzero_columns=np.flatnonzero(abs(G[i])>1e-8).tolist()) for i in indices],cases=[])
tubes=3 if len(np.flatnonzero(G[-3,:35]))==29 else 2
for case,remove in [('all',0),('without_tracking_tubes',tubes)]:
 C=G[:-remove] if remove else G;bound=h[:-remove] if remove else h
 H=np.diag(np.r_[np.ones(35),np.full(n-35,1e-8)]);g=np.r_[-want,np.zeros(n-35)]
 As=np.maximum(np.linalg.norm(A,axis=1),1e-8);Gs=np.maximum(np.linalg.norm(C,axis=1),1e-8)
 solver=osqp.OSQP();solver.setup(P=sparse.csc_matrix(H),q=g,A=sparse.csc_matrix(np.vstack((A/As[:,None],C/Gs[:,None],np.eye(n)))),l=np.r_[b/As,np.full(len(C),-np.inf),lb],u=np.r_[b/As,bound/Gs,ub],verbose=False,eps_abs=1e-8,eps_rel=1e-8,max_iter=100000,polish=True)
 sol=solver.solve();row=dict(case=case,status=sol.info.status)
 if sol.x is not None:
  x=sol.x;duals=sol.y[len(A):len(A)+len(C)]/Gs;top=np.argsort(abs(duals))[-6:][::-1]
  row.update(acceleration_difference_max=float(abs(x[:35]-want).max()),base_angular_acceleration=x[3:6].tolist(),residual=float(max(abs(A@x-b).max(),(C@x-bound).max(),(lb-x).max(),(x-ub).max())),largest_inequality_duals=[dict(row=int(i),dual=float(duals[i])) for i in top])
 report['cases'].append(row)
(p/'surface_projection_constraints_audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
