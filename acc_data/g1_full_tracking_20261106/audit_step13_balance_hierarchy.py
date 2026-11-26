"""Counterfactual hierarchy at saved states, never applied to the simulator."""
from pathlib import Path
import json,numpy as np
from scipy import sparse
from solve_upper_fit_qp import solve
p=Path(__file__).parent;rows=[]
for filename in ['thirteenth_feasible_reference.qp_2150.npz','thirteenth_feasible_root_braking_v2.qp_2050.npz']:
 d=np.load(p/filename);get=lambda k:d['priority_problem_'+k][0].astype(float);H,h,E,e,G,b,lo,hi=[get(k) for k in ['H','h','A_eq','b_eq','G','b','lb','ub']];n=len(h)
 for names in [('center_of_mass',),('center_of_mass','body_orientation','source_pelvis_orientation')]:
  H1=np.eye(n)*1e-8;h1=np.zeros(n);locks=[]
  for name in names:
   A=d['task_'+name+'_A'][0].astype(float);M=d['task_'+name+'_metric'][0].astype(float);r=(d['task_'+name+'_bias'][0]-d['task_'+name+'_desired'][0]).astype(float);J=np.c_[A,np.zeros((len(A),n-35))];H1+=J.T@M@J;h1+=J.T@M@r;locks.append(J)
  C=sparse.csc_matrix(np.vstack((E,G,np.eye(n))));low=np.r_[e,np.full(len(G),-np.inf),lo];high=np.r_[e,b,hi];x,report=solve(H1,h1,C,low,high)
  row=dict(snapshot=filename,first_tasks=names,first_solver=report)
  if report['accepted']:
   L=np.vstack(locks);target=L@x;C2=sparse.vstack((C,sparse.csc_matrix(L)),format='csc');x,report=solve(H,h,C2,np.r_[low,target-1e-4],np.r_[high,target+1e-4]);row['second_solver']=report
   row.update(waist_acceleration=x[18:21].tolist(),root_acceleration=x[:6].tolist())
   for name in ['center_of_mass','joint_posture','source_pelvis_orientation']:
    residual=d['task_'+name+'_A'][0]@x[:35]+d['task_'+name+'_bias'][0]-d['task_'+name+'_desired'][0];row[name+'_residual']=residual.tolist()
  rows.append(row);print(json.dumps(row),flush=True)
(p/'step13_balance_hierarchy_probe.json').write_text(json.dumps(dict(scope='Saved-state counterfactual only; no command or tracking claim.',cases=rows),indent=2)+'\n')
