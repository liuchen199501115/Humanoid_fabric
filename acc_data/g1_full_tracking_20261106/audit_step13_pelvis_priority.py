"""Saved-state priority counterfactuals only; no control command is executed."""
from pathlib import Path
import numpy as np,json
from scipy import sparse
from solve_upper_fit_qp import solve
p=Path(__file__).parent;d=np.load(p/'thirteenth_feasible_reference.qp_2150.npz');get=lambda k:d['priority_problem_'+k][0].astype(float)
H,h,E,e,G,b,lo,hi=[get(k) for k in ['H','h','A_eq','b_eq','G','b','lb','ub']];n=len(h)
C=sparse.csc_matrix(np.vstack((E,G,np.eye(n))));low=np.r_[e,np.full(len(G),-np.inf),lo];high=np.r_[e,b,hi]
A=d['task_source_pelvis_orientation_A'][0].astype(float);M=d['task_source_pelvis_orientation_metric'][0].astype(float);r=(d['task_source_pelvis_orientation_bias'][0]-d['task_source_pelvis_orientation_desired'][0]).astype(float);J=np.c_[A,np.zeros((3,n-35))]
rows=[]
for axes in ['none','yaw','all']:
 for scale in ([0.] if axes=='none' else [.1,.3,1.,3.]):
  metric=M*scale
  if axes=='yaw':metric[:2]=0.;metric[:,:2]=0.
  x,rep=solve(H+J.T@metric@J,h+J.T@metric@r,C,low,high);row=dict(axes=axes,scale=scale,solver=rep)
  if rep['accepted']:
   row.update(waist_acceleration=x[18:21].tolist(),root_acceleration=x[:6].tolist())
   for task in ['center_of_mass','feet_pose','joint_posture','body_orientation','source_pelvis_orientation']:
    residual=d['task_'+task+'_A'][0]@x[:35]+d['task_'+task+'_bias'][0]-d['task_'+task+'_desired'][0];row[task+'_residual_norm']=float(np.linalg.norm(residual))
  rows.append(row);print(json.dumps(row),flush=True)
(p/'step13_pelvis_priority_probe.json').write_text(json.dumps(dict(scope='Saved-state first-priority solve; no new command or tracking success.',cases=rows),indent=2)+'\n')
