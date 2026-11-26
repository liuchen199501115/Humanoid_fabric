from pathlib import Path
import numpy as np,json
from scipy import sparse
from solve_upper_fit_qp import solve
p=Path('/home/sky/Humanoid_fabric/acc_data/g1_full_tracking_20261106')
d=np.load(p/'thirteenth_support_knee20.qp_2150.npz');get=lambda k:d['priority_problem_'+k][0].astype(float)
H,h,E,e,G,b,lo,hi=[get(k) for k in ['H','h','A_eq','b_eq','G','b','lb','ub']];n=len(h)
A=d['task_joint_posture_A'][0].astype(float);M=d['task_joint_posture_metric'][0].astype(float);r=(d['task_joint_posture_bias'][0]-d['task_joint_posture_desired'][0]).astype(float);J=np.c_[A,np.zeros((29,n-35))]
C=sparse.csc_matrix(np.vstack([E,G,np.eye(n)]));low=np.r_[e,np.full(len(G),-np.inf),lo];high=np.r_[e,b,hi];out=[]
for axes in ['roll','waist']:
 for scale in [4,16,64,256]:
  delta=np.zeros_like(M)
  for k in ([13] if axes=='roll' else [12,13,14]):delta[k,k]=(scale-1)*M[k,k]
  x,rep=solve(H+J.T@delta@J,h+J.T@delta@r,C,low,high)
  result=dict(axes=axes,scale=scale,solver=rep)
  if rep['accepted']:
   result.update(waist_acc=x[18:21].tolist(),root_acc=x[:6].tolist())
   for name in ['center_of_mass','feet_pose','joint_posture']:
    residual=d['task_'+name+'_A'][0]@x[:35]+d['task_'+name+'_bias'][0]-d['task_'+name+'_desired'][0]
    result[name+'_residual_norm']=float(np.linalg.norm(residual))
  out.append(result);print(json.dumps(result),flush=True)
(p/'step13_waist_weight_probe.json').write_text(json.dumps(dict(scope='Saved-state counterfactual only; no solution applied',cases=out),indent=2)+'\n')
