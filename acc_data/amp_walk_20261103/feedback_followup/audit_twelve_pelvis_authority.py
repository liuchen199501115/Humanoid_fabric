from pathlib import Path
import json,numpy as np,osqp
from scipy import sparse
p=Path(__file__).parent;rows=[]
for time in [4.,4.15,4.25]:
 d=np.load(p/f'wbc_twelve_reproduced.snapshot_{time:.3f}.npz');H0=d['problem_H'][0].astype(float);g0=d['problem_h'][0].astype(float);A=d['problem_A_eq'][0].astype(float);b=d['problem_b_eq'][0].astype(float);G=d['problem_G'][0].astype(float);h=d['problem_b'][0].astype(float);lb=d['problem_lb'][0].astype(float);ub=d['problem_ub'][0].astype(float);n=len(g0);J=d['task_source_pelvis_orientation_A'][0].astype(float);M=d['task_source_pelvis_orientation_metric'][0].astype(float);bias=d['task_source_pelvis_orientation_bias'][0];desired=d['task_source_pelvis_orientation_desired'][0];AA=np.c_[J,np.zeros((3,n-35))];As=np.maximum(np.linalg.norm(A,axis=1),1e-8);Gs=np.maximum(np.linalg.norm(G,axis=1),1e-8);C=sparse.csc_matrix(np.vstack((A/As[:,None],G/Gs[:,None],np.eye(n))));lo=np.r_[b/As,np.full(len(G),-np.inf),lb];hi=np.r_[b/As,h/Gs,ub]
 for scale in [0,.05,.1,.25]:
  H=H0+scale*AA.T@M@AA;g=g0+scale*AA.T@M@(bias-desired);solver=osqp.OSQP();solver.setup(P=sparse.triu(sparse.csc_matrix(H)),q=g,A=C,l=lo,u=hi,verbose=False,eps_abs=1e-8,eps_rel=1e-8,max_iter=100000,polish=True);sol=solver.solve();row=dict(time=time,scale=scale,status=sol.info.status)
  if sol.info.status_val in [1,2]:
   x=sol.x;row.update(pelvis_angular_acceleration=(AA@x+bias).tolist(),root_angular_acceleration=x[3:6].tolist(),physical_residual=float(max(abs(A@x-b).max(),(G@x-h).max(),(lb-x).max(),(x-ub).max(),0)))
  rows.append(row);print(row,flush=True)
(p/'twelve_pelvis_authority_audit.json').write_text(json.dumps(rows,indent=2)+'\n')
