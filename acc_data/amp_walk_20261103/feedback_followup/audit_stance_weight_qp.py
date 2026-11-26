from pathlib import Path
import json,numpy as np,osqp
from scipy import sparse
p=Path(__file__).parent;d=np.load(p/'fixed_clock120_body10_twelfth_foot4.snapshot_3.980.npz');baseH=d['problem_H'][0].astype(float);baseh=d['problem_h'][0].astype(float);n=len(baseh);A=d['problem_A_eq'][0].astype(float);b=d['problem_b_eq'][0].astype(float);G=d['problem_G'][0].astype(float);h=d['problem_b'][0].astype(float);lo=d['problem_lb'][0];hi=d['problem_ub'][0];I=np.eye(n);up=np.isfinite(hi);low=np.isfinite(lo);G=np.vstack((G,I[up],-I[low]));h=np.r_[h,hi[up],-lo[low]];eqscale=np.maximum(np.linalg.norm(A,axis=1),1e-8);gscale=np.maximum(np.linalg.norm(G,axis=1),1e-8);C=sparse.csc_matrix(np.vstack((A/eqscale[:,None],G/gscale[:,None])));rhs=np.r_[b/eqscale,h/gscale];lower=np.r_[b/eqscale,np.full(len(h),-np.inf)];records=[]
def task(name):
 J=d['task_'+name+'_A'][0].astype(float);metric=d['task_'+name+'_metric'][0].astype(float);act=d['task_'+name+'_active'][0];metric*=act[:,None]*act[None,:];return J.T@metric@J,J.T@metric@(d['task_'+name+'_bias'][0]-d['task_'+name+'_desired'][0])
for name,stance,body in [('baseline',1,1),('stance30',.3,1),('stance10',.1,1),('stance30_pelvis',.3,1)]:
 H=baseH.copy();g=baseh.copy()
 J=d['task_joint_posture_A'][0].astype(float);M=d['task_joint_posture_metric'][0].astype(float);M[6:,:]=0;M[:,6:]=0
 H[:35,:35]+=(stance-1)*J.T@M@J;g[:35]+=(stance-1)*J.T@M@(d['task_joint_posture_bias'][0]-d['task_joint_posture_desired'][0])
 if 'pelvis' in name:
  th,tg=task('source_pelvis_orientation');H[:35,:35]+=th;g[:35]+=tg
 solver=osqp.OSQP();solver.setup(P=sparse.triu(sparse.csc_matrix(H)),q=g,A=C,l=lower,u=rhs,verbose=False,eps_abs=1e-8,eps_rel=1e-8,max_iter=100000,polish=True)
 sol=solver.solve();x=np.array(sol.x);violation=float(max(abs(A@x-b).max(),(G@x-h).max(),0));records.append(dict(case=name,status=str(sol.info.status),maximum_physical_residual=violation,root_angular_acceleration=x[3:6].tolist(),waist_acceleration=x[18:21].tolist()))
(p/'stance_weight_qp_audit.json').write_text(json.dumps(records,indent=2)+'\n');print(json.dumps(records,indent=2))
