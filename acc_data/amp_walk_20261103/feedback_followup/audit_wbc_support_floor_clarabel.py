"""Saved instantaneous WBC QPs only: test a sole-support normal-load floor."""
from pathlib import Path
import json,numpy as np,clarabel
from scipy import sparse
p=Path(__file__).parent;reports=[]
for time in [3.9,4.,4.04,4.06]:
 d=np.load(p/f'fixed_clock120_body10_constraint_audit.snapshot_{time:.3f}.npz');H=d['problem_H'][0].astype(float);g=d['problem_h'][0].astype(float);G=d['problem_G'][0].astype(float);h=d['problem_b'][0].astype(float);A=d['problem_A_eq'][0].astype(float);b=d['problem_b_eq'][0].astype(float);lb=d['problem_lb'][0].astype(float);ub=d['problem_ub'][0].astype(float);n=len(g);mode=int(d['contact_mode'].item());side=0 if mode==1 else 1
 if mode not in [1,2]:continue
 cases=[]
 for minimum in [0.,100.,150.,200.]:
  row=np.zeros((1,n));row[0,37+6*side]=-1.;C=np.vstack((G,row));bound=np.r_[h,-minimum]
  As=np.maximum(np.linalg.norm(A,axis=1),1e-8);Gs=np.maximum(np.linalg.norm(C,axis=1),1e-8)
  upper=np.isfinite(ub);lower=np.isfinite(lb);I=np.eye(n)
  M=sparse.csc_matrix(np.vstack((A/As[:,None],C/Gs[:,None],I[upper],-I[lower])))
  rhs=np.r_[b/As,bound/Gs,ub[upper],-lb[lower]]
  settings=clarabel.DefaultSettings();settings.verbose=False;settings.max_iter=200;settings.tol_feas=1e-9;settings.tol_gap_abs=1e-9;settings.tol_gap_rel=1e-9
  sol=clarabel.DefaultSolver(sparse.triu(sparse.csc_matrix((H+H.T)/2),format='csc'),g,M,rhs,[clarabel.ZeroConeT(len(A)),clarabel.NonnegativeConeT(len(rhs)-len(A))],settings).solve();r=dict(minimum_force=minimum,status=str(sol.status))
  if str(sol.status) in ['Solved','AlmostSolved']:
   x=np.asarray(sol.x);com=d['task_center_of_mass_A'][0]@x[:35]+d['task_center_of_mass_bias'][0]
   r.update(residual=float(max(abs(A@x-b).max(),(C@x-bound).max(),(lb-x).max(),(x-ub).max())),support_force=float(x[37+6*side]),com_acceleration=com.tolist(),base_angular_acceleration=x[3:6].tolist(),acceleration_change=float(abs(x[:35]-d['predicted_acc'][0]).max()))
  cases.append(r)
 reports.append(dict(time=time,mode=mode,cases=cases))
(p/'wbc_support_floor_clarabel_audit.json').write_text(json.dumps(reports,indent=2)+'\n');print(json.dumps(reports,indent=2))
