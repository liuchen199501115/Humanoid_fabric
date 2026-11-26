"""Instantaneous saved-QP support-orientation priority audit (no horizon)."""
from pathlib import Path
import json
import numpy as np
import clarabel
from scipy import sparse
p=Path(__file__).parent
reports=[]
for t in [3.9,4.,4.04]:
 d=np.load(p/f'fixed_clock120_body10_constraint_audit.snapshot_{t:.3f}.npz')
 H0=d['problem_H'][0].astype(float);g0=d['problem_h'][0].astype(float)
 A=d['problem_A_eq'][0].astype(float);b=d['problem_b_eq'][0].astype(float)
 G=d['problem_G'][0].astype(float);h=d['problem_b'][0].astype(float)
 lb=d['problem_lb'][0];ub=d['problem_ub'][0];n=len(g0)
 an=np.maximum(np.linalg.norm(A,axis=1),1e-8);gn=np.maximum(np.linalg.norm(G,axis=1),1e-8)
 iu=np.isfinite(ub);il=np.isfinite(lb);eye=np.eye(n)
 M=sparse.csc_matrix(np.vstack([A/an[:,None],G/gn[:,None],eye[iu],-eye[il]]))
 rhs=np.r_[b/an,h/gn,ub[iu],-lb[il]]
 names=[k.removeprefix('task_').removesuffix('_A') for k in d.files if k.startswith('task_support_orientation_') and k.endswith('_A')]
 for weight in [0.,4.]:
  H=H0.copy();g=g0.copy()
  for name in names:
   J=d[f'task_{name}_A'][0];B=d[f'task_{name}_bias'][0];goal=d[f'task_{name}_desired'][0]
   active=d[f'task_{name}_active'][0];W=d[f'task_{name}_metric'][0]*active[:,None]*active[None,:]
   H[:35,:35]+=weight*J.T@W@J;g[:35]+=weight*J.T@W@(B-goal)
  settings=clarabel.DefaultSettings();settings.verbose=False;settings.max_iter=200
  settings.tol_feas=settings.tol_gap_abs=settings.tol_gap_rel=1e-9
  sol=clarabel.DefaultSolver(sparse.triu(sparse.csc_matrix((H+H.T)/2),format='csc'),g,M,rhs,[clarabel.ZeroConeT(len(A)),clarabel.NonnegativeConeT(len(rhs)-len(A))],settings).solve()
  row=dict(time=t,added_support_weight=weight,status=str(sol.status))
  if str(sol.status) in ['Solved','AlmostSolved']:
   x=np.array(sol.x);row.update(residual=float(max(abs(A@x-b).max(),(G@x-h).max(),(lb-x).max(),(x-ub).max(),0)),com_acceleration=(d['task_center_of_mass_A'][0]@x[:35]+d['task_center_of_mass_bias'][0]).tolist(),root_angular_acceleration=x[3:6].tolist(),max_acceleration_difference=float(abs(x[:35]-d['predicted_acc'][0]).max()),support_tasks=[])
   for name in names:
    row['support_tasks'].append(dict(name=name,active=d[f'task_{name}_active'][0].tolist(),desired=d[f'task_{name}_desired'][0].tolist(),achieved=(d[f'task_{name}_A'][0]@x[:35]+d[f'task_{name}_bias'][0]).tolist()))
  reports.append(row)
(p/'wbc_support_priority_audit.json').write_text(json.dumps(reports,indent=2)+'\n')
print(json.dumps(reports,indent=2))
