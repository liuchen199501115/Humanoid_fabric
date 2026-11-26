"""Independent LP diagnosis of the restricted fixed-momentum COM fit."""
from pathlib import Path
import numpy as np,json
from scipy import sparse
from scipy.optimize import linprog
p=Path(__file__).parent;d=np.load(p/'wbc_remaining_slow1.problem.npz');C=sparse.csc_matrix((d['data'],d['indices'],d['indptr']),shape=tuple(d['shape']));lo=d['low'];hi=d['high'];eq=np.isfinite(lo)&(lo==hi);n=C.shape[1];rows=[]
for mode in ['physical','minimum_friction_excess','without_friction']:
 M=C;high=hi.copy();bounds=[(None,None)]*n;objective=np.zeros(n);select=~eq
 if mode=='minimum_friction_excess':
  extra=np.zeros(len(hi));extra[d['friction_rows']]=-d['friction_g'];M=sparse.hstack((C,extra[:,None]),format='csc');bounds.append((0,None));objective=np.r_[np.zeros(n),1.]
 if mode=='without_friction':select=select.copy();select[d['friction_rows']]=False
 result=linprog(objective,A_ub=M[select],b_ub=high[select],A_eq=M[eq],b_eq=lo[eq],bounds=bounds,method='highs',options={'primal_feasibility_tolerance':1e-8,'dual_feasibility_tolerance':1e-8})
 row=dict(mode=mode,status=result.message,feasible=bool(result.success),physical_tracking_certified=False)
 if result.success:
  row['residual']=float(max(0.,abs(M[eq]@result.x-lo[eq]).max(),(M[select]@result.x-high[select]).max()))
  if mode=='minimum_friction_excess':row['additional_friction_required']=float(result.x[-1])
 rows.append(row);print(row,flush=True)
(p/'remaining_com_feasibility_audit.json').write_text(json.dumps(rows,indent=2)+'\n')
