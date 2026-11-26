"""Independent feasibility diagnostics; removed rows are never executed."""
from pathlib import Path
import json,numpy as np
from scipy.optimize import linprog
p=Path(__file__).parent;d=np.load(p/'upper_braking_before12.failure.npz');prefix='priority_problem_';get=lambda k:d[prefix+k][0].astype(float)
E,e,G,b,lb,ub=[get(k) for k in ['A_eq','b_eq','G','b','lb','ub']];lo=d['physical_acc_lower'][0];hi=d['physical_acc_upper'][0];groups={name:[] for name in ['waist_yaw','waist_roll','waist_pitch','arms']}
for i,row in enumerate(G):
 nz=np.flatnonzero(abs(row)>1e-9)
 if len(nz)!=1 or not 18<=nz[0]<35:continue
 j=int(nz[0]);a=row[j]
 if abs(a-1)<1e-7 and np.isfinite(hi[j]) and abs(b[i]-hi[j])<1e-4 or abs(a+1)<1e-7 and np.isfinite(lo[j]) and abs(b[i]+lo[j])<1e-4:
  groups[['waist_yaw','waist_roll','waist_pitch'][j-18] if j<21 else 'arms'].append(i)
rows=[]
for removed in [[],['arms'],['waist_yaw'],['waist_pitch'],['waist_roll'],['arms','waist_yaw','waist_pitch'],list(groups)]:
 keep=np.ones(len(G),dtype=bool)
 for name in removed:keep[groups[name]]=False
 r=linprog(np.zeros(len(lb)),A_ub=G[keep],b_ub=b[keep],A_eq=E,b_eq=e,bounds=list(zip(lb,ub)),method='highs')
 row=dict(removed_braking_groups=removed,success=bool(r.success),status=int(r.status),remaining_rows=int(keep.sum()));rows.append(row);print(row)
(p/'upper_braking_failure_audit.json').write_text(json.dumps(dict(scope='Counterfactual diagnostics only; no infeasible solution or removed-constraint variant was applied.',braking_row_groups=groups,variants=rows),indent=2)+'\n')
