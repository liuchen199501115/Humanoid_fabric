"""Independent saved-state feasibility checks; never apply counterfactual controls."""
from pathlib import Path
import json
import numpy as np
from scipy.optimize import linprog
out=Path(__file__).resolve().parent
results=[]
for path in sorted(out.glob('*.failure.npz')):
 with np.load(path) as data:
  prefix='priority_problem_' if 'priority_problem_G' in data else 'problem_'
  if prefix+'G' not in data:continue
  get=lambda k:data[prefix+k][0].astype(float)
  E,e,G,b,lower,upper=[get(k) for k in ('A_eq','b_eq','G','b','lb','ub')]
  record=dict(run=path.name.removesuffix('.failure.npz'),time_s=float(data['control_time']),cases=[])
  for relax_orientation in (False,True):
   if relax_orientation:
    row=G[-1].copy();angular_nonzero=np.any(abs(row[3:6])>1e-12);row[3:6]=0
    if not angular_nonzero or np.any(abs(row)>1e-12):continue
    GG,bb=G[:-1],b[:-1]
   else:GG,bb=G,b
   r=linprog(np.zeros(len(lower)),A_ub=GG,b_ub=bb,A_eq=E,b_eq=e,bounds=list(zip(lower,upper)),method='highs')
   case=dict(auxiliary_orientation_row_removed=relax_orientation,status=int(r.status),message=r.message)
   if r.success:
    case['max_residual']=float(max(np.abs(E@r.x-e).max(),(GG@r.x-bb).max(),(lower-r.x).max(),(r.x-upper).max(),0))
   record['cases'].append(case)
  results.append(record)
(out/'constraint_audit.json').write_text(json.dumps(dict(scope='Offline feasibility counterfactual only. All physical and state braking rows retained; optional removal concerns only the final auxiliary orientation tube.',trials=results),indent=2)+'\n')
print(json.dumps(results,indent=2))
