"""Independent physical lateral-acceleration feasibility; no solutions executed."""
from pathlib import Path
import json
import numpy as np
from scipy.optimize import linprog
p=Path(__file__).parent
paths=list(p.glob('lateral_com_first_before_turn_v2.qp_*.npz'))+[
 p/'rolling_contact_damping8.qp_2150.npz',p/'pelvis_replace_before_turn.qp_1900.npz',
 p/'root_margin150_turn.qp_2000.npz',p/'weighted_after12.qp_2150.npz']
rows=[]
for path in paths:
 if not path.exists():continue
 d=np.load(path)
 prefix=next(k for k in ['priority_level_0_','priority_problem_','problem_'] if k+'H' in d)
 get=lambda k:d[prefix+k][0].astype(float)
 E,e,G,b,lb,ub=[get(k) for k in ['A_eq','b_eq','G','b','lb','ub']]
 n=len(lb);c=np.zeros(n);c[:35]=d['task_center_of_mass_A'][0,1]
 bias=float(d['task_center_of_mass_bias'][0,1]);desired=float(d['task_center_of_mass_desired'][0,1])
 row=dict(snapshot=path.name,time=float(d['time']),frame=float(d['motion_frame_position']),
  constraint_source=prefix,desired=desired,
  actual_solution=float(c[:35]@d['predicted_acc'][0]+bias),bounds=[])
 for sign in [1.,-1.]:
  r=linprog(sign*c,A_ub=G,b_ub=b,A_eq=E,b_eq=e,bounds=list(zip(lb,ub)),method='highs')
  row['bounds'].append(dict(success=bool(r.success),status=int(r.status),
   acceleration=float(c@r.x+bias) if r.success else None))
 r=linprog(np.zeros(n),A_ub=G,b_ub=b,A_eq=np.vstack((E,c)),b_eq=np.r_[e,desired-bias],bounds=list(zip(lb,ub)),method='highs')
 row['desired_acceleration_feasible']=bool(r.success)
 if r.success:
  row['constraint_residual']=float(max(abs(E@r.x-e).max(),(G@r.x-b).max(),(lb-r.x).max(),(r.x-ub).max(),abs(c@r.x+bias-desired),0.))
 rows.append(row);print(json.dumps(row),flush=True)
(p/'lateral_authority_audit.json').write_text(json.dumps(dict(scope='Instantaneous acceleration and saved constraints only; does not certify future steps. Direct WBC snapshots only; no diagnostic solution applied.',samples=rows),indent=2)+'\n')
