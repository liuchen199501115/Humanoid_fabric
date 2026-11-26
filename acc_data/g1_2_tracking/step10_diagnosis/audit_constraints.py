"""Diagnostic removals only; never apply removed constraints to a controller."""
from pathlib import Path
import numpy as np,json
from scipy.optimize import linprog
out=Path(__file__).resolve().parent;rows=[]
for p in sorted(out.glob('baseline_probe_v2.qp_*.npz')):
 with np.load(p) as d:
  get=lambda k:d['priority_level_0_'+k][0].astype(float)
  E,e,G,b,lo,hi=[get(k) for k in ['A_eq','b_eq','G','b','lb','ub']];n=len(lo)
  A=np.c_[d['task_center_of_mass_A'][0],np.zeros((3,n-35))]
  target=d['task_center_of_mass_desired'][0]-d['task_center_of_mass_bias'][0]+d['task_center_of_mass_A'][0]@d['correction'][0]
  for case in ['all','without_auxiliary_tubes','without_collision_rows','without_torque_bounds','without_contact_kinematics']:
   keep=np.ones(len(G),dtype=bool);bounds=np.c_[lo,hi];eq=E;rhs=e
   if case=='without_auxiliary_tubes':keep[-3:]=False
   if case=='without_collision_rows':
    first=np.where(np.any(abs(G[:,35:])>1e-9,axis=1))[0].max()+1;keep[first:-3]=False
   if case=='without_torque_bounds':bounds[47:]=[-1e6,1e6]
   if case=='without_contact_kinematics':eq=E[:35];rhs=e[:35]
   r=linprog(np.zeros(n),A_ub=G[keep],b_ub=b[keep],A_eq=np.vstack((eq,A)),b_eq=np.r_[rhs,target],bounds=bounds,method='highs')
   rows.append(dict(time_s=float(d['time']),case=case,status=int(r.status),message=r.message))
(out/'constraint_audit.json').write_text(json.dumps(dict(scope='Offline causal probes only; diagnostic removals are not controller changes.',cases=rows),indent=2)+'\n')
for r in rows:print(r['time_s'],r['case'],r['status'])
