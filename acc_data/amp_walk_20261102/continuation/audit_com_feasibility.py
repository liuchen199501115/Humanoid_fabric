"""Offline QP feasibility; retains all recorded physical constraints."""
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
p=Path('acc_data/amp_walk_20261102/continuation')
reports=[]
for path in sorted(p.glob('baseline.qp_*.npz')):
 d=np.load(path);get=lambda k:d[k][0].astype(float)
 A=get('priority_level_0_A_eq');b=get('priority_level_0_b_eq');G=get('priority_level_0_G');h=get('priority_level_0_b')
 bounds=list(zip(get('priority_level_0_lb'),get('priority_level_0_ub')))
 C=np.zeros((3,76));C[:,:35]=get('task_center_of_mass_A')
 target=get('task_center_of_mass_desired')-get('task_center_of_mass_bias')
 x=get('problem_solution');ref=get('reference_q');q=get('q');qd=get('qd')
 row=dict(time=float(d['time']),requested_com_acc=get('task_center_of_mass_desired').tolist(),
 predicted_com_acc=(C@x+get('task_center_of_mass_bias')).tolist(),tests={})
 for name,indices in [('xyz',[0,1,2]),('xy',[0,1]),('y',[1])]:
  extra=C[indices];rhs=target[indices]
  r=linprog(np.zeros(76),A_ub=G,b_ub=h,A_eq=np.r_[A,extra],b_eq=np.r_[b,rhs],bounds=bounds,method='highs')
  row['tests'][name]={'feasible': True if r.success else (False if r.status == 2 else None), 'solver_status_code': int(r.status), 'status':r.message}
  if r.success:row['tests'][name]['max_acceleration']=float(abs(r.x[:35]).max())
 print(json.dumps(row));reports.append(row)
(p/'com_feasibility.json').write_text(json.dumps(reports,indent=2)+'\n')
