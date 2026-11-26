"""Offline linear feasibility bounds at recorded states; never used by runtime control."""
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
OUT=Path(__file__).resolve().parent;rows=[]
for stamp in ['4.300','4.400','4.500']:
 a=np.load(OUT/('eleven_continuity1000.snapshot_'+stamp+'.npz'));G=a['soft_problem_G'][0].astype(float);b=a['soft_problem_b'][0].astype(float);E=a['soft_problem_A_eq'][0].astype(float);e=a['soft_problem_b_eq'][0].astype(float);lo=a['soft_problem_lb'][0].astype(float);hi=a['soft_problem_ub'][0].astype(float);n=len(lo);c=np.zeros(n);c[:35]=a['composed_task_center_of_mass_A'][0,1]
 for label in ['original','without_acceleration_barriers','without_motor_limits']:
  mask=np.ones(len(G),dtype=bool);bounds=list(zip(lo,hi))
  if label=='without_acceleration_barriers':mask&=np.any(abs(G[:,35:47])>0,axis=1)|np.any(abs(G[:,47:])>0,axis=1)
  if label=='without_motor_limits':bounds[47:76]=[(-1e6,1e6)]*29
  r=linprog(c,A_ub=G[mask],b_ub=b[mask],A_eq=E,b_eq=e,bounds=bounds,method='highs')
  row=dict(time=float(stamp),variant=label,success=r.success,message=r.message)
  if r.success:
   row.update(minimum_com_y_acceleration=float(r.fun+a['composed_task_center_of_mass_bias'][0,1]),requested_com_y_acceleration=float(a['composed_task_center_of_mass_desired'][0,1]),active_inequality_rows=np.flatnonzero(abs(G@r.x-b)<1e-5).tolist(),lower_bound_indices=np.flatnonzero(abs(r.x-lo)<1e-5).tolist(),upper_bound_indices=np.flatnonzero(abs(r.x-hi)<1e-5).tolist(),equality_error=float(np.max(abs(E@r.x-e))),inequality_error=float(np.max(G[mask]@r.x-b[mask])),root_acceleration=r.x[:6].tolist())
  rows.append(row);print(row,flush=True)
(OUT/'lateral_feasibility.json').write_text(json.dumps(rows,indent=2)+'\n')
