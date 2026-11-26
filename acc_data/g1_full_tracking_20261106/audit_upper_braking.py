"""Saved-state feasibility of existing measured upper-joint braking bounds."""
import json
from pathlib import Path
import numpy as np,torch
from scipy.optimize import linprog
from fabrics_sim.solver.joint_control import measured_joint_acceleration_bounds
p=Path(__file__).parent;rows=[]
for name in ['rolling_contact_damping8.qp_2150.npz','pelvis_replace_before_turn.qp_1900.npz','root_margin150_turn.qp_2000.npz']:
 d=np.load(p/name);record=np.load(p/(name.split('.qp_')[0]+'.npz'));q=torch.tensor(d['q'],dtype=torch.float64);v=torch.tensor(d['qd'],dtype=torch.float64);limits=torch.tensor(record['joint_position_limits'],dtype=torch.float64)
 try:
  lo,hi=measured_joint_acceleration_bounds(q[:,19:],v[:,18:],limits[:,12:,0],limits[:,12:,1],.002,barrier_bandwidth=8.)
 except ValueError as error:
  row=dict(snapshot=name,time=float(d['time']),feasible=False,bound_construction_failure=str(error));rows.append(row);print(json.dumps(row),flush=True);continue
 prefix='priority_level_0_';get=lambda k:d[prefix+k][0].astype(float)
 E,e,G,b,lb,ub=[get(k) for k in ['A_eq','b_eq','G','b','lb','ub']]
 lb[18:35]=np.maximum(lb[18:35],lo[0]);ub[18:35]=np.minimum(ub[18:35],hi[0])
 r=linprog(np.zeros(len(lb)),A_ub=G,b_ub=b,A_eq=E,b_eq=e,bounds=list(zip(lb,ub)),method='highs')
 row=dict(snapshot=name,time=float(d['time']),feasible=bool(r.success),status=int(r.status),waist_position=q[0,19:22].tolist(),waist_velocity=v[0,18:21].tolist(),waist_acc_lower=lo[0,:3].tolist(),waist_acc_upper=hi[0,:3].tolist())
 if r.success:row['constraint_residual']=float(max(abs(E@r.x-e).max(),(G@r.x-b).max(),(lb-r.x).max(),(r.x-ub).max(),0.))
 rows.append(row);print(json.dumps(row),flush=True)
(p/'upper_braking_feasibility.json').write_text(json.dumps(rows,indent=2)+'\n')
