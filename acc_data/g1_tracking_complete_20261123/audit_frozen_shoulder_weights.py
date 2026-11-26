"""Offline saved-state sensitivity for the existing shoulder-roll weight option."""
import json,sys
from pathlib import Path
import numpy as np,torch
from fabrics_sim.solver.admm_qp import QPData
from fabrics_sim.solver.daqp_backend import DAQPBackend
OUT=Path(__file__).resolve().parent;name=sys.argv[1];rows=[]
names=np.load(OUT/'ten_verify.npz')['final_joint_order'].tolist();index=names.index('right_shoulder_roll_joint')
for stamp in [3.9,4.0]:
 a=np.load(OUT/(name+f'.snapshot_{stamp:.3f}.npz'));prefix='soft_problem_' if 'soft_problem_H' in a else 'problem_';p0={k:a[prefix+k].astype(float) for k in ['H','h','G','b','lb','ub','A_eq','b_eq']}
 A=a['composed_task_joint_posture_A'][0].astype(float);W=a['composed_task_joint_posture_metric'][0].astype(float);mask=a['composed_task_joint_posture_active'][0];W*=mask[:,None]*mask[None,:]
 r=a['composed_task_joint_posture_bias'][0]-a['composed_task_joint_posture_desired'][0]
 for scale in [1.,4.,16.]:
  factors=np.ones(29);factors[index]=np.sqrt(scale);D=W*(factors[:,None]*factors[None,:]-1);p={k:v.copy() for k,v in p0.items()};p['H'][0,:35,:35]+=A.T@D@A;p['h'][0,:35]+=A.T@D@r
  solver=DAQPBackend(single_solve=True,single_proximal=.0001,equilibration_cap=1.)
  fit=solver.solve(QPData(**{k:torch.tensor(v) for k,v in p.items()}));valid=bool(fit.status_ok.all());row=dict(time=stamp,scale=scale,valid=valid)
  if valid:
   acc=fit.z.numpy()[0,:35];row.update(shoulder_roll_acceleration=float(acc[6+index]),waist_acceleration=acc[18:21].tolist(),root_angular_acceleration=acc[3:6].tolist())
  rows.append(row)
(OUT/(name+'.frozen_shoulder_weights.json')).write_text(json.dumps(rows,indent=2)+'\n')
for row in rows:print(row,flush=True)
