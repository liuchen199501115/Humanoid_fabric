"""Offline step13 cost sensitivity, retaining the frozen physical constraints."""
import json
from pathlib import Path
import numpy as np, torch
from fabrics_sim.solver.admm_qp import QPData
from fabrics_sim.solver.daqp_backend import DAQPBackend
OUT=Path(__file__).resolve().parent
rows=[]
for stamp in ['4.740','4.780','4.800']:
 a=np.load(OUT/f'twelve_pelvis1.snapshot_{stamp}.npz')
 for label,factor,task in [('original',1,None),('tube100',.1,None),('tube10',.01,None),('tube1',.001,None),('posture4',4,'joint_posture'),('com10',10,'center_of_mass'),('support01',.1,'support_orientation_0')]:
  qp=QPData(**{k:torch.tensor(a['soft_problem_'+k]).double() for k in ['H','h','G','b','lb','ub','A_eq','b_eq']})
  if task is None: qp.H[:,76:,76:]*=factor
  else:
   prefix='composed_task_'+task;A=torch.tensor(a[prefix+'_A']).double();active=torch.tensor(a[prefix+'_active']);M=torch.tensor(a[prefix+'_metric']).double()*active[:,:,None]*active[:,None,:];d=torch.tensor(a[prefix+'_bias']-a[prefix+'_desired']).double();pull=A.transpose(1,2)@M
   qp.H[:,:35,:35]+=(factor-1)*pull@A;qp.h[:,:35]+=(factor-1)*(pull@d[...,None]).squeeze(-1)
  solver=DAQPBackend(50000,single_solve=True,single_proximal=.0001,equilibration_cap=1);result=solver.solve(qp);z=result.z[0].numpy()
  row=dict(time=float(stamp),variant=label,certified=bool(result.status_ok.all()),waist_acceleration=z[18:21].tolist(),shoulder_pitch_acceleration=z[[21,28]].tolist(),slack=z[76:].tolist(),support_loads=z[[37,43]].tolist())
  rows.append(row);print(row,flush=True)
(OUT/'step13_objective_sensitivity.json').write_text(json.dumps(rows,indent=2)+'\n')
