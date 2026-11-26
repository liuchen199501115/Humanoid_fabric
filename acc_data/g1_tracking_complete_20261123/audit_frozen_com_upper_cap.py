"""Frozen-state soft COM target sensitivity; no runtime retries or constraints removed."""
import json,sys
from pathlib import Path
import numpy as np
import torch
from fabrics_sim.solver.admm_qp import QPData
from fabrics_sim.solver.daqp_backend import DAQPBackend
OUT=Path(__file__).resolve().parent;name=sys.argv[1];records=[]
for stamp in (3.8,3.9):
 a=np.load(OUT/(name+f'.snapshot_{stamp:.3f}.npz'));prefix='soft_problem_' if 'soft_problem_H' in a else 'problem_'
 original={k:a[prefix+k].astype(float) for k in ['H','h','G','b','lb','ub','A_eq','b_eq']}
 A=a['composed_task_center_of_mass_A'][0].astype(float);bias=a['composed_task_center_of_mass_bias'][0].astype(float)
 desired=a['composed_task_center_of_mass_desired'][0].astype(float);W=a['composed_task_center_of_mass_metric'][0].astype(float);mask=a['composed_task_center_of_mass_active'][0];W*=mask[:,None]*mask[None,:]
 for maximum in (None,6.,3.,0.):
  for scale in (1.,4.,16.):
   new=desired.copy()
   if maximum is not None:new[2]=min(new[2],maximum)
   Wnew=W.copy();Wnew[2,2]*=scale
   p={k:v.copy() for k,v in original.items()};p['H'][0,:35,:35]+=A.T@(Wnew-W)@A
   p['h'][0,:35]+=A.T@(Wnew@(bias-new)-W@(bias-desired))
   solver=DAQPBackend(single_solve=True,single_proximal=.0001,equilibration_cap=1. if stamp>3.806 else 10.)
   fit=solver.solve(QPData(**{k:torch.tensor(v) for k,v in p.items()}));valid=bool(fit.status_ok.all())
   row=dict(time=stamp,maximum=maximum,additional_z_weight=scale,valid=valid,original_com_target=desired.tolist())
   if valid:
    acc=fit.z.numpy()[0,:35];row.update(com_acceleration=(A@acc+bias).tolist(),root_acceleration=acc[:6].tolist())
   records.append(row)
(OUT/(name+'.frozen_com_upper_cap.json')).write_text(json.dumps(records,indent=2)+'\n')
for r in records:print(r,flush=True)
