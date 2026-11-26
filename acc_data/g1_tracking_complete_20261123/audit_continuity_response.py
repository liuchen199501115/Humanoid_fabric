"""Offline frozen-state objective sensitivity; each variant uses one native solve."""
import json
from pathlib import Path
import numpy as np,torch
from fabrics_sim.solver.admm_qp import QPData
from fabrics_sim.solver.daqp_backend import DAQPBackend
OUT=Path(__file__).resolve().parent;history=np.load(OUT/'eleven_continuity1000.npz');rows=[]
for stamp in ['4.300','4.400','4.500']:
 a=np.load(OUT/('eleven_continuity1000.snapshot_'+stamp+'.npz'));i=int(np.argmin(abs(history['history_time']-float(stamp))));previous=history['history_torque_qp_total'][i-1,0].astype(float)
 for label,continuity,task,factor in [('original',1000,None,1),('continuity100',100,None,1),('continuity0',0,None,1),('com4',1000,'center_of_mass',4),('rooty10',1000,'root_position',10),('feet4',1000,'feet_pose',4),('body4',1000,'body_orientation',4)]:
  qp=QPData(**{k:torch.tensor(a['soft_problem_'+k]).double() for k in ['H','h','G','b','lb','ub','A_eq','b_eq']})
  qp.H[:,47:76,47:76]+=(continuity-1000)*torch.eye(29,dtype=torch.float64);qp.h[:,47:76]-=(continuity-1000)*torch.tensor(previous)
  if task:
   prefix='composed_task_'+task;A=torch.tensor(a[prefix+'_A']).double();active=torch.tensor(a[prefix+'_active']);M=torch.tensor(a[prefix+'_metric']).double()*active[:,:,None]*active[:,None,:];d=torch.tensor(a[prefix+'_bias']-a[prefix+'_desired']).double();pull=A.transpose(1,2)@M
   qp.H[:,:35,:35]+=(factor-1)*(pull@A);qp.h[:,:35]+=(factor-1)*(pull@d[...,None]).squeeze(-1)
  solver=DAQPBackend(50000,single_solve=True,single_proximal=.0001,equilibration_cap=1);result=solver.solve(qp);z=result.z[0].numpy();row=dict(time=float(stamp),variant=label,certified=bool(result.status_ok.all()),root_acceleration=z[:6].tolist(),force=z[35:38].tolist(),torque_step_norm=float(np.linalg.norm(z[47:76]-previous)))
  for name in ['center_of_mass','feet_pose','body_orientation','root_position']:
   prefix='composed_task_'+name;res=(a[prefix+'_A'][0]@z[:35]+a[prefix+'_bias'][0]-a[prefix+'_desired'][0])*a[prefix+'_active'][0];row[name+'_residual']=res.tolist()
  rows.append(row);print(stamp,label,row['certified'],'comYres',row['center_of_mass_residual'][1],'rootYres',row['root_position_residual'][1],flush=True)
(OUT/'continuity_objective_sensitivity.json').write_text(json.dumps(rows,indent=2)+'\n')
