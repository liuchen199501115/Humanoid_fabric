"""Frozen paused-step12 task-weight tradeoff, with original physical constraints."""
import json
from pathlib import Path
import numpy as np,torch
from fabrics_sim.solver.admm_qp import QPData
from fabrics_sim.solver.daqp_backend import DAQPBackend
OUT=Path(__file__).resolve().parent;a=np.load(OUT/'eleven_source60_flat64_pause12.snapshot_4.550.npz');rows=[]
for factor in [1,4,16,64,256]:
 qp=QPData(**{k:torch.tensor(a['soft_problem_'+k]).double() for k in ['H','h','G','b','lb','ub','A_eq','b_eq']});pre='composed_task_feet_pose';A=torch.tensor(a[pre+'_A']).double();d=torch.tensor(a[pre+'_bias']-a[pre+'_desired']).double();active=torch.tensor(a[pre+'_active']);M=torch.tensor(a[pre+'_metric']).double()*active[:,:,None]*active[:,None,:];pull=A.transpose(1,2)@M;qp.H[:,:35,:35]+=(factor-1)*pull@A;qp.h[:,:35]+=(factor-1)*(pull@d[...,None]).squeeze(-1)
 solver=DAQPBackend(50000,single_solve=True,single_proximal=.0001,equilibration_cap=1);sol=solver.solve(qp);z=sol.z[0].numpy();r=dict(factor=factor,certified=bool(sol.status_ok.all()),status=solver.last_status,root_acceleration=z[:6].tolist(),peak_motor_torque=float(abs(z[47:76]).max()))
 for name in ['center_of_mass','feet_pose','body_orientation','joint_posture']:
  prefix='composed_task_'+name;r[name+'_residual']=((a[prefix+'_A'][0]@z[:35]+a[prefix+'_bias'][0]-a[prefix+'_desired'][0])*a[prefix+'_active'][0]).tolist()
 rows.append(r);print(factor,r['certified'],'swing',r['feet_pose_residual'][6:9],'com',r['center_of_mass_residual'],flush=True)
(OUT/'step12_swing_weight_sensitivity.json').write_text(json.dumps(rows,indent=2)+'\n')
