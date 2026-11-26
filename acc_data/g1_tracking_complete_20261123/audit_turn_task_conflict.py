"""Frozen-state task/barrier ablation; physical constraints remain unchanged.
Not a controller, rollout, or tracking result. Softening auxiliary fidelity
barriers here only diagnoses cost conflict; the source auditor is unchanged.
"""
import json
from pathlib import Path
import numpy as np
import torch
from fabrics_sim.solver.admm_qp import QPData
from fabrics_sim.solver.daqp_backend import DAQPBackend
OUT=Path(__file__).resolve().parent
names=np.load(OUT/'ten_verify.npz')['final_joint_order'].tolist()
records=[]
for timestamp in ['3.800','3.900']:
 a=np.load(OUT/f'ten_com15_ang100_pelvis405.snapshot_{timestamp}.npz')
 fields=['H','h','G','b','lb','ub','A_eq','b_eq']
 original={k:a['soft_problem_'+k].astype(float) for k in fields}
 def task(name,scale):
  prefix='composed_task_'+name+'_';A=a[prefix+'A'][0].astype(float);mask=a[prefix+'active'][0]
  W=a[prefix+'metric'][0].astype(float)*mask[:,None]*mask[None,:]
  if name=='joint_posture':
   factors=np.ones(29);factors[6:12]=np.sqrt(scale)
   delta=W*factors[:,None]*factors[None,:]-W
  else:delta=(scale-1)*W
  return A.T@delta@A,A.T@delta@(a[prefix+'bias'][0]-a[prefix+'desired'][0])
 for case in ['original','stance25','pelvis4','soft_orientation','soft_position','soft_both']:
  p={k:v.copy() for k,v in original.items()}
  if case in ['stance25','pelvis4']:
   H,h=task('joint_posture' if case=='stance25' else 'source_pelvis_orientation',.25 if case=='stance25' else 4.)
   p['H'][0,:35,:35]+=H;p['h'][0,:35]+=h
  if case in ['soft_orientation','soft_both']:p['ub'][0,-1]=np.inf
  if case in ['soft_position','soft_both']:p['ub'][0,-2]=np.inf
  solver=DAQPBackend(single_solve=True,single_proximal=.0001,equilibration_cap=10.)
  solution=solver.solve(QPData(**{k:torch.tensor(v) for k,v in p.items()}))
  r=dict(time_s=float(timestamp),case=case,certificate_passed=bool(solution.status_ok.all()))
  if r['certificate_passed']:
   z=solution.z.numpy()[0];acc=z[:35];com='composed_task_center_of_mass_'
   r.update(root_acceleration=acc[:6].tolist(),com_acceleration=(a[com+'A'][0]@acc+a[com+'bias'][0]).tolist(),
    right_knee_acceleration=float(acc[6+names.index('right_knee_joint')]),waist_roll_acceleration=float(acc[6+names.index('waist_roll_joint')]),tube_slack=z[-3:].tolist(),
    reproduced_acceleration_error=float(np.max(abs(acc-a['predicted_acc'][0]))) if case=='original' else None)
  records.append(r)
(OUT/'turn_task_conflict.json').write_text(json.dumps(dict(scope=__doc__,cases=records),indent=2)+'\n')
for r in records:print(r,flush=True)
