"""Offline row ablations, never used to apply an uncertified runtime command."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from fabrics_sim.solver.daqp_backend import DAQPBackend

OUT=Path(__file__).resolve().parent
h=np.load(OUT/'eight_snapshot.npz');records=[]
for path in sorted(OUT.glob('eight_snapshot.snapshot_*.npz')):
 d=np.load(path);arrays={k:d['soft_problem_'+k].copy() for k in ['H','h','G','b','lb','ub','A_eq','b_eq']}
 first=int(np.flatnonzero(np.any(abs(arrays['G'][0,:,:35])>1e-12,axis=1))[0])
 i=abs(h['history_time']-float(d['time'])).argmin();record=dict(time_s=float(d['time']),cases={})
 for kind in ['all','without_collision','without_joint_tube','without_position_tube','without_orientation_tube','without_all_tubes','without_contact_kinematics','without_torque_bounds']:
  a={k:v.copy() for k,v in arrays.items()};mask=np.ones(a['G'].shape[1],dtype=bool)
  if kind=='without_collision':mask[first:-61]=False
  if kind=='without_joint_tube':mask[-3]=False
  if kind=='without_position_tube':mask[-2]=False
  if kind=='without_orientation_tube':mask[-1]=False
  if kind=='without_all_tubes':mask[-3:]=False
  if kind=='without_contact_kinematics':a['A_eq']=a['A_eq'][:,:35];a['b_eq']=a['b_eq'][:,:35]
  if kind=='without_torque_bounds':a['lb'][:,47:76]=-np.inf;a['ub'][:,47:76]=np.inf
  a['G']=a['G'][:,mask];a['b']=a['b'][:,mask]
  solver=DAQPBackend(single_solve=True,single_proximal=.0001)
  qp=SimpleNamespace(**{k:torch.tensor(v,dtype=torch.float64) for k,v in a.items()})
  r=solver.solve(qp);z=r.z.numpy()[0]
  record['cases'][kind]=dict(valid=bool(r.converged.item()),status=solver.last_status,
   right_shoulder_roll_qdd=float(z[29]),left_shoulder_roll_qdd=float(z[22]),
   left_hip_roll_qdd=float(z[7]),left_ankle_roll_qdd=float(z[11]),
   joint_acceleration_distance_from_fabric=float(np.linalg.norm(z[6:35]-h['history_qdd_fabric'][i,0,6:])),
   tracking_tube_slack=z[76:].tolist())
  if kind=='all':
   record['reconstructed_solution_max_difference']=float(abs(z-d['soft_problem_solution'][0]).max())
 records.append(record)
(OUT/'post_landing_qp.json').write_text(json.dumps(records,indent=2)+'\n')
print(json.dumps(records,indent=2))
