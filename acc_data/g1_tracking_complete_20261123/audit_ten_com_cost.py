"""Offline objective comparisons on the same certified swing-phase QP."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from fabrics_sim.solver.daqp_backend import DAQPBackend
OUT=Path(__file__).resolve().parent;records=[]
for path in sorted(OUT.glob('ten_momentum_late.snapshot_*.npz')):
 d=np.load(path);base={k:d['soft_problem_'+k].astype(float) for k in ['H','h','G','b','lb','ub','A_eq','b_eq']}
 tasks={};cost=[];x=d['soft_problem_solution'][0,:35]
 for key in d.files:
  if not key.startswith('composed_task_') or not key.endswith('_A'):continue
  name=key[len('composed_task_'):-2];t={k:d['composed_task_'+name+'_'+k][0].astype(float) for k in ['A','bias','desired','metric','active']};W=t['metric']*t['active'][:,None]*t['active'][None,:];t['H']=t['A'].T@W@t['A'];t['h']=t['A'].T@W@(t['bias']-t['desired']);e=t['A']@x+t['bias']-t['desired'];cost.append(dict(name=name,cost=float(.5*e@W@e)));tasks[name]=t
 record=dict(time_s=float(d['time']),costs=sorted(cost,key=lambda row:-row['cost']),cases=[])
 for name,scales in [('original',{}),('foot025',{'feet_pose':.25}),('body4',{'body_orientation':4}),('com2',{'center_of_mass':2}),('com8',{'center_of_mass':8}),('posture025',{'joint_posture':.25})]:
  a={k:v.copy() for k,v in base.items()}
  for task,scale in scales.items():a['H'][0,:35,:35]+=(scale-1)*tasks[task]['H'];a['h'][0,:35]+=(scale-1)*tasks[task]['h']
  solver=DAQPBackend(single_solve=True,single_proximal=.0001,equilibration_cap=10);r=solver.solve(SimpleNamespace(**{k:torch.tensor(v) for k,v in a.items()}));qdd=r.z[0,:35].numpy();foot=tasks['feet_pose'];record['cases'].append(dict(name=name,valid=bool(r.converged.item()),com_acc=(tasks['center_of_mass']['A']@qdd+tasks['center_of_mass']['bias']).tolist(),root_angular_acc=qdd[3:6].tolist(),left_foot_acc=(foot['A']@qdd+foot['bias'])[:3].tolist(),right_foot_acc=(foot['A']@qdd+foot['bias'])[6:9].tolist(),right_foot_desired=foot['desired'][6:9].tolist(),right_shoulder_roll_qdd=float(qdd[29]),right_hip_roll_qdd=float(qdd[13])))
 records.append(record)
(OUT/'ten_com_cost.json').write_text(json.dumps(records,indent=2));print(json.dumps(records,indent=2))
