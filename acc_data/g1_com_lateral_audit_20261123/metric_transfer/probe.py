"""Offline frozen-state diagnosis. Does not modify or run the controller."""
import json
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import linprog
from fabrics_sim.solver.admm_qp import QPData
from fabrics_sim.solver.daqp_backend import DAQPBackend
ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
BASE=ROOT/'acc_data/g1_projected_acc_qp_20261114/continue_20261114_0338'
with np.load(BASE/'baseline_snapshots.qp_0677.npz') as h:
 times=h['history_time']; pd=h['history_com_target_acceleration'][:,0,1]+np.clip(-35*(h['history_com'][:,0,1]-h['history_com_target'][:,0,1])-12*(h['history_com_velocity'][:,0,1]-h['history_com_target_velocity'][:,0,1]),-4,4)
with np.load(ROOT/'acc_data/g1_projected_acc_qp_20261114/continue_knee_20261114/baseline_final.failure.npz') as h:
 names=h['joint_names'].tolist()
reports=[]
for step in [600,650,670,677]:
 with np.load(BASE/f'baseline_snapshots.qp_{step:04}.npz') as a:
  p={k:a['problem_'+k].astype(float).copy() for k in ('H','h','G','b','lb','ub','A_eq','b_eq')}
  j=np.zeros(76);j[:35]=a['com_jacobian'][0,1]
  bias=float(a['diagnostic_com_curvature'][0,1]); projected=float(a['diagnostic_com_acc_projected'][0,1]);fabric=float(a['diagnostic_com_acc_fabric'][0,1]);t=float(a['time'])
  idx=int(np.argmin(abs(times-t)));pdtarget=float(pd[idx]) if abs(times[idx]-t)<1e-5 else None
  actual=a['problem_solution'][0].astype(float)
 G=p['G'][0]; b=p['b'][0]
 accrows=np.any(abs(G[:,:35])>1e-9,axis=1)&~np.any(abs(G[:,35:])>1e-9,axis=1)
 wrows=np.any(abs(G[:,35:47])>1e-9,axis=1)&~np.any(abs(G[:,:35])>1e-9,axis=1)
 modes=['original','exact_projected_y','exact_fabric_y','com_y_cost250','com_y_cost1000']
 if pdtarget is not None:modes+=['exact_pd_y','pd_y_cost1000']
 if step==600:modes+=['no_acceleration_inequalities','no_wrench_inequalities','no_torque_bounds','no_inequalities_or_torque_bounds']
 cases=[]
 for mode in modes:
  trial={k:v.copy() for k,v in p.items()};target=projected
  if 'fabric' in mode:target=fabric
  if 'pd_' in mode:target=pdtarget
  if mode.startswith('exact'):
   trial['A_eq']=np.concatenate([trial['A_eq'],j[None,None]],1);trial['b_eq']=np.concatenate([trial['b_eq'],[[target-bias]]],1)
  if 'cost' in mode:
   weight=float(mode.split('cost')[1]);trial['H'][0]+=weight*np.outer(j,j);trial['h'][0]-=weight*j*(target-bias)
  drop=np.zeros(len(b),bool)
  if mode=='no_acceleration_inequalities':drop=accrows
  if mode=='no_wrench_inequalities':drop=wrows
  if mode=='no_inequalities_or_torque_bounds':drop[:]=True
  if mode in ['no_torque_bounds','no_inequalities_or_torque_bounds']:
   trial['lb'][0,47:]=-1e6;trial['ub'][0,47:]=1e6
  trial['G']=trial['G'][:,~drop];trial['b']=trial['b'][:,~drop]
  solver=DAQPBackend(single_solve=True);sol=solver.solve(QPData(**{k:torch.from_numpy(v) for k,v in trial.items()}));z=sol.z[0].numpy()
  violation=max(0.,float(np.max(trial['G'][0]@z-trial['b'][0],initial=0)),float(abs(trial['A_eq'][0]@z-trial['b_eq'][0]).max()),float((trial['lb'][0]-z).max()),float((z-trial['ub'][0]).max()))
  ok=bool(sol.status_ok.item());row=dict(case=mode,accepted=ok,status=solver.last_status,max_violation=violation)
  if ok:
   assert violation<1e-6
   desired=-p['h'][0]/np.diag(p['H'][0]);slack=b-G@z
   row.update(com_y_acc=float(j@z+bias),original_cost=float(.5*np.sum(np.diag(p['H'][0])*(z-desired)**2)),active_nonzero_inequalities=np.flatnonzero((abs(slack)<1e-6)&np.any(abs(G)>1e-9,axis=1)).tolist(),active_torque_bounds=[names[k] for k in range(29) if min(abs(z[47+k]-trial['lb'][0,47+k]),abs(z[47+k]-trial['ub'][0,47+k]))<1e-6],force=z[35:47].tolist(),root_acc=z[:6].tolist(),joint_acc=z[6:35].tolist())
   if mode=='original':row['max_difference_saved_solution']=float(abs(z-actual).max())
  else:
   lp=linprog(np.zeros(76),A_ub=trial['G'][0],b_ub=trial['b'][0],A_eq=trial['A_eq'][0],b_eq=trial['b_eq'][0],bounds=list(zip(trial['lb'][0],trial['ub'][0])),method='highs')
   row['lp_feasibility_status']=int(lp.status);row['lp_message']=lp.message
  cases.append(row)
  print(t,mode,ok,row.get('com_y_acc'),row.get('lp_feasibility_status'),flush=True)
 extrema=[]
 for sign in [1,-1]:
  lp=linprog(sign*j,A_ub=G,b_ub=b,A_eq=p['A_eq'][0],b_eq=p['b_eq'][0],bounds=list(zip(p['lb'][0],p['ub'][0])),method='highs')
  extrema.append(float(j@lp.x+bias) if lp.success else None)
 reports.append(dict(time=t,step=step,projected_com_y=projected,fabric_com_y=fabric,reconstructed_pd_y=pdtarget,feasible_com_y_range=extrema,cases=cases))
(OUT/'results.json').write_text(json.dumps(dict(scope='Frozen states, offline only; inequality removal is diagnosis, not a proposed controller setting. PD targets reconstructed from logs. No full Fabric metric reconstruction.',joint_names=names,states=reports),indent=2)+'\n')
