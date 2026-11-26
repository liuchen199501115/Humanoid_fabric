import json
from pathlib import Path
import numpy as np, torch, warp as wp
from scipy.optimize import linprog
from fabrics_sim.fabrics.g1_pose_fabric import G1PoseFabric
wp.config.kernel_cache_dir='/tmp/g1_warp_cache';wp.init()
f=G1PoseFabric(1,'cpu',robot_name='g1_29dof_rev_1_0',task_profile='dynamics-aware')
p=Path('acc_data/g1_tracking_continue_20261123');h=np.load(p/'slow_com8.npz');out=[]
for path in sorted(p.glob('slow_com8.snapshot_*.npz')):
 d=np.load(path);prefix='soft_problem_' if 'soft_problem_H' in d else 'problem_';a={k:d[prefix+k][0].astype(float) for k in ['G','b','A_eq','b_eq','lb','ub']};n=len(a['lb']);st=f.dynamics_tasks.model.snapshot(torch.tensor(d['q']).float());J=st['com_jac'][0].detach().numpy();idx=abs(h['history_time']-float(d['time'])).argmin();bias=h['history_com_curvature'][idx].ravel();cols=list(range(18,35));extra=np.zeros((3+len(cols),n));extra[:3,:35]=J
 for row,col in enumerate(cols):extra[3+row,col]=1.
 target=np.r_[h['history_com_acc_original_desired'][idx].ravel()-bias,h['history_qdd_fabric'][idx,0,cols]]
 record=dict(time=float(d['time']),cases={})
 for kind in ['all','without_collision_rows','without_joint_tube','without_torquebounds','without_velocitybounds','without_contact','without_all_bounds']:
  mask=np.ones(len(a['b']),dtype=bool);lo=a['lb'].copy();hi=a['ub'].copy();A=a['A_eq'];e=a['b_eq']
  if kind=='without_collision_rows':mask[int(np.flatnonzero(np.any(abs(a['G'][:,:35])>1e-12,axis=1))[0]):-61]=False
  if kind=='without_joint_tube':mask[-3]=False
  if kind=='without_torquebounds':lo[47:76]=-np.inf;hi[47:76]=np.inf
  if kind=='without_velocitybounds':
   lo[:35]=-np.inf;hi[:35]=np.inf;mask[-61:-3]=False
  if kind=='without_all_bounds':lo[:76]=-np.inf;hi[:76]=np.inf
  if kind=='without_contact':A=A[:35];e=e[:35]
  E=np.vstack((A,extra));eq=np.r_[e,target];es=np.maximum(abs(E).max(axis=1),1e-12);G=a['G'][mask];b=a['b'][mask];gs=np.maximum(abs(G).max(axis=1),1e-12)
  r=linprog(np.zeros(n),A_ub=G/gs[:,None],b_ub=b/gs,A_eq=E/es[:,None],b_eq=eq/es,bounds=list(zip(lo,hi)),method='highs');record['cases'][kind]=dict(feasible=r.success,status=r.status)
 out.append(record)
Path('acc_data/g1_tracking_contacts_20261123/constraint_conflict.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
