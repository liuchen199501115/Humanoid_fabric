import json
from pathlib import Path
import numpy as np, torch, warp as wp
from scipy.optimize import linprog
from fabrics_sim.fabrics.g1_pose_fabric import G1PoseFabric
wp.config.kernel_cache_dir='/tmp/g1_warp_cache';wp.init()
f=G1PoseFabric(1,'cpu',robot_name='g1_29dof_rev_1_0',task_profile='dynamics-aware')
p=Path('acc_data/g1_tracking_continue_20261123');h=np.load(p/'slow_com8.npz');out=[]
for path in sorted(p.glob('slow_com8.snapshot_*.npz')):
 d=np.load(path);prefix='soft_problem_' if 'soft_problem_H' in d else 'problem_';a={k:d[prefix+k][0].astype(float) for k in ['G','b','A_eq','b_eq','lb','ub']};n=len(a['lb']);s=f.dynamics_tasks.model.snapshot(torch.tensor(d['q']).float());J=s['com_jac'][0].detach().numpy();idx=abs(h['history_time']-float(d['time'])).argmin();bias=h['history_com_curvature'][idx].ravel();record=dict(time=float(d['time']),requested=h['history_com_acc_original_desired'][idx].ravel().tolist(),predicted=h['history_com_acc_qp_predicted'][idx].ravel().tolist(),ranges={})
 gs=np.maximum(abs(a['G']).max(axis=1),1e-12);es=np.maximum(abs(a['A_eq']).max(axis=1),1e-12)
 for name,coeff,offset in [('com_y',J[1],bias[1]),('left_shoulder_pitch',np.eye(35)[21],0.),('right_shoulder_roll',np.eye(35)[29],0.)]:
  cost=np.r_[coeff,np.zeros(n-35)];values=[]
  for direction in [1.,-1.]:
   r=linprog(direction*cost,A_ub=a['G']/gs[:,None],b_ub=a['b']/gs,A_eq=a['A_eq']/es[:,None],b_eq=a['b_eq']/es,bounds=list(zip(a['lb'],a['ub'])),method='highs',options={'time_limit':10.});values.append(float(cost@r.x+offset) if r.success else r.message)
  record['ranges'][name]=values
 record['simultaneous_targets']={}
 for group,cols in [('com_only',[]),('com_and_upper',list(range(18,35)))]:
  extra=np.zeros((3+len(cols),n));extra[:3,:35]=J
  for row,col in enumerate(cols):extra[3+row,col]=1.
  target=np.r_[np.array(record['requested'])-bias,h['history_qdd_fabric'][idx,0,cols]]
  E=np.vstack((a['A_eq'],extra));e=np.r_[a['b_eq'],target];scale=np.maximum(abs(E).max(axis=1),1e-12)
  r=linprog(np.zeros(n),A_ub=a['G']/gs[:,None],b_ub=a['b']/gs,A_eq=E/scale[:,None],b_eq=e/scale,bounds=list(zip(a['lb'],a['ub'])),method='highs')
  record['simultaneous_targets'][group]=dict(feasible=r.success,status=r.status)
 out.append(record)
(p/'com_authority.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
