import json
from pathlib import Path
import numpy as np, torch, warp as wp
from scipy.optimize import linprog
from fabrics_sim.fabrics.g1_pose_fabric import G1PoseFabric
wp.config.kernel_cache_dir='/tmp/g1_warp_cache';wp.init()
f=G1PoseFabric(1,'cpu',robot_name='g1_29dof_rev_1_0',task_profile='dynamics-aware')
p=Path('acc_data/g1_tracking_continue_20261123');h=np.load(p/'slow_com8.npz');out=[]
f.dynamics_tasks.enable_wrist_hip_avoidance(include_upper_body=True,convex=False)
from fabrics_sim.solver.collision_barrier import wrist_collision_barrier
for path in sorted(p.glob('slow_com8.snapshot_*.npz')):
 d=np.load(path);prefix='soft_problem_' if 'soft_problem_H' in d else 'problem_';a={k:d[prefix+k][0].astype(float) for k in ['G','b','A_eq','b_eq','lb','ub']};n=len(a['lb']);st=f.dynamics_tasks.model.snapshot(torch.tensor(d['q']).float());J=st['com_jac'][0].detach().numpy();idx=abs(h['history_time']-float(d['time'])).argmin();bias=h['history_com_curvature'][idx].ravel();cols=list(range(18,35));extra=np.zeros((3+len(cols),n));extra[:3,:35]=J
 for row,col in enumerate(cols):extra[3+row,col]=1.
 target=np.r_[h['history_com_acc_original_desired'][idx].ravel()-bias,h['history_qdd_fabric'][idx,0,cols]]
 record=dict(time=float(d['time']),cases={})
 state,plus,minus=f.dynamics_tasks.model.evaluate(torch.tensor(d['q']).float(),torch.tensor(d['v']).float(),f.integrate_root_position,.001)
 task=f.dynamics_tasks;distance,Jd=task._sphere_geometry(state);Jp=task._sphere_geometry(plus)[1];Jm=task._sphere_geometry(minus)[1];velocity=torch.tensor(d['v']).float();bias_d=(((Jp-Jm)/.002)@velocity[...,None]).squeeze(-1);speed=(Jd@velocity[...,None]).squeeze(-1)
 for rate in [8.,12.,16.,20.,24.,32.]:
  C,D=wrist_collision_barrier(distance,Jd,bias_d,speed,task.original_collision_pairs,len(task.pairs),.02,False,rate)
  C=C.detach().numpy()[0];D=D.detach().numpy()[0];G=a['G'].copy();b=a['b'].copy();first=int(np.flatnonzero(np.any(abs(G[:,:35])>1e-12,axis=1))[0]);assert len(C)==len(G)-first-61,(len(C),len(G),first)
  if rate==8.:record['reconstruction_error']=dict(C=float(abs(G[first:-61,:35]-C).max()),d=float(abs(b[first:-61]-D).max()))
  G[first:-61,:35]=C;b[first:-61]=D
  E=np.vstack((a['A_eq'],extra));eq=np.r_[a['b_eq'],target];es=np.maximum(abs(E).max(axis=1),1e-12);gs=np.maximum(abs(G).max(axis=1),1e-12)
  r=linprog(np.zeros(n),A_ub=G/gs[:,None],b_ub=b/gs,A_eq=E/es[:,None],b_eq=eq/es,bounds=list(zip(a['lb'],a['ub'])),method='highs');record['cases'][str(rate)]=dict(feasible=r.success,status=r.status)
 out.append(record)
Path('acc_data/g1_tracking_contacts_20261123/barrier_bandwidth.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
