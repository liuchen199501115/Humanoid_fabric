import json
from pathlib import Path
import numpy as np, torch, warp as wp
from fabrics_sim.fabrics.g1_pose_fabric import G1PoseFabric
wp.config.kernel_cache_dir='/tmp/g1_warp_cache';wp.init()
f=G1PoseFabric(1,'cpu',robot_name='g1_29dof_rev_1_0',task_profile='dynamics-aware');t=f.dynamics_tasks
t.enable_wrist_hip_avoidance(include_upper_body=True)
reports=[]
for tick in ['0550','0800']:
 d=np.load('acc_data/amp_walk_20261102/impact_followup/default_verified.qp_'+tick+'.npz');G=d['priority_level_0_G'][0];x=d['problem_solution'][0];r=d['priority_level_0_b'][0]-G@x;A=d['task_collision_A'][0]
 row=dict(time=float(d['time']),active_collision_constraints=[])
 for k in np.where((r<.001)&(np.linalg.norm(G[:,:35],axis=1)>1e-7))[0]:
  differences=np.linalg.norm(A+G[k,:35],axis=1);j=int(differences.argmin())
  if differences[j]>1e-5: continue
  pair=[t.sphere_names[i] for i in t.pairs[j]]
  row['active_collision_constraints'].append(dict(qp_row=int(k),pair_index=j,pair=pair,measured_distance=float(d['diagnostic_collision_distance'][0,j])))
 reports.append(row)
print(json.dumps(reports,indent=2));Path('acc_data/amp_walk_20261102/priority_followup/collision_audit.json').write_text(json.dumps(reports,indent=2)+'\n')
