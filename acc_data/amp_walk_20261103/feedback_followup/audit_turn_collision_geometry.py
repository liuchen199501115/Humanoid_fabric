from pathlib import Path
import json,numpy as np,torch,warp as wp
from fabrics_sim.fabrics.g1_pose_fabric import G1PoseFabric
p=Path(__file__).parent
wp.config.kernel_cache_dir='/tmp/g1_warp_cache';wp.init()
fabric=G1PoseFabric(1,'cpu',robot_name='g1_29dof_rev_1_0',task_profile='dynamics-aware')
tasks=fabric.dynamics_tasks;tasks.enable_wrist_hip_avoidance(include_upper_body=True,convex=True)
d=np.load(p/'full_motion_no_mpc.npz');time=d['history_time'].ravel();rows=[]
for sec in np.arange(3.2,4.081,.04):
 i=abs(time-sec).argmin()
 for kind,key in [('observed','history_q'),('source','history_reference_q')]:
  q=torch.from_numpy(d[key][i].reshape(1,36)).float();s=tasks.model.snapshot(q)
  sd,sj=tasks._sphere_geometry(s);cd,cj=tasks.convex_wrist_geometry.evaluate(s)
  sphere=sd[0,tasks.original_collision_pairs:len(tasks.pairs)].numpy();convex=cd[0].numpy()
  k=int(np.argmin(sphere));j=int(np.argmin(convex))
  rows.append(dict(time=float(time[i]),kind=kind,sphere_minimum=float(sphere[k]),sphere_pair=tasks.pairs[k+tasks.original_collision_pairs].tolist(),convex_minimum=float(convex[j]),convex_pair=tasks.convex_wrist_geometry.pairs[j]))
(p/'turn_collision_geometry_audit.json').write_text(json.dumps(rows,indent=2)+'\n')
for row in rows:
 if row['sphere_minimum']<.02:print(row)
