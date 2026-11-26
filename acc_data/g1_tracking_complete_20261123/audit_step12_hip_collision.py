"""Compare native hip collision pair against measured and source poses, offline only."""
import json
from pathlib import Path
import numpy as np
import torch
import warp as wp
from fabrics_sim.fabrics.g1_pose_fabric import G1PoseFabric
from fabrics_sim.reference.convex_collision import ConvexLinkCollision
OUT=Path(__file__).resolve().parent
wp.config.kernel_cache_dir='/tmp/g1_warp_cache';wp.init()
f=G1PoseFabric(1,'cpu',robot_name='g1_29dof_rev_1_0',task_profile='dynamics-aware');d=f.dynamics_tasks
pair=('left_hip_yaw_link','right_hip_yaw_link')
g=ConvexLinkCollision(f.urdf_path,d.model.kinematics,[pair]);a=np.load(OUT/'eleven_leg16_lateupper64.npz');rows=[]
with torch.no_grad():
 for t in [4.25,4.3,4.344,4.4,4.45,4.5,4.52,4.54,4.55,4.56,4.57]:
  i=int(abs(a['history_time']-t).argmin());row={'time':float(a['history_time'][i])}
  for label,key in [('measured','history_q'),('control','history_control_reference_q'),('source','history_reference_q')]:
   state=d.model.snapshot(torch.tensor(a[key][i]));dist,jac=g.evaluate(state);row[label+'_distance_m']=float(dist[0,0])
  rows.append(row);print(row,flush=True)
(OUT/'step12_hip_collision.json').write_text(json.dumps(dict(pair=pair,native_events=a['self_contact_events'].tolist(),samples=rows),indent=2)+'\n')
