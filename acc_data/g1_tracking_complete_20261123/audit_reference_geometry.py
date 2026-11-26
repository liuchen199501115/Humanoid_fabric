"""Compare cover and mesh clearance of measured and commanded poses."""
import json
from pathlib import Path
import numpy as np
import torch
import warp as wp
from fabrics_sim.fabrics.g1_pose_fabric import G1PoseFabric

OUT=Path(__file__).resolve().parent
wp.config.kernel_cache_dir='/tmp/g1_warp_cache'; wp.init()
f=G1PoseFabric(1,'cpu',robot_name='g1_29dof_rev_1_0',task_profile='dynamics-aware')
d=f.dynamics_tasks;d.enable_wrist_hip_avoidance(include_upper_body=True,convex=True)
h=np.load('acc_data/g1_tracking_contacts_20261123/eight_verify.npz')
records=[]
with torch.no_grad():
 for sec in [2.7,2.9,3.0,3.05,3.1,3.132]:
  i=abs(h['history_time']-sec).argmin()
  for kind,key in [('measured','history_q'),('control','history_control_reference_q'),('source','history_reference_q')]:
   state=d.model.snapshot(torch.tensor(h[key][i]))
   sd,sj=d._sphere_geometry(state);cd,cj=d.convex_wrist_geometry.evaluate(state)
   j=int(sd[0,d.original_collision_pairs:].argmin())+d.original_collision_pairs
   names=[d.sphere_names[k] for k in d.pairs[j].tolist()]
   links=tuple(n.rsplit('_cover_',1)[0] for n in names)
   k=d.convex_wrist_geometry.pairs.index(links)
   records.append(dict(time_s=float(h['history_time'][i]),kind=kind,
      cover_min_m=float(sd[0,j]),cover_pair=names,corresponding_mesh_m=float(cd[0,k]),
      mesh_min_m=float(cd.min()),mesh_min_pair=d.convex_wrist_geometry.pairs[int(cd.argmin())]))
(OUT/'reference_geometry.json').write_text(json.dumps(records,indent=2)+'\n')
for r in records: print(r)
