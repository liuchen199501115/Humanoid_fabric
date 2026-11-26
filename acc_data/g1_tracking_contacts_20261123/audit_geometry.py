"""Compare existing cover spheres and collision-mesh convex hulls at logged poses."""
import json
from pathlib import Path
import numpy as np, torch, warp as wp
from fabrics_sim.fabrics.g1_pose_fabric import G1PoseFabric
wp.config.kernel_cache_dir='/tmp/g1_warp_cache';wp.init()
f=G1PoseFabric(1,'cpu',robot_name='g1_29dof_rev_1_0',task_profile='dynamics-aware')
t=f.dynamics_tasks;t.enable_wrist_hip_avoidance(include_upper_body=True,convex=True)
h=np.load('acc_data/g1_tracking_continue_20261123/slow_landing25.npz');out=[]
with torch.no_grad():
 for sec in [2.1,2.3,2.5,2.7,3.0]:
  i=abs(h['history_time']-sec).argmin();q=torch.tensor(h['history_q'][i]);v=torch.tensor(h['history_qd'][i]);state,plus,minus=t.model.evaluate(q,v,f.integrate_root_position,.001)
  sd,sj=t._sphere_geometry(state);cd,cj=t.convex_wrist_geometry.evaluate(state)
  sp=t._sphere_geometry(plus)[1];sm=t._sphere_geometry(minus)[1];cp=t.convex_wrist_geometry.evaluate(plus)[1];cm=t.convex_wrist_geometry.evaluate(minus)[1]
  sb=(((sp-sm)/.002)@v[...,None]).squeeze(-1);cb=(((cp-cm)/.002)@v[...,None]).squeeze(-1)
  acc=torch.tensor(h['history_qdd_fabric'][i]);sphere_violation=-(sj@acc[...,None]).squeeze(-1)-sb-16*(sj@v[...,None]).squeeze(-1)-64*(sd-.02)
  first=t.original_collision_pairs;worst=sphere_violation[0,first:].topk(5).indices+first;rows=[]
  for j in worst.tolist():
   names=[t.sphere_names[k] for k in t.pairs[j].tolist()];links=tuple(n.rsplit('_cover_',1)[0] for n in names);k=t.convex_wrist_geometry.pairs.index(links)
   cv=float(-(cj[:,k]@acc[...,None]).squeeze()-cb[0,k]-16*(cj[:,k]@v[...,None]).squeeze()-64*(cd[0,k]-.02))
   rows.append(dict(links=links,sphere_names=names,sphere_distance=float(sd[0,j]),convex_distance=float(cd[0,k]),sphere_barrier_violation=float(sphere_violation[0,j]),convex_barrier_violation=cv))
  out.append(dict(time=float(h['history_time'][i]),pairs=rows))
Path('acc_data/g1_tracking_contacts_20261123/geometry_audit.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
