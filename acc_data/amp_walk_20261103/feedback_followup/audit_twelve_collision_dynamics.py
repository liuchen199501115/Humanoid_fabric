"""Check whether a collision-free guide satisfies the selected local barrier."""
from pathlib import Path
import json,numpy as np,torch,warp as wp
from fabrics_sim.fabrics.g1_pose_fabric import G1PoseFabric
p=Path(__file__).parent
wp.config.kernel_cache_dir='/tmp/g1_warp_cache';wp.init()
fabric=G1PoseFabric(1,'cpu',robot_name='g1_29dof_rev_1_0',task_profile='dynamics-aware')
tasks=fabric.dynamics_tasks;tasks.enable_wrist_hip_avoidance(include_upper_body=True,convex=True)
d=np.load(p/'wbc_twelve_reproduced.npz');times=d['history_time'].ravel();rows=[];eps=tasks.config['difference_step']
for time in np.arange(3.2,4.301,.02):
 i=abs(times-time).argmin()
 q,v,a=[torch.from_numpy(d['history_whole_body_reference_'+k][i].reshape(1,-1)).float() for k in ['q','v','a']]
 state,plus,minus=tasks.model.evaluate(q,v,fabric.integrate_root_position,eps)
 dist,J=tasks.convex_wrist_geometry.evaluate(state)
 _,Jp=tasks.convex_wrist_geometry.evaluate(plus);_,Jm=tasks.convex_wrist_geometry.evaluate(minus)
 vel=(J@v[...,None]).squeeze(-1);acc=(J@a[...,None]).squeeze(-1)+(((Jp-Jm)/(2*eps))@v[...,None]).squeeze(-1)
 results=[]
 for rate in [8.,16.,24.]:
  h=dist-.02;constraint=acc+2*rate*vel+rate**2*h
  j=int(constraint[0].argmin());results.append(dict(rate=rate,minimum_constraint_value=float(constraint[0,j]),pair=tasks.convex_wrist_geometry.pairs[j],distance=float(dist[0,j]),speed=float(vel[0,j]),acceleration=float(acc[0,j])))
 rows.append(dict(time=float(times[i]),source_frame=float(d['history_frame'][i]),minimum_distance=float(dist.min()),cases=results))
(p/'twelve_collision_dynamics_audit.json').write_text(json.dumps(rows,indent=2)+'\n')
for rate in [8.,16.,24.]:
 values=[next(c for c in row['cases'] if c['rate']==rate)['minimum_constraint_value'] for row in rows]
 print(rate,'violating samples',sum(v<0 for v in values),'of',len(values),'worst',min(values))
print('minimum guide clearance',min(r['minimum_distance'] for r in rows))
