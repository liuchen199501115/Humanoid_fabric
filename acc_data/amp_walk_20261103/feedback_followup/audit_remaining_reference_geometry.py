from pathlib import Path
import json,numpy as np,torch,warp as wp
from fabrics_sim.fabrics.g1_pose_fabric import G1PoseFabric
p=Path(__file__).parent;wp.config.kernel_cache_dir='/tmp/g1_warp_cache';wp.init()
f=G1PoseFabric(1,'cpu',robot_name='g1_29dof_rev_1_0',task_profile='dynamics-aware');t=f.dynamics_tasks;t.enable_wrist_hip_avoidance(include_upper_body=True,convex=True)
q=np.load(p/'wbc_convex_catch15_25_clear35.npz')['control_q'];rows=[]
for frame in range(700,len(q),4):
 state=t.model.snapshot(torch.from_numpy(q[frame:frame+1]).float());d,J=t.convex_wrist_geometry.evaluate(state);i=int(d[0].argmin());rows.append(dict(frame=frame,distance=float(d[0,i]),pair=t.convex_wrist_geometry.pairs[i]))
(p/'remaining_reference_geometry_audit.json').write_text(json.dumps(rows,indent=2)+'\n')
print('minimum',min(rows,key=lambda r:r['distance']));print('below 2cm',sum(r['distance']<.02 for r in rows),'of',len(rows))
