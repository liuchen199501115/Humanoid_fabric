import json,numpy as np,torch,warp as wp
from pathlib import Path
from fabrics_sim.fabrics.g1_pose_fabric import G1PoseFabric
wp.config.kernel_cache_dir='/tmp/g1_warp_cache';wp.init()
f=G1PoseFabric(1,'cpu',robot_name='g1_29dof_rev_1_0',task_profile='dynamics-aware');m=f.dynamics_tasks.model;foot=m.kinematics.get_link_index('left_ankle_roll_link')
p=Path(__file__).resolve().parent;d=np.load(p/'unload_priority.npz');t=d['history_time'];rows=[]
for tt in [1.7,1.75,1.76,1.78,1.8,1.85,1.9]:
 i=np.abs(t-tt).argmin();q=torch.tensor(d['history_whole_body_reference_q'][i]).reshape(1,36);v=torch.tensor(d['history_whole_body_reference_v'][i]).reshape(1,35);st=m.snapshot(q)
 rows.append(dict(time=float(t[i]),source_foot_position=st['transforms'][0,foot,:3].tolist(),source_foot_velocity=(st['origin_jac'][0,foot]@v[0]).tolist(),actual_foot_position=d['history_feet'][i].reshape(2,7)[0,:3].tolist()))
print(json.dumps(rows,indent=2));(p/'fifth_reference_audit.json').write_text(json.dumps(rows,indent=2)+'\n')
