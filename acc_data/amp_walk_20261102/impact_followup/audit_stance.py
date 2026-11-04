import json,numpy as np,torch,warp as wp
from pathlib import Path
from fabrics_sim.fabrics.g1_pose_fabric import G1PoseFabric
from fabrics_sim.reference.whole_body import WholeBodyReference
from fabrics_sim.reference.anchored_reference import anchor_reference
from test_g1_two_tasks import G1_JOINT_NAMES
wp.config.kernel_cache_dir='/tmp/g1_warp_cache';wp.init()
f=G1PoseFabric(1,'cpu',robot_name='g1_29dof_rev_1_0',task_profile='dynamics-aware');m=f.dynamics_tasks.model
feet=[m.kinematics.get_link_index(s+'_ankle_roll_link') for s in ('left','right')]
d=np.load('acc_data/amp_walk_20261102/continuation/measured_load.failure.npz');t=d['history_time'].reshape(-1);rows=[]
for time in [.2,.25,.6,.7,.95,1.,1.1,1.2,1.5,1.6]:
 i=int(np.abs(t-time).argmin());q=torch.tensor(d['history_whole_body_reference_q'][i]).reshape(1,36);v=torch.tensor(d['history_whole_body_reference_v'][i]).reshape(1,35);a=torch.tensor(d['history_whole_body_reference_a'][i]).reshape(1,35)
 st,_,_=m.evaluate(q,v,f.integrate_root_position,.001);live=torch.tensor(d['history_feet'][i]).reshape(1,2,7);mode=int(d['history_commanded_mode'][i].item());side=0 if mode==1 else 1
 row=dict(time=time,mode=mode,stance_displacement=(st['transforms'][0,feet[side],:3]-live[0,side,:3]).tolist(),source_stance_velocity=(st['origin_jac'][0,feet[side]]@v[0]).tolist())
 try:
  projected=anchor_reference(f,WholeBodyReference(q,v,a),G1_JOINT_NAMES,live,torch.tensor([[mode==1,mode==2]]))
  row.update(projection_joint_rmse=float((projected.q[:,7:]-q[:,7:]).square().mean().sqrt()),projection_max_v=float(projected.v.abs().max()),projection_max_a=float(projected.a.abs().max()))
 except (ValueError,RuntimeError) as e:row['projection_failure']=str(e)
 print(row,flush=True);rows.append(row)
Path('acc_data/amp_walk_20261102/impact_followup/stance_audit.json').write_text(json.dumps(rows,indent=2)+'\n')
