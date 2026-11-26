from pathlib import Path
import numpy as np,torch,warp as wp
from scipy.spatial.transform import Rotation
from test_g1_two_tasks import load_motion_trajectory
from fabrics_sim.fabrics.g1_pose_fabric import G1PoseFabric
from fabrics_sim.reference.interpolation import MotionInterpolator
wp.config.kernel_cache_dir='/tmp/g1_warp_cache';wp.init()
f=G1PoseFabric(1,'cpu',robot_name='g1_29dof_rev_1_0',task_profile='dynamics-aware')
q,fps=load_motion_trajectory(Path('acc_data/amp_walk_20261102/amp_walk_native_contacts.pkl'));s=MotionInterpolator(q,method='pchip');model=f.dynamics_tasks.model
ids=[model.kinematics.get_link_index(n+'_ankle_roll_link') for n in ['left','right']]
d=np.load('acc_data/amp_walk_20261102/contact_clock_followup/vertical_feedback_verified.failure.npz');t=d['history_time'].ravel();feet=d['history_feet'].reshape(-1,2,7)
with torch.no_grad():
 for time in [.45,.5,.52,.86,.9,.92,1.35,1.4,1.445,1.8,1.85,1.865,2.35,2.4,2.425,2.8,2.85,2.86]:
  x,v,a=s.kinematics(time*fps,fps,0.);st=model.snapshot(x);i=abs(t-time).argmin()
  print(time,'ref pitch',np.round(Rotation.from_quat(st['transforms'][0,ids,3:].numpy()).as_euler('xyz')[:,1],3),'actual',np.round(Rotation.from_quat(feet[i,:,3:]).as_euler('xyz')[:,1],3),'angular_v',np.round((st['angular_jac'][0,ids]@v[0,:,None]).squeeze(-1).numpy()[:,1],2),'forces',np.round(d['history_contact_normal_force'].reshape(-1,2)[i],1))
