from pathlib import Path
import numpy as np,torch,warp as wp
from test_g1_two_tasks import G1_JOINT_NAMES
from fabrics_sim.fabrics.g1_pose_fabric import G1PoseFabric
from fabrics_sim.reference.interpolation import MotionInterpolator
from consistent_planner import plan_reference
wp.config.kernel_cache_dir='/tmp/g1_warp_cache';wp.init()
f=G1PoseFabric(1,'cpu',robot_name='g1_29dof_rev_1_0',task_profile='dynamics-aware');p=Path(__file__).parent
with np.load(p/'flat_heel.centroidal_path.npz') as d:q=torch.tensor(d['q'])
with np.load(p/'flat_heel.foot_targets.npz') as d:contact=d['contact'];target=d['position'];rotation=d['rotation']
s=MotionInterpolator(q,method='bspline')
plan_reference(f,s,contact,200.,G1_JOINT_NAMES,p/'flat_consistent.npz',foot_targets=(target,rotation))
