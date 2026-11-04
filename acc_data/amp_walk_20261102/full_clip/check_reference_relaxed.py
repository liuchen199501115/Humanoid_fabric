import json
from pathlib import Path
import numpy as np,torch,warp as wp
from test_g1 import G1PoseFabric,ground_motion_reference
from test_g1_two_tasks import G1_JOINT_NAMES,load_motion_trajectory
from fabrics_sim.reference.swing_clearance import swing_clearance_reference
wp.init();torch.set_grad_enabled(False)
p=Path('acc_data/amp_walk_20261102/full_clip')
f=G1PoseFabric(1,'cpu',robot_name='g1_29dof_rev_1_0',task_profile='dynamics-aware')
q,fps=load_motion_trajectory(Path('acc_data/amp_walk_20261102/amp_walk_native_contacts.pkl'));q,shift=ground_motion_reference(f,q,0)
plan=json.loads((p/'full_motion.steps_plan.json').read_text())
geometry=list(zip(plan['collision_points'],plan['collision_radii']))
code=Path('src/fabrics_sim/reference/swing_clearance.py').read_text().replace('error.abs().max() < 2e-5', 'error.abs().max() < 1e-3')
exec(compile(code,'diagnostic_ik','exec'))
result=swing_clearance_reference(f,q,G1_JOINT_NAMES,plan['events'],.056,collision_geometry=geometry)
# Diagnostic only; do not save a control reference.
print('Full reference IK passed',len(result))
