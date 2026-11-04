"""Offline first-step audit from the saved lifted pose; NOT live four-step tracking.

Run with PYTHONPATH=src:examples. The input is the historical first-lift record;
zero boundary rates are an explicit hypothesis, not a reset of a live robot.
"""
import json
from pathlib import Path
import numpy as np
import torch,warp as wp
import xml.etree.ElementTree as ET
from fabrics_sim.fabrics.g1_pose_fabric import G1PoseFabric
from fabrics_sim.reference.whole_body import WholeBodyReference
from fabrics_sim.reference.source_step import SourceStepCandidate
from fabrics_sim.reference.source_step_optimization import evaluate_candidate,constraint_score
from fabrics_sim.reference.collision_audit import audit_collision_rows
from test_g1_two_tasks import G1_JOINT_NAMES,load_motion_trajectory
from g1_armature import unitree_armatures
wp.config.kernel_cache_dir='/tmp/g1_warp_cache';wp.init();torch.set_grad_enabled(False)
f=G1PoseFabric(1,'cpu',robot_name='g1_29dof_rev_1_0',task_profile='dynamics-aware')
f.dynamics_tasks.enable_wrist_hip_avoidance(include_upper_body=True);f.dynamics_tasks.wrist_hip_barrier=True
record=np.load('acc_data/walk1_no_reduced_four_20261009.reference_failure.npz')
q=torch.tensor(record['q']);zero=torch.zeros(1,35)
# A stopped measured-pose boundary hypothesis, not a live reference jump.
initial=WholeBodyReference(q,zero,zero)
source,_=load_motion_trajectory(Path('retarget/lafan/walk1_g1.pkl'))
source[:,2]-=.021463990211486816
model=f.dynamics_tasks.model;idx=model.kinematics.get_link_index('right_ankle_roll_link')
goal=model.snapshot(source[129:130])['transforms'][:,idx].clone()
limits={n.attrib['name']:n.find('limit') for n in ET.parse(f.urdf_path).getroot().findall('joint')}
velocity=q.new_tensor([float(limits[n].attrib['velocity']) for n in G1_JOINT_NAMES]);effort=q.new_tensor([float(limits[n].attrib['effort']) for n in G1_JOINT_NAMES])
bounds=dict(lower=q.new_tensor([float(limits[n].attrib['lower']) for n in G1_JOINT_NAMES]),upper=q.new_tensor([float(limits[n].attrib['upper']) for n in G1_JOINT_NAMES]),dt=.002,bandwidth=8.)
dyn=dict(armature=q.new_tensor(unitree_armatures(G1_JOINT_NAMES)),effort=effort,joint_bounds=bounds,acceleration=5*f.accel_limits)
from fabrics_sim.reference.contact_transfer import MeasuredContactTransferReference
# Acceleration is not measured by this recording; use and report zero as the entry-reference assumption.
measured=WholeBodyReference(q,torch.tensor(record['qd']),zero)
feet=[model.kinematics.get_link_index(side+'_ankle_roll_link') for side in ('left','right')]
stance=model.snapshot(q)['transforms'][:,feet].clone()
result=dict(source_record='acc_data/walk1_no_reduced_four_20261009.reference_failure.npz',
 acceleration_assumption='zero reference acceleration, not measured acceleration',simulation_validated=False)
try:
 plan=MeasuredContactTransferReference(f,measured,G1_JOINT_NAMES,stance,0,0.,joint_bounds=bounds,preflight_dt=.002)
 start=plan.sample(0.,3)
 result.update(preflight=plan.preflight,initial_q_error=float((start.q-measured.q).abs().max()),
   initial_v_error=float((start.v-measured.v).abs().max()),initial_a_error=float((start.a-measured.a).abs().max()))
except (ValueError,RuntimeError) as exc:result['error']=str(exc)
Path('acc_data/walk1_measured_entry_transfer_20261012.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
