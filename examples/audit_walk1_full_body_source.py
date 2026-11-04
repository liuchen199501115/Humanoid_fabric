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
record=np.load('acc_data/walk1_strict_four_transfer_20261008.reference_failure.npz')
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
# Software acceleration/braking checks removed; keep full dynamics and effort checks.
f.fabric_params['joint_limits']['active']=False
dyn=dict(armature=q.new_tensor(unitree_armatures(G1_JOINT_NAMES)),effort=effort)
rows_out=[]
for join in ('first_sample','distributed','endpoint'):
 options=dict(duration=1.5,clearance=.02,body_join=join,
              com_preview='full_body_source',landing_fraction=1.,landing_search_depth=0.)
 rows=[];time=0.
 try:
  plan=SourceStepCandidate(f,initial,G1_JOINT_NAMES,source,114,129,0,goal,**options)
  for time in np.linspace(0,plan.duration,61):
   # Evaluate prefixes to preserve the IK seed; a failure reports its first time.
   rows=evaluate_candidate(plan,np.linspace(0,time,len(rows)+1),velocity,dyn)
  result=dict(options=options,score=constraint_score(rows),collision=audit_collision_rows(f,rows,source),
              goal=goal.tolist(),goal_unchanged=bool(torch.equal(plan.landing_pose,goal)))
 except (ValueError,RuntimeError) as e:
  result=dict(options=options,error=str(e),first_failed_time_s=float(time),
              completed_sample_count=len(rows),goal=goal.tolist())
  if rows:
   result['prefix_constraint_score']=constraint_score(rows)
   for key in ('velocity_ratio','torque_ratio','acceleration_ratio','joint_bound_score','exact_wrench_score'):
    worst=max(rows,key=lambda r:r.get(key,0.))
    result[key]=dict(maximum=float(worst.get(key,0.)),time=float(worst['time']))

 rows_out.append(result);print(json.dumps(result),flush=True)
 Path('acc_data/walk1_full_body_source_feet_20261011.json').write_text(json.dumps(dict(
  scope='Historical lifted pose with zero rate hypothesis; offline only',
  source_record='acc_data/walk1_strict_four_transfer_20261008.reference_failure.npz',
  ground_offset_m=-.021463990211486816,
  stance_source_minus_anchor_m=(model.snapshot(source[129:130])['transforms'][0,model.kinematics.get_link_index('left_ankle_roll_link'),:3]-model.snapshot(q)['transforms'][0,model.kinematics.get_link_index('left_ankle_roll_link'),:3]).tolist(),
  candidates=rows_out,software_acceleration_limits_enabled=False,joint_braking_limits_enabled=False,simulation_validated=False),indent=2)+'\n')
