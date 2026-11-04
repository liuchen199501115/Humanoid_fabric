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
dyn=dict(armature=q.new_tensor(unitree_armatures(G1_JOINT_NAMES)),effort=effort,joint_bounds=bounds,acceleration=5*f.accel_limits)
from fabrics_sim.reference.full_body_com_path import OffsetComPath
rows_out=[]
body=source.clone();body[:,19:]=q[:,19:]
for duration in (.8,1.2):
 for fraction in (.25,.5):
  height_offset=-.06
  options=dict(duration=duration,body_join='endpoint',pelvis_pitch_offset=-.15,com_preview='full_body_source',
               landing_fraction=1.,landing_search_depth=0.)
  result=dict(height_offset=height_offset,stance_fraction=fraction,held_upper_body=True,options=options)
  try:
   plan=SourceStepCandidate(f,initial,G1_JOINT_NAMES,body,114,129,0,goal,**options)
   final_com=plan.preview.sample(plan.duration)[0]
   stance=plan.start_feet[0,0,:3].numpy()
   offset=np.r_[fraction*(stance[:2]-final_com[:2]),height_offset]
   plan.preview=OffsetComPath(plan.preview,offset)
   rows=evaluate_candidate(plan,np.linspace(0,plan.duration,31),velocity,dyn)
   collision=audit_collision_rows(f,rows,source)
   result.update(score=constraint_score(rows),collision=collision,com_end_offset_m=offset.tolist(),
     peak_torque_ratio=max(r['torque_ratio'] for r in rows),
     peak_wrench_score=max(r['exact_wrench_score'] for r in rows),
     worst_wrench_time_s=max(rows,key=lambda r:r['exact_wrench_score'])['time'],
     support_cop_min_m=np.asarray([r['cop'] for r in rows]).reshape(-1,2).min(0).tolist(),
     support_cop_max_m=np.asarray([r['cop'] for r in rows]).reshape(-1,2).max(0).tolist(),
     peak_acceleration_ratio=max(r['acceleration_ratio'] for r in rows),
     max_joint_bound_score=max(r['joint_bound_score'] for r in rows),
     goal_unchanged=bool(torch.equal(plan.landing_pose,goal)))
   if result['score']==0 and collision['passed']:
    np.savez('acc_data/walk1_full_body_posture_candidate_20261011.npz',
      time=np.array([r['time'] for r in rows]),q=np.array([r['q'] for r in rows]),
      v=np.array([r['v'] for r in rows]),a=np.array([r['a'] for r in rows]),
      com_end_offset_m=offset,goal=goal.numpy())
  except (ValueError,RuntimeError) as e:result['error']=str(e)
  rows_out.append(result);print(json.dumps(result),flush=True)
  Path('acc_data/walk1_full_body_posture_20261011.json').write_text(json.dumps(dict(
   scope='Offline full-model COM adaptation, historical zero-rate boundary; NOT physical tracking',
   source_record='acc_data/walk1_strict_four_transfer_20261008.reference_failure.npz',
   ground_offset_m=-.021463990211486816,candidates=rows_out,simulation_validated=False),indent=2)+'\n')
