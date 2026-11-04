"""First-step full-body audit using synchronized measured unloading state."""
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
import argparse
from analyze_contact_transfer import summarize_unloading
parser=argparse.ArgumentParser(description='Audit first original landing from measured unloading; no live reset')
parser.add_argument('--record',type=Path,default=Path('acc_data/walk1_unload_verified_20261012.npz'))
parser.add_argument('--output',type=Path,default=Path('acc_data/walk1_unload_handoff_20261014.json'))
parser.add_argument('--duration',type=float,default=1.5)
parser.add_argument('--samples',type=int,default=31)
args=parser.parse_args()
if not np.isfinite(args.duration) or args.duration<=0 or args.samples<3:parser.error('Invalid duration or sample count')
args.output.parent.mkdir(parents=True,exist_ok=True)
wp.config.kernel_cache_dir='/tmp/g1_warp_cache';wp.init();torch.set_grad_enabled(False)
f=G1PoseFabric(1,'cpu',robot_name='g1_29dof_rev_1_0',task_profile='dynamics-aware')
f.dynamics_tasks.enable_wrist_hip_avoidance(include_upper_body=True);f.dynamics_tasks.wrist_hip_barrier=True
with np.load(args.record) as record:
 audit=summarize_unloading(record)
 if not audit['unloading_passed']:raise ValueError('Recorded unloading did not pass: '+str(audit))
 q=torch.tensor(record['history_q'][-1])
 initial=WholeBodyReference(q,torch.tensor(record['history_qd'][-1]),torch.tensor(record['history_qdd_predicted'][-1]))
 ground_shift=float(record['motion_ground_shift'])
 boundary=dict(time=float(record['history_time'][-1]),frame=float(record['history_frame'][-1]),
  acceleration_source='WBC prediction at the same sample; not measured acceleration',unloading=audit)
source,_=load_motion_trajectory(Path('retarget/lafan/walk1_g1.pkl'))
source[:,2]+=ground_shift
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
 options=dict(duration=args.duration,clearance=.02,body_join=join,
              com_preview='full_body_source',landing_fraction=1.,landing_search_depth=0.)
 rows=[];time=0.;entry_error=None
 try:
  plan=SourceStepCandidate(f,initial,G1_JOINT_NAMES,source,114,129,0,goal,**options)
  start=plan.sample(0.)
  entry_error={name:float((getattr(start,name)-getattr(initial,name)).abs().max()) for name in ('q','v','a')}
  for time in np.linspace(0,plan.duration,args.samples):
   # Evaluate prefixes to preserve the IK seed; a failure reports its first time.
   rows=evaluate_candidate(plan,np.linspace(0,time,len(rows)+1),velocity,dyn)
  landed=model.snapshot(q.new_tensor(rows[-1]['q']))['transforms'][:,idx]
  endpoint_position_error=float((landed[:,:3]-goal[:,:3]).norm())
  endpoint_rotation_error=float(torch.minimum((landed[:,3:]-goal[:,3:]).norm(),(landed[:,3:]+goal[:,3:]).norm()))
  result=dict(options=options,entry_error=entry_error,endpoint_position_error_m=endpoint_position_error,
              endpoint_quaternion_error=endpoint_rotation_error,score=constraint_score(rows),collision=audit_collision_rows(f,rows,source),
              goal=goal.tolist(),goal_unchanged=bool(torch.equal(plan.landing_pose,goal)))
 except (ValueError,RuntimeError) as e:
  result=dict(options=options,entry_error=entry_error,error=str(e),first_failed_time_s=float(time),
              completed_sample_count=len(rows),goal=goal.tolist())
  if rows:
   result['prefix_constraint_score']=constraint_score(rows)
   for key in ('velocity_ratio','torque_ratio','acceleration_ratio','joint_bound_score','exact_wrench_score'):
    worst=max(rows,key=lambda r:r.get(key,0.))
    result[key]=dict(maximum=float(worst.get(key,0.)),time=float(worst['time']))

 result['sampled_constraints_passed']=bool(result.get('score',1)==0 and result.get('collision',{}).get('passed',False) and result.get('endpoint_position_error_m',1)<=5e-6 and result.get('endpoint_quaternion_error',1)<=5e-6 and entry_error is not None and entry_error['q']<=5e-6 and entry_error['v']<=1e-4 and entry_error['a']<=1e-3)
 rows_out.append(result);print(json.dumps(result),flush=True)
 args.output.write_text(json.dumps(dict(
  scope='Measured unloading boundary; offline necessary checks only',boundary=boundary,
  source_record=str(args.record),
  ground_offset_m=ground_shift,
  stance_source_minus_anchor_m=(model.snapshot(source[129:130])['transforms'][0,model.kinematics.get_link_index('left_ankle_roll_link'),:3]-model.snapshot(q)['transforms'][0,model.kinematics.get_link_index('left_ankle_roll_link'),:3]).tolist(),
  candidates=rows_out,software_acceleration_limits_enabled=False,joint_braking_limits_enabled=False,simulation_validated=False),indent=2)+'\n')
