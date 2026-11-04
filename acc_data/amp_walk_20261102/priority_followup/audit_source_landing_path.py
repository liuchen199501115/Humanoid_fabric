"""Independent audit only: compare landed foot links with same-frame source FK.
The original fixed-touchdown reports and pass/fail flags are not overwritten.
"""
import json,numpy as np,torch,warp as wp
from pathlib import Path
from fabrics_sim.fabrics.g1_pose_fabric import G1PoseFabric
from fabrics_sim.reference.bounded_steps import MeasuredSteps
from analyze_contact_transfer import sole_clearances
wp.config.kernel_cache_dir='/tmp/g1_warp_cache';wp.init();torch.set_grad_enabled(False)
f=G1PoseFabric(1,'cpu',robot_name='g1_29dof_rev_1_0',task_profile='dynamics-aware');model=f.dynamics_tasks.model
indices=[model.kinematics.get_link_index(s+'_ankle_roll_link') for s in ('left','right')]
p=Path(__file__).resolve().parent;reports=[]
for stem in ['unload_priority','landing_tail']:
 path=p/(stem+'.npz')
 if not path.exists():path=p/(stem+'.failure.npz')
 if not path.exists():continue
 d=np.load(path);plan=json.loads((p/(stem+'.steps_plan.json')).read_text());t=d['history_time'];frames=d['history_frame'].ravel();feet=d['history_feet'].reshape(-1,2,7);forces=d['history_ground_foot_forces'].reshape(-1,2,3)[:,:,2].clip(min=0);mode=d['history_measured_mode'].ravel();q=d['history_q'].reshape(-1,36);ref=d['history_reference_q'].reshape(-1,36)
 reference_feet=np.stack([model.snapshot(torch.tensor(row)[None])['transforms'][0,indices,:3].numpy() for row in ref]);np.savez_compressed(p/(stem+'.source_feet.npz'),time=t,frame=frames,reference_feet=reference_feet)
 clearance=sole_clearances(feet,plan['collision_points'],plan['collision_radii']);monitor=MeasuredSteps(plan['events'],plan['landing_positions_world'],continuous=True)
 for i in range(len(t)):
  if monitor.complete:break
  event=monitor.events[len(monitor.landings)];sw=1-event['support'];monitor.goals[len(monitor.landings)]=reference_feet[i,sw]
  before=len(monitor.landings);monitor.update(frames[i],t[i],mode[i],forces[i],clearance[i],feet[i,:,:3],t[i]-t[i-1] if i else t[1]-t[0])
  if len(monitor.landings)>before:
   angle=2*np.arccos(np.clip(abs(np.dot(q[i,3:7],ref[i,3:7]))/(np.linalg.norm(q[i,3:7])*np.linalg.norm(ref[i,3:7])),0,1))
   monitor.landings[-1].update(root_angle_error_rad=float(angle),root_position_error_m=float(np.linalg.norm(q[i,:3]-ref[i,:3])),joint_rmse_rad=float(np.sqrt(np.mean((q[i,7:]-ref[i,7:])**2))))
 event=plan['events'][4];sw=1-event['support'];goal=plan['landing_positions_world'][4];window=(frames>=event['end_frame']-1)&(frames<event['landing_window_end_frame']);inside=window&(np.linalg.norm(reference_feet[:,sw]-goal,axis=1)<=.04);held=longest=0.
 for i,valid in enumerate(inside):
  dt=t[i]-t[i-1] if i else t[1]-t[0];held=held+dt if valid else 0.;longest=max(longest,held)
 row=dict(run=stem,diagnostic_only=True,official_pass_flags_unchanged=True,same_frame_source_pose_landings=monitor.landings,ideal_source_fifth_fixed_goal_max_hold_s=float(longest),required_hold_s=.05)
 reports.append(row);print(stem,'same-frame source pose landings',len(monitor.landings),'ideal source max fixed goal hold',longest,flush=True)
(p/'source_landing_path_audit.json').write_text(json.dumps(reports,indent=2)+'\n')
