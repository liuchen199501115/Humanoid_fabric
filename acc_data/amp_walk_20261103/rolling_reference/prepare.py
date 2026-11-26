from pathlib import Path
import json,numpy as np,torch,warp as wp
from scipy.spatial.transform import Rotation
from test_g1_two_tasks import load_motion_trajectory,load_motion_contact_sequence,G1_JOINT_NAMES
from fabrics_sim.fabrics.g1_pose_fabric import G1PoseFabric
from fabrics_sim.reference.interpolation import MotionInterpolator
from fabrics_sim.solver.AccelerationQP import build_single_support_wrench_constraints
from rolling_planner import plan_reference
wp.config.kernel_cache_dir='/tmp/g1_warp_cache';wp.init()
f=G1PoseFabric(1,'cpu',robot_name='g1_29dof_rev_1_0',task_profile='dynamics-aware');p=Path(__file__).parent
q,fps=load_motion_trajectory(Path('acc_data/amp_walk_20261102/amp_walk_native_contacts.pkl'))
contact=load_motion_contact_sequence(Path('acc_data/amp_walk_20261102/amp_walk_native_contacts.pkl')).numpy()
model=f.dynamics_tasks.model;ids=[model.kinematics.get_link_index(n+'_ankle_roll_link') for n in ['left','right']]
geometry=json.loads(Path('acc_data/amp_walk_20261102/contact_clock_followup/vertical_feedback_verified.steps_plan.json').read_text())
points=np.asarray(geometry['collision_points']);radii=np.asarray(geometry['collision_radii'])
position=[];rotation=[]
with torch.no_grad():
 for frame in range(len(q)):
  st=model.snapshot(q[frame:frame+1]);position.append(st['transforms'][0,ids,:3].numpy());rotation.append(st['rotations'][0,ids].numpy())
position=np.asarray(position);rotation=np.asarray(rotation)
corners=np.array([[-.05,-.025,-.035],[-.05,.025,-.035],[.12,-.025,-.035],[.12,.025,-.035]])
shift=-np.min(position[0,:,2,None]+(rotation[0]@corners.T)[:,2,:]);position[:,:,2]+=shift;q[:,2]+=shift
angles=Rotation.from_matrix(rotation.reshape(-1,3,3)).as_euler('xyz').reshape(-1,2,3)
target=position.copy();target_R=rotation.copy();support=position.copy();support[:,:,2]=0.
kind=np.zeros((len(q),2),int);halfwidth=np.full((len(q),2),.025);yaw=angles[:,:,2].copy()
for side in range(2):
 active=contact[:,side];cuts=np.r_[0,np.flatnonzero(active[1:]!=active[:-1])+1,len(q)]
 for lo,hi in zip(cuts[:-1],cuts[1:]):
  if not active[lo]:continue
  low=np.arange(lo,hi)[position[lo:hi,side,2]<position[lo:hi,side,2].min()+.008]
  heading=float(np.median(angles[low,side,2]));frame_R=Rotation.from_euler('z',heading).as_matrix();last_local=last_anchor=last_radius=None
  for k in range(lo,hi):
   pitch=angles[k,side,1];phase=min(1.,max(0.,(abs(pitch)-.02)/.06));blend=phase**3*(10-15*phase+6*phase*phase)
   R=frame_R@Rotation.from_euler('y',pitch*blend).as_matrix();yaw[k,side]=heading;target_R[k,side]=R
   height=(R@points[side].T)[2]-radii[side];touch=height<=height.min()+1e-6
   local=points[side,touch].mean(0);radius=float(radii[side,touch].mean())
   if last_local is None:
    ankle=position[k,side].copy();ankle[2]=-height.min()
   else:ankle=last_anchor-R@last_local+np.array([0.,0.,last_radius])
   if last_local is None or np.linalg.norm(local-last_local)>1e-5 or abs(radius-last_radius)>1e-6:
    last_anchor=ankle+R@local-np.array([0.,0.,radius]);last_anchor[2]=0.;last_local=local;last_radius=radius
   ankle=last_anchor-R@local+np.array([0.,0.,radius]);target[k,side]=ankle
   if abs(pitch*blend)<1e-7:
    kind[k,side]=0;support[k,side]=ankle;support[k,side,2]=0.
   else:
    kind[k,side]=1;support[k,side]=last_anchor
    halfwidth[k,side]=min(.025,float(np.ptp(points[side,touch,1]))/2)
 for lo,hi in zip(cuts[:-1],cuts[1:]):
  if active[lo]:continue
  a=max(0,lo-1);b=min(len(q)-1,hi);span=b-a
  if span<=0:continue
  start=target[a,side]-position[a,side];end=target[b,side]-position[b,side]
  rs=(Rotation.from_matrix(target_R[a,side])*Rotation.from_matrix(rotation[a,side]).inv()).as_rotvec()
  re=(Rotation.from_matrix(target_R[b,side])*Rotation.from_matrix(rotation[b,side]).inv()).as_rotvec()
  for k in range(a,b+1):
   u=(k-a)/span;t0=min(1.,u/.2);t1=min(1.,(1-u)/.2)
   w0=0. if lo==0 else 1-t0**3*(10-15*t0+6*t0*t0);w1=0. if hi==len(q) else 1-t1**3*(10-15*t1+6*t1*t1)
   target[k,side]=position[k,side]+w0*start+w1*end
   target_R[k,side]=(Rotation.from_rotvec(w0*rs+w1*re)*Rotation.from_matrix(rotation[k,side])).as_matrix()
   bottom=np.min((target_R[k,side]@points[side].T)[2]-radii[side]);clear=.035*16*u*u*(1-u)**2
   target[k,side,2]=max(target[k,side,2],clear-bottom)
R_yaw=Rotation.from_euler('z',yaw.ravel()).as_matrix().reshape(len(q),2,3,3)
full_G,full_b=build_single_support_wrench_constraints(contact_mode=torch.full((len(q),),3,dtype=torch.long),foot_rotation_world=torch.tensor(R_yaw,dtype=torch.float32),
 friction_coefficient=.5,yaw_friction_coefficient=.015,toe_length=.12,heel_length=.05,half_width=.025,sole_depth=0.,coupled_yaw=True,maximum_normal_force=800.)
G=full_G.numpy().reshape(len(q),2,20,12);bb=full_b.numpy().reshape(len(q),2,20);cones=np.stack([G[:,side,:,6*side:6*side+6] for side in range(2)],axis=1)
for width in [0.,.025]:
 line_G,line_b=build_single_support_wrench_constraints(contact_mode=torch.full((len(q),),3,dtype=torch.long),foot_rotation_world=torch.tensor(R_yaw,dtype=torch.float32),
 friction_coefficient=.5,yaw_friction_coefficient=.015,toe_length=0.,heel_length=0.,half_width=width,sole_depth=0.,coupled_yaw=True,maximum_normal_force=800.)
 line=line_G.numpy().reshape(len(q),2,20,12)
 for side in range(2):
  select=(kind[:,side]==1)&((halfwidth[:,side]<.024) if width==0 else (halfwidth[:,side]>=.024))
  cones[select,side]=line[select,side,:,6*side:6*side+6]
np.savez_compressed(p/'rolling.foot_targets.npz',position=target,rotation=target_R,source_position=position,contact=contact,source_contact=contact,support=support,kind=kind,cones=cones,cone_bounds=bb)
trajectory=plan_reference(f,MotionInterpolator(q,method='pchip'),contact,fps,G1_JOINT_NAMES,p/'rolling.npz',foot_targets=(target,target_R),support_geometry=(support,cones,bb))
report=dict(peak_reference_joint_rmse_rad=float((trajectory[:,7:]-q[:,7:]).square().mean(-1).sqrt().max()),peak_root_offset_m=float((trajectory[:,:3]-q[:,:3]).norm(dim=-1).max()),source_frames=len(q),full_tracking_passed=False)
(p/'rolling.adaptation.json').write_text(json.dumps(report,indent=2)+'\n');print(report,flush=True)
