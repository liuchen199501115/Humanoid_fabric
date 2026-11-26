from pathlib import Path
import json,numpy as np,torch,warp as wp
from scipy.spatial.transform import Rotation
from test_g1_two_tasks import load_motion_trajectory,load_motion_contact_sequence,G1_JOINT_NAMES
from fabrics_sim.fabrics.g1_pose_fabric import G1PoseFabric
from fabrics_sim.reference.interpolation import MotionInterpolator
from momentum_planner import plan_reference
wp.config.kernel_cache_dir='/tmp/g1_warp_cache';wp.init()
f=G1PoseFabric(1,'cpu',robot_name='g1_29dof_rev_1_0',task_profile='dynamics-aware')
q,fps=load_motion_trajectory(Path('acc_data/amp_walk_20261102/amp_walk_native_contacts.pkl'))
with torch.no_grad():
 st0=f.dynamics_tasks.model.snapshot(q[:1])
 ids0=[f.dynamics_tasks.model.kinematics.get_link_index(n+'_ankle_roll_link') for n in ['left','right']]
 corners=q.new_tensor([[-.05,-.025,-.035],[-.05,.025,-.035],[.12,-.025,-.035],[.12,.025,-.035]])
 world=st0['transforms'][:,ids0,:3,None]+st0['rotations'][:,ids0]@corners.T
 ground_shift=-float(world[:,:,2].min())
 q[:,2]+=ground_shift
contact=load_motion_contact_sequence(Path('acc_data/amp_walk_20261102/amp_walk_native_contacts.pkl')).numpy()
model=f.dynamics_tasks.model;ids=[model.kinematics.get_link_index(n+'_ankle_roll_link') for n in ['left','right']]
position=[];rotation=[]
with torch.no_grad():
 for frame in range(len(q)):
  st=model.snapshot(q[frame:frame+1]);position.append(st['transforms'][0,ids,:3].numpy());rotation.append(st['rotations'][0,ids].numpy())
position=np.asarray(position);rotation=np.asarray(rotation)
source_contact=contact.copy()
for side in range(2):
 active=source_contact[:,side];cuts=np.r_[0,np.flatnonzero(active[1:]!=active[:-1])+1,len(q)]
 for lo,hi in zip(cuts[:-1],cuts[1:]):
  if not active[lo] or hi==len(q):continue
  floor=float(position[lo:hi,side,2].min())
  for frame in range(max(lo+10,hi-40),hi):
   if (position[frame,side,2]>floor+.02 and contact[frame:hi,1-side].all()
       and position[min(frame+3,hi-1),side,2]>=position[frame,side,2]):
    contact[frame:hi,side]=False;break
if (~contact.any(1)).any():raise RuntimeError('Flat contact adaptation introduced unsupported flight')
target=position.copy();orientation_blend=np.zeros((len(q),2));yaw=np.unwrap(np.arctan2(-rotation[:,:,0,1],rotation[:,:,1,1]),axis=0);target_yaw=yaw.copy()
segments=[]
for side in range(2):
 active=contact[:,side];cuts=np.r_[0,np.flatnonzero(active[1:]!=active[:-1])+1,len(q)]
 for lo,hi in zip(cuts[:-1],cuts[1:]):
  if not active[lo]:continue
  # Use the low, flat portion of this observed stance, before toe rollover.
  rows=np.arange(lo,hi);height=position[rows,side,2];rows=rows[height<=height.min()+.008]
  anchor=np.median(position[rows,side],axis=0);anchor[2]=.035
  target[lo:hi,side]=anchor;target_yaw[lo:hi,side]=np.median(yaw[rows,side]);segments.append([side,int(lo),int(hi),anchor.tolist()])
  for frame in range(lo,min(lo+12,hi)):
   phase=min(1.,(frame-lo)/12);blend=phase**3*(10-15*phase+6*phase*phase)
   target[frame,side]=position[frame,side]*(1-blend)+anchor*blend
   orientation_blend[frame,side]=1-blend
 for lo,hi in zip(cuts[:-1],cuts[1:]):
  if active[lo]:continue
  a=max(0,lo-1);b=min(len(q)-1,hi);span=b-a
  # Smooth rest-to-rest swing, follows the source path in the middle.
  if span<=0:continue
  u=np.arange(b-a+1)/span;h=u**3*(10-15*u+6*u*u)
  start=target[a,side].copy();end=target[b,side].copy()
  yaw_start=target_yaw[a,side];yaw_end=target_yaw[b,side]
  for j,k in enumerate(range(a,b+1)):
   u0=min(1.,u[j]/.25);u1=min(1.,(1-u[j])/.25)
   w0=1-u0**3*(10-15*u0+6*u0*u0);w1=1-u1**3*(10-15*u1+6*u1*u1)
   if lo==0:w0=0.
   orientation_blend[k,side]=(1-w0)
   target[k,side]=position[k,side]*(1-w0-w1)+w0*start+w1*end
   target[k,side,2]=max(target[k,side,2],.035+.05*16*u[j]**2*(1-u[j])**2)
   target_yaw[k,side]=yaw[k,side]*(1-w0-w1)+w0*yaw_start+w1*yaw_end
# Yaw-only flat soles. Preserve a source flight's initial foot position.
flat_rotation=Rotation.from_euler('z',target_yaw.ravel()).as_matrix().reshape(len(q),2,3,3)
for side in range(2):
 flat=Rotation.from_matrix(flat_rotation[:,side]);source=Rotation.from_matrix(rotation[:,side])
 delta=(source*flat.inv()).as_rotvec()*orientation_blend[:,side,None]
 flat_rotation[:,side]=(Rotation.from_rotvec(delta)*flat).as_matrix()
corners=np.array([[-.05,-.025,-.035],[-.05,.025,-.035],[.12,-.03,-.035],[.12,.03,-.035]])
for side in range(2):
 surface=(flat_rotation[:,side]@corners.T)[:,2,:].min(1)
 target[contact[:,side],side,2]=-surface[contact[:,side]]
s=MotionInterpolator(q,method='pchip');out=Path(__file__).with_name('flat_momentum.npz')
np.savez_compressed(out.with_suffix('.foot_targets.npz'),position=target,rotation=flat_rotation,source_position=position,contact=contact,source_contact=source_contact)
trajectory=plan_reference(f,s,contact,fps,G1_JOINT_NAMES,out,foot_targets=(target,flat_rotation))
error=(trajectory[:,7:]-q[:,7:]).square().mean(-1).sqrt()
report=dict(peak_reference_joint_rmse_rad=float(error.max()),peak_root_offset_m=float((trajectory[:,:3]-q[:,:3]).norm(dim=-1).max()),source_frames=len(q),segments=segments,full_tracking_passed=False)
out.with_suffix('.adaptation.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
