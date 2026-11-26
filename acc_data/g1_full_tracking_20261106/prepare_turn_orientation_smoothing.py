"""Offline turn angular smoothing, retaining guide feet with bounded leg IK.
No physical tracking claim follows from this kinematic/necessary-wrench audit.
"""
from pathlib import Path
import json
import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from numpy_leg_kinematics import LegKinematics
from bounded_leg_ik import solve_leg_pose
from blended_spline import BlendedSplineReference
from fabrics_sim.reference.interpolation import MotionInterpolator
from reference_joint_extrema import joint_extrema
from mujoco_reference_dynamics import ReferenceDynamics
p=Path(__file__).parent
parent=np.load('acc_data/amp_walk_20261103/feedback_followup/wbc_convex_catch15_25_clear35.npz')
base=parent['control_q'].astype(float);source=np.load(p/'rolling_contact.control_reference.npz')['source_q'];record=np.load(p/'rolling_contact.npz')
names=record['final_joint_order'];urdf='src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf'
model=LegKinematics(urdf,names);dyn=ReferenceDynamics(urdf,names)
plan=json.loads((p/'rolling_contact.steps_plan.json').read_text())
qrot=base[:,3:7].copy()
for i in range(1,len(qrot)):
 if qrot[i]@qrot[i-1]<0:qrot[i]*=-1
filtered=gaussian_filter1d(qrot,16.,axis=0,mode='nearest');filtered/=np.linalg.norm(filtered,axis=1)[:,None]
upper=gaussian_filter1d(base[:,19:],16.,axis=0,mode='nearest')
def smooth(u):
 u=np.clip(u,0,1);return u**3*(10-15*u+6*u*u)
def curve(q):
 t=torch.tensor(q,dtype=torch.float64);return BlendedSplineReference(MotionInterpolator(t),t,stride=6,start_frame=130,ramp_frames=20)
frames=np.arange(len(base));blend=smooth((frames-810)/50)*smooth((1180-frames)/60)
rot_error=(Rotation.from_quat(filtered)*Rotation.from_quat(qrot).inv()).as_rotvec()
# Limit the reference angular departure smoothly; preserve source audits.
norm=np.linalg.norm(rot_error,axis=1);scale=.12/np.sqrt(.12**2+norm**2)
out=base.copy();out[:,3:7]=(Rotation.from_rotvec(rot_error*(blend*scale)[:,None])*Rotation.from_quat(qrot)).as_quat()
out[:,19:]=base[:,19:]+blend[:,None]*(upper-base[:,19:]);errors=[]
for f in np.flatnonzero(blend):
 q=out[f].copy()
 for side in (0,1):
  pos,R,_=model.evaluate(base[f],side);q,pe,re=solve_leg_pose(model,q,side,pos,R,position_tolerance=1e-9,orientation_tolerance=1e-8);errors.append([pe,re])
 out[f]=q
reference=curve(out);previous=curve(base)
continuous=joint_extrema(reference.smooth,804.,1186.,240.,record['joint_position_limits'][0],record['history_joint_velocity_limits'][-1].reshape(-1))
feet_delta=[];clear=[];original_clear=[];normal=[];moment=[];old_normal=[];old_moment=[]
for f in np.arange(804.,1186.01,.5):
 state=[v.numpy()[0] for v in reference.kinematics(float(f),240.)];oldstate=[v.numpy()[0] for v in previous.kinematics(float(f),240.)]
 q,old=state[0],oldstate[0]
 for side in (0,1):
  pos,R,_=model.evaluate(q,side);op,oR,_=model.evaluate(old,side);points=np.array(plan['collision_points'][side]);radii=np.array(plan['collision_radii'][side])
  feet_delta.append(np.linalg.norm(pos-op));clear.append(pos[2]+min((points@R.T)[:,2]-radii));original_clear.append(op[2]+min((points@oR.T)[:,2]-radii))
 if f%2==0:
  F=dyn.evaluate(*state)['generalized_force'][:6];oldF=dyn.evaluate(*oldstate)['generalized_force'][:6]
  normal.append(F[2]);old_normal.append(oldF[2]);moment.append(F[3:]);old_moment.append(oldF[3:])
errors=np.array(errors);moment=np.array(moment);old_moment=np.array(old_moment)
report=dict(parent='wbc_convex_catch15_25_clear35',sigma_frames=16,blend_start=810,blend_full=860,blend_end=1180,
 peak_ik_position=float(errors[:,0].max()),peak_ik_orientation=float(errors[:,1].max()),maximum_dense_foot_change=float(max(feet_delta)),minimum_native_clearance=float(min(clear)),minimum_parent_clearance=float(min(original_clear)),
 peak_source_joint_reference_rmse=float(np.sqrt(((out[:,7:]-source[:,7:])**2).mean(1)).max()),peak_source_root_orientation=float((Rotation.from_quat(out[:,3:7])*Rotation.from_quat(source[:,3:7]).inv()).magnitude().max()),
 minimum_reference_normal_force=float(min(normal)),minimum_parent_normal_force=float(min(old_normal)),rms_reference_root_moment=np.sqrt((moment**2).mean(0)).tolist(),rms_parent_root_moment=np.sqrt((old_moment**2).mean(0)).tolist(),
 physical_tracking_certified=False,**continuous)
report['preflight_passed']=bool(errors[:,0].max()<.002 and errors[:,1].max()<.01 and max(feet_delta)<.002 and min(clear)>=min(original_clear)-.0003 and continuous['minimum_continuous_joint_margin']>=0 and continuous['maximum_continuous_joint_velocity_fraction']<=1 and report['peak_source_joint_reference_rmse']<.19 and report['peak_source_root_orientation']<.15)
path=p/'turn_angular_smooth16';path.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if not k.startswith('joint_')},indent=2),flush=True)
if not report['preflight_passed']:raise RuntimeError('Angular reference candidate rejected before physical trial')
np.savez_compressed(path.with_suffix('.npz'),control_q=out,fps=200.,baseline_sha256=parent['baseline_sha256'],source_sha256=parent['source_sha256'])
