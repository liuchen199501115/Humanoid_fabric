"""Offline root-orientation smoothing with both original foot poses restored by IK."""
from pathlib import Path
import json
import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation
from numpy_leg_kinematics import LegKinematics
from bounded_leg_ik import solve_leg_pose
from blended_spline import BlendedSplineReference
from reference_joint_extrema import joint_extrema
from mujoco_reference_dynamics import ReferenceDynamics
from fabrics_sim.reference.interpolation import MotionInterpolator
p=Path(__file__).parent;name='step13_root_smooth6_feet_fixed'
if (p/(name+'.json')).exists():raise RuntimeError('Existing report')
parent=np.load(p/'step13_leg_smooth5_clearance_audit.npz');base=parent['control_q'].astype(float);out=base.copy();source=np.load(p/'rolling_contact.control_reference.npz')['source_q'];record=np.load(p/'rolling_contact.npz');limits=record['joint_position_limits'][0];speed=record['history_joint_velocity_limits'][-1].reshape(-1)
urdf='src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf';leg=LegKinematics(urdf,record['final_joint_order']);model=ReferenceDynamics(urdf,record['final_joint_order'])
quats=base[:,3:7].copy()
for i in range(1,len(quats)):
 if quats[i-1]@quats[i]<0:quats[i]*=-1
filtered=gaussian_filter1d(quats,6.,axis=0,mode='nearest');filtered/=np.linalg.norm(filtered,axis=1)[:,None]
x=np.arange(len(base));up=np.clip((x-900)/30,0,1);down=np.clip((1092-x)/30,0,1);blend=up**3*(10-15*up+6*up*up)*down**3*(10-15*down+6*down*down)
quats+=(filtered-quats)*blend[:,None];quats/=np.linalg.norm(quats,axis=1)[:,None]
ik=[]
for f in range(901,1092):
 q=base[f].copy();q[3:7]=quats[f]
 for foot in (0,1):
  pos,R,_=leg.evaluate(base[f],foot);q,pe,re=solve_leg_pose(leg,q,foot,pos,R);ik.append([pe,re])
 out[f]=q

def curve(q):
 t=torch.tensor(q,dtype=torch.float64);return BlendedSplineReference(MotionInterpolator(t),t,stride=6,start_frame=130,ramp_frames=20)
fs=np.arange(894,1098.01,.5);sampler=curve(out);old=curve(base);foot_errors=[];angular=[];old_angular=[];root_angles=[]
plan=json.loads((p/'rolling_contact.steps_plan.json').read_text());geometry=[(np.array(plan['collision_points'][j]),np.array(plan['collision_radii'][j])) for j in (0,1)];clear=[];oldclear=[]
for f in fs:
 q,v,a=[x.numpy()[0] for x in sampler.kinematics(float(f),240.)];b,bv,ba=[x.numpy()[0] for x in old.kinematics(float(f),240.)]
 angular.append(float(np.linalg.norm(a[3:6])));old_angular.append(float(np.linalg.norm(ba[3:6])));root_angles.append(float((Rotation.from_quat(q[3:7])*Rotation.from_quat(b[3:7]).inv()).magnitude()))
 for foot in (0,1):
  pos,R,_=leg.evaluate(q,foot);bp,bR,_=leg.evaluate(b,foot);foot_errors.append([np.linalg.norm(pos-bp),Rotation.from_matrix(R@bR.T).magnitude()]);pts,radius=geometry[foot];clear.append(pos[2]+min((pts@R.T)[:,2]-radius));oldclear.append(bp[2]+min((pts@bR.T)[:,2]-radius))
report=dict(parent='step13_leg_smooth5_clearance_audit',scope='Kinematic reference smoothing only; no force or physical tracking certificate',maximum_ik_errors=np.max(ik,axis=0).tolist(),maximum_runtime_foot_errors=np.max(foot_errors,axis=0).tolist(),maximum_root_orientation_change_rad=max(root_angles),maximum_root_angular_acceleration=max(angular),parent_maximum_root_angular_acceleration=max(old_angular),peak_raw_joint_rmse=float(np.sqrt(np.mean((out[:,7:]-source[:,7:])**2,axis=1)).max()),minimum_clearance_m=float(min(clear)),minimum_parent_clearance_m=float(min(oldclear)),physical_tracking_certified=False)
report.update(joint_extrema(sampler.smooth,894,1098,240.,limits,speed))
report['kinematic_preflight_passed']=bool(max(np.array(foot_errors)[:,0])<.002 and max(np.array(foot_errors)[:,1])<.01 and max(root_angles)<.03 and min(clear)>=min(oldclear)-.0005 and report['peak_raw_joint_rmse']<.18 and report['minimum_continuous_joint_margin']>0 and report['maximum_continuous_joint_velocity_fraction']<=1)
(p/(name+'.json')).write_text(json.dumps(report,indent=2)+'\n')
if report['kinematic_preflight_passed']:np.savez_compressed(p/(name+'.npz'),control_q=out,fps=200.,baseline_sha256=parent['baseline_sha256'],source_sha256=parent['source_sha256'])
print(json.dumps({k:v for k,v in report.items() if not k.startswith('joint_')},indent=2))
