"""Retain a landing correction during stance; remove it during the next swing."""
from pathlib import Path
import json, hashlib
import numpy as np, torch
from fabrics_sim.reference.interpolation import MotionInterpolator
from numpy_leg_kinematics import LegKinematics
from bounded_leg_ik import solve_leg_pose
from blended_spline import BlendedSplineReference
p=Path(__file__).parent;input_data=np.load(p/'fixed_smooth_early.control_reference.npz');source=input_data['source_q'];base=np.load(p/'turn_recovery_y40.npz')['control_q'];d=np.load(p/'wrench_fixed_ten.npz')
model=LegKinematics('src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf',d['final_joint_order']);limits=d['joint_position_limits'][0];speed_limits=d['history_joint_velocity_limits'][-1].reshape(-1)
plan=json.loads((p/'wrench_fixed_ten.steps_plan.json').read_text());points=np.array(plan['collision_points'][1]);radii=np.array(plan['collision_radii'][1])
def smooth(x):
 x=np.clip(x,0,1);return x**3*(10-15*x+6*x*x)
out=base.copy();errors=[];offset=np.array([.015,-.02,0.])
for f in range(840,953):
 q=base[f].astype(float);pos,R,_=model.evaluate(q,1)
 old=1-smooth((f-840)/44);new=1-smooth((f-908)/44)
 q,pe,re=solve_leg_pose(model,q,1,pos+(new-old)*offset,R);out[f]=q;errors.append([pe,re])
curve=BlendedSplineReference(MotionInterpolator(torch.tensor(out)),torch.tensor(out),stride=6,start_frame=130,ramp_frames=20)
samples=[curve.kinematics(float(f),240.) for f in np.arange(834,960,.2)];dense=np.array([r[0].numpy()[0] for r in samples]);vel=np.array([r[1].numpy()[0,6:] for r in samples]);margin=float(np.minimum(dense[:,7:]-limits[:,0],limits[:,1]-dense[:,7:]).min());clear=[]
for q in dense:
 pos,R,_=model.evaluate(q,1);clear.append(pos[2]+np.min((points@R.T)[:,2]-radii))
source_error=max(np.linalg.norm(model.evaluate(out[f],1)[0]-model.evaluate(source[f],1)[0]) for f in range(840,909));errors=np.array(errors)
report=dict(retained_offset=offset.tolist(),release_start_frame=908,release_end_frame=952,peak_ik_position_error=float(errors[:,0].max()),peak_ik_rotation_error=float(errors[:,1].max()),maximum_source_stance_foot_error=float(source_error),minimum_dense_joint_margin=margin,minimum_dense_native_clearance=float(min(clear)),maximum_joint_velocity_fraction=float(np.max(abs(vel)/speed_limits)),peak_source_joint_reference_rmse=float(np.sqrt(np.mean((out[:,7:]-source[:,7:])**2,axis=1)).max()),physical_tracking_certified=False)
report['preflight_passed']=bool(errors[:,0].max()<.002 and errors[:,1].max()<.01 and margin>=0 and min(clear)>=-.002 and report['maximum_joint_velocity_fraction']<=1 and report['peak_source_joint_reference_rmse']<.19 and source_error<.04)
name='wbc_stance_offset_until908';np.savez_compressed(p/(name+'.npz'),control_q=out,fps=200.,baseline_sha256=hashlib.sha256(input_data['control_q'].tobytes()).hexdigest(),source_sha256=hashlib.sha256(source.tobytes()).hexdigest());(p/(name+'.json')).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
