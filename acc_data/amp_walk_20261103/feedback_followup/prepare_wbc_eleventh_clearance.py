"""Reduce excess eleventh swing clearance while retaining foot XY and orientation."""
from pathlib import Path
import numpy as np, json, torch, hashlib
from scipy.spatial.transform import Rotation
from fabrics_sim.reference.interpolation import MotionInterpolator
from numpy_leg_kinematics import LegKinematics
from bounded_leg_ik import solve_leg_pose
from blended_spline import BlendedSplineReference
p=Path(__file__).parent
base=np.load(p/'turn_recovery_y40.npz')['control_q'];source=np.load(p/'fixed_smooth_early.control_reference.npz')['source_q']
d=np.load(p/'wrench_fixed_ten.npz');limits=d['joint_position_limits'][0];speed_limits=d['history_joint_velocity_limits'][-1].reshape(-1)
model=LegKinematics('src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf',d['final_joint_order'])
plan=json.loads((p/'wrench_fixed_ten.steps_plan.json').read_text());points=np.array(plan['collision_points'][0]);radii=np.array(plan['collision_radii'][0])
def smooth(x):
 x=np.clip(x,0,1);return x**3*(10-15*x+6*x*x)
for drop in [0.,.01,.02]:
 out=base.copy();errors=[]
 for f in range(810,961):
  q=base[f].astype(float);w=smooth((f-810)/20)*(1-smooth((f-870)/20));q[2]-=drop*smooth((f-810)/30)*(1-smooth((f-900)/60))
  for side in [0,1]:
   pos,R,_=model.evaluate(base[f],side)
   target=pos.copy();target_R=R
   if side==0:
    height=pos[2]+np.min((points@R.T)[:,2]-radii)
    target[2]-=.35*w*max(height,0.)
   q,pe,re=solve_leg_pose(model,q,side,target,target_R);errors.append([f,side,pe,re])
  out[f]=q
 curve=BlendedSplineReference(MotionInterpolator(torch.tensor(out)),torch.tensor(out),stride=6,start_frame=130,ramp_frames=20)
 samples=[curve.kinematics(float(f),240.) for f in np.arange(804,968,.2)]
 dense=np.array([r[0].numpy()[0] for r in samples]);vel=np.array([r[1].numpy()[0,6:] for r in samples])
 margin=float(np.minimum(dense[:,7:]-limits[:,0],limits[:,1]-dense[:,7:]).min())
 clearance=[]
 for q in dense:
  pos,R,_=model.evaluate(q,0);clearance.append(pos[2]+np.min((points@R.T)[:,2]-radii))
 error=np.array(errors);report=dict(root_drop=drop,peak_position_residual=float(error[:,2].max()),peak_orientation_residual=float(error[:,3].max()),peak_source_joint_reference_rmse=float(np.sqrt(np.mean((out[:,7:]-source[:,7:])**2,axis=1)).max()),minimum_dense_joint_margin=margin,minimum_dense_native_clearance=float(min(clearance)),maximum_joint_velocity_fraction=float(np.max(abs(vel)/speed_limits)),physical_tracking_certified=False)
 report['preflight_passed']=bool(report['peak_position_residual']<.002 and report['peak_orientation_residual']<.01 and report['peak_source_joint_reference_rmse']<.19 and margin>=0 and min(clearance)>=-.002 and report['maximum_joint_velocity_fraction']<=1)
 name=f'wbc_eleventh_clearance065_drop{round(drop*1000)}';np.savez_compressed(p/(name+'.npz'),control_q=out,fps=200.,baseline_sha256=hashlib.sha256(np.load(p/'fixed_smooth_early.control_reference.npz')['control_q'].tobytes()).hexdigest(),source_sha256=hashlib.sha256(source.tobytes()).hexdigest());(p/(name+'.json')).write_text(json.dumps(report,indent=2)+'\n');print(name,report,flush=True)
