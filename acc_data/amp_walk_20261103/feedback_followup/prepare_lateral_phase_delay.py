"""Shift COM before the turn; allow bounded sole rotation with original clearance.
This changes a control reference, not source samples or tracking tolerances.
"""
from pathlib import Path
import numpy as np,json,hashlib,torch
from scipy.spatial.transform import Rotation
from scipy.interpolate import CubicSpline
from fabrics_sim.reference.interpolation import MotionInterpolator
from numpy_leg_kinematics import LegKinematics
from bounded_leg_ik import solve_leg_pose
from blended_spline import BlendedSplineReference
p=Path(__file__).parent;baseline=np.load(p/'fixed_smooth_early.control_reference.npz');source=baseline['source_q'];base=np.load(p/'tenth_landing_forward15_right20.npz')['control_q'];d=np.load(p/'wrench_fixed_ten.npz');limits=d['joint_position_limits'][0];model=LegKinematics('src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf',d['final_joint_order']);model.lower+=.01;model.upper-=.01;plan=json.load(open(p/'wrench_fixed_ten.steps_plan.json'));geometry=[(np.array(plan['collision_points'][j]),np.array(plan['collision_radii'][j])) for j in [0,1]]
lateral=CubicSpline(np.arange(len(base)),base[:,1])
def smooth(x):x=np.clip(x,0,1);return x**3*(10-15*x+6*x*x)
for delay in [12.,16.]:
 name='lateral_phase_delay'+str(int(delay));out=base.copy();records=[]
 for f in range(580,len(base)):
  original=base[f].astype(float);q=original.copy();w=smooth((f-580)/100);q[1]=float(lateral(f-delay*w));shift=q[1]-original[1];q[2]-=.2*(np.sqrt(shift*shift+1e-6)-.001);errors=[]
  if w==0:continue
  for side in [0,1]:
   pos,R,_=model.evaluate(original,side);points,radii=geometry[side];clearance=pos[2]+np.min((points@R.T)[:,2]-radii);target=pos.copy()
   for _ in range(8):
    q,pe,re=solve_leg_pose(model,q,side,target,R,orientation_weight=.05);actual,actualR,_=model.evaluate(q,side);actual_clearance=actual[2]+np.min((points@actualR.T)[:,2]-radii);delta=clearance-actual_clearance
    if abs(delta)<1e-6:break
    target[2]+=delta
   errors.append((np.linalg.norm(actual[:2]-pos[:2]),abs(delta),np.linalg.norm(actual-pos),Rotation.from_matrix(actualR@R.T).magnitude()))
  out[f]=q;records.append([f,*np.max(errors,axis=0)])
 records=np.array(records);rmse=np.sqrt(np.mean((out[:,7:]-source[:,7:])**2,axis=1));root=np.linalg.norm(out[:,:3]-source[:,:3],axis=1);t=torch.tensor(out);curve=BlendedSplineReference(MotionInterpolator(t),t,stride=6,start_frame=130,ramp_frames=20);samples=[curve.kinematics(f,240) for f in np.arange(574,1999.01,.5)];dense=np.array([x[0].numpy()[0] for x in samples]);vel=np.array([x[1].numpy()[0,6:] for x in samples]);margin=float(np.minimum(dense[:,7:]-limits[:,0],limits[:,1]-dense[:,7:]).min());minimum_clearance=1.
 for q in dense:
  for side in [0,1]:
   pos,R,_=model.evaluate(q,side);points,radii=geometry[side];minimum_clearance=min(minimum_clearance,float(pos[2]+np.min((points@R.T)[:,2]-radii)))
 report=dict(source_frame_delay=delay,maximum_root_y_shift=float(abs(out[:,1]-base[:,1]).max()),maximum_root_drop=float(abs(out[:,2]-base[:,2]).max()),peak_foot_xy_error=float(records[:,1].max()),peak_native_clearance_error=float(records[:,2].max()),peak_foot_position_adjustment=float(records[:,3].max()),peak_foot_orientation_adjustment=float(records[:,4].max()),peak_source_joint_reference_rmse=float(rmse.max()),peak_root_position_change=float(root.max()),minimum_dense_joint_limit_margin=margin,minimum_dense_native_clearance=minimum_clearance,maximum_dense_joint_velocity=float(abs(vel).max()),physical_tracking_certified=False,native_floor_repair_required=minimum_clearance<-.0015)
 report['preflight_passed']=bool(not report['native_floor_repair_required'] and records[:,1].max()<.002 and records[:,2].max()<.001 and records[:,3].max()<.012 and records[:,4].max()<.15 and rmse.max()<.19 and root.max()<.19 and margin>.005 and minimum_clearance>-.0015 and abs(vel).max()<20);np.savez_compressed(p/(name+'.npz'),control_q=out,fps=200.,baseline_sha256=hashlib.sha256(baseline['control_q'].tobytes()).hexdigest(),source_sha256=hashlib.sha256(source.tobytes()).hexdigest());(p/(name+'.json')).write_text(json.dumps(report,indent=2)+'\n');print(name,report,flush=True)
