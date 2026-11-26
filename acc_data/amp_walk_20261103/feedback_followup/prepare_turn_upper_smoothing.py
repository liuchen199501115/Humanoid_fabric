"""Smooth the turn's pelvis/upper-body angles, preserving both world foot poses."""
from pathlib import Path
import numpy as np,json,hashlib,torch
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation
from fabrics_sim.reference.interpolation import MotionInterpolator
from numpy_leg_kinematics import LegKinematics
from bounded_leg_ik import solve_leg_pose
from blended_spline import BlendedSplineReference
p=Path(__file__).parent;baseline=np.load(p/'fixed_smooth_early.control_reference.npz');source=baseline['source_q'];base=np.load(p/'turn_recovery_y40.npz')['control_q'];d=np.load(p/'wrench_fixed_ten.npz');limits=d['joint_position_limits'][0];model=LegKinematics('src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf',d['final_joint_order']);filtered=savgol_filter(base,61,3,axis=0)
def smooth(x):x=np.clip(x,0,1);return x**3*(10-15*x+6*x*x)
for drop in [0.,.01]:
 name='turn_upper_smooth61_drop'+str(int(drop*1000));out=base.copy();records=[]
 for f in range(740,1021):
  original=base[f].astype(float);q=original.copy();w=smooth((f-740)/40)*(1-smooth((f-960)/60));q[3:7]+=w*(filtered[f,3:7]-original[3:7]);q[3:7]/=np.linalg.norm(q[3:7]);q[19:]+=w*(filtered[f,19:]-original[19:]);q[2]-=drop*w;errors=[]
  for side in [0,1]:
   pos,R,_=model.evaluate(original,side);q,pe,re=solve_leg_pose(model,q,side,pos,R);errors.append((pe,re))
  out[f]=q;records.append([f,*np.max(errors,axis=0)])
 records=np.array(records);rmse=np.sqrt(np.mean((out[:,7:]-source[:,7:])**2,axis=1));t=torch.tensor(out);curve=BlendedSplineReference(MotionInterpolator(t),t,stride=6,start_frame=130,ramp_frames=20);samples=[curve.kinematics(f,240) for f in np.arange(734,1027,.2)];dense=np.array([x[0].numpy()[0] for x in samples]);vel=np.array([x[1].numpy()[0,6:] for x in samples]);margin=float(np.minimum(dense[:,7:]-limits[:,0],limits[:,1]-dense[:,7:]).min());angular=(Rotation.from_quat(out[:,3:7])*Rotation.from_quat(source[:,3:7]).inv()).magnitude();report=dict(root_drop=drop,peak_position_residual=float(records[:,1].max()),peak_orientation_residual=float(records[:,2].max()),peak_source_joint_reference_rmse=float(rmse.max()),peak_source_root_orientation_delta=float(angular.max()),minimum_dense_joint_limit_margin=margin,maximum_dense_joint_velocity=float(abs(vel).max()),physical_tracking_certified=False)
 report['preflight_passed']=bool(records[:,1].max()<.002 and records[:,2].max()<.01 and rmse.max()<.19 and angular.max()<.3 and margin>=0 and abs(vel).max()<20);np.savez_compressed(p/(name+'.npz'),control_q=out,fps=200.,baseline_sha256=hashlib.sha256(baseline['control_q'].tobytes()).hexdigest(),source_sha256=hashlib.sha256(source.tobytes()).hexdigest());(p/(name+'.json')).write_text(json.dumps(report,indent=2)+'\n');print(name,report,flush=True)
