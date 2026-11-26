"""Bounded root lateral adjustment to reduce turn capture-point overshoot.
Both source-derived foot world poses are retained; q/v/a use one interpolator.
"""
from pathlib import Path
import numpy as np,json,hashlib,torch
from fabrics_sim.reference.interpolation import MotionInterpolator
from numpy_leg_kinematics import LegKinematics
from bounded_leg_ik import solve_leg_pose
from blended_spline import BlendedSplineReference
p=Path(__file__).parent;baseline=np.load(p/'fixed_smooth_early.control_reference.npz');source=baseline['source_q'];base=np.load(p/'tenth_landing_forward15_right20.npz')['control_q'];d=np.load(p/'wrench_fixed_ten.npz');limits=d['joint_position_limits'][0];model=LegKinematics('src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf',d['final_joint_order'])
def smooth(x):x=np.clip(x,0,1);return x**3*(10-15*x+6*x*x)
for offset in [.06,.08]:
 name='turn_recovery_y'+str(int(offset*1000));out=base.copy();records=[]
 for f in range(740,1001):
  original=base[f].astype(float);q=original.copy();w=smooth((f-820)/60)*(1-smooth((f-940)/60));q[1]+=offset*w;errors=[]
  for side in [0,1]:
   pos,R,_=model.evaluate(original,side);q,pe,re=solve_leg_pose(model,q,side,pos,R);errors.append((pe,re))
  out[f]=q;records.append([f,*np.max(errors,axis=0)])
 records=np.array(records);rmse=np.sqrt(np.mean((out[:,7:]-source[:,7:])**2,axis=1));root=np.linalg.norm(out[:,:3]-source[:,:3],axis=1);t=torch.tensor(out);curve=BlendedSplineReference(MotionInterpolator(t),t,stride=6,start_frame=130,ramp_frames=20);dense=np.array([curve.kinematics(f,240)[0].numpy()[0] for f in np.arange(734,1007,.2)]);margin=float(np.minimum(dense[:,7:]-limits[:,0],limits[:,1]-dense[:,7:]).min());report=dict(root_y_offset=offset,peak_position_residual=float(records[:,1].max()),peak_orientation_residual=float(records[:,2].max()),peak_source_joint_reference_rmse=float(rmse.max()),peak_root_position_change=float(root.max()),minimum_dense_joint_limit_margin=margin,physical_tracking_certified=False)
 report['preflight_passed']=bool(records[:,1].max()<.002 and records[:,2].max()<.01 and rmse.max()<.19 and root.max()<.19 and margin>=0);np.savez_compressed(p/(name+'.npz'),control_q=out,fps=200.,baseline_sha256=hashlib.sha256(baseline['control_q'].tobytes()).hexdigest(),source_sha256=hashlib.sha256(source.tobytes()).hexdigest());(p/(name+'.json')).write_text(json.dumps(report,indent=2)+'\n');print(name,report,flush=True)
