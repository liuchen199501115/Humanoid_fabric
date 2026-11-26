"""Advance the eleventh swing with knee flexion; preserve the incoming left foot world pose."""
from pathlib import Path
import numpy as np,json,hashlib,torch
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation,Slerp
from fabrics_sim.reference.interpolation import MotionInterpolator
from numpy_leg_kinematics import LegKinematics
from bounded_leg_ik import solve_leg_pose
from blended_spline import BlendedSplineReference
p=Path(__file__).parent;baseline=np.load(p/'fixed_smooth_early.control_reference.npz');source=baseline['source_q'];base=np.load(p/'turn_recovery_y40.npz')['control_q'];d=np.load(p/'wrench_fixed_ten.npz');limits=d['joint_position_limits'][0];model=LegKinematics('src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf',d['final_joint_order']);frames=np.arange(820,1042);poses=[model.evaluate(base[f],0)[:2] for f in frames];path=CubicSpline(frames,np.array([x[0] for x in poses]));rotations=Slerp(frames,Rotation.from_matrix(np.array([x[1] for x in poses])))
def smooth(x):x=np.clip(x,0,1);return x**3*(10-15*x+6*x*x)
for lead,drop in [(8.,.02),(12.,.03)]:
 name='eleventh_recovery_lead'+str(int(lead))+'_drop'+str(int(drop*1000));out=base.copy();records=[]
 for f in range(820,1041):
  original=base[f].astype(float);q=original.copy();w=smooth((f-834)/20)*(1-smooth((f-884)/24));q[2]-=drop*smooth((f-820)/40)*(1-smooth((f-980)/60));errors=[]
  for side in [0,1]:
   pos,R,_=model.evaluate(original,side)
   if side==0:pos=path(f+lead*w);R=rotations(f+lead*w).as_matrix()
   q,pe,re=solve_leg_pose(model,q,side,pos,R);errors.append((pe,re))
  out[f]=q;records.append([f,*np.max(errors,axis=0)])
 records=np.array(records);rmse=np.sqrt(np.mean((out[:,7:]-source[:,7:])**2,axis=1));t=torch.tensor(out);curve=BlendedSplineReference(MotionInterpolator(t),t,stride=6,start_frame=130,ramp_frames=20);samples=[curve.kinematics(f,240) for f in np.arange(814,1047,.2)];dense=np.array([x[0].numpy()[0] for x in samples]);vel=np.array([x[1].numpy()[0,6:] for x in samples]);margin=float(np.minimum(dense[:,7:]-limits[:,0],limits[:,1]-dense[:,7:]).min());report=dict(lead_frames=lead,root_drop=drop,peak_position_residual=float(records[:,1].max()),peak_orientation_residual=float(records[:,2].max()),peak_source_joint_reference_rmse=float(rmse.max()),minimum_dense_joint_limit_margin=margin,maximum_dense_joint_velocity=float(abs(vel).max()),physical_tracking_certified=False)
 report['preflight_passed']=bool(records[:,1].max()<.002 and records[:,2].max()<.01 and rmse.max()<.19 and margin>=0 and abs(vel).max()<20);np.savez_compressed(p/(name+'.npz'),control_q=out,fps=200.,baseline_sha256=hashlib.sha256(baseline['control_q'].tobytes()).hexdigest(),source_sha256=hashlib.sha256(source.tobytes()).hexdigest());(p/(name+'.json')).write_text(json.dumps(report,indent=2)+'\n');print(name,report,flush=True)
