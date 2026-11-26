"""Prepare native sole flattening before the eleventh landing with bounded leg IK."""
from pathlib import Path
import numpy as np,json,hashlib,torch
from scipy.spatial.transform import Rotation
from fabrics_sim.reference.interpolation import MotionInterpolator
from numpy_leg_kinematics import LegKinematics
from bounded_leg_ik import solve_leg_pose
from blended_spline import BlendedSplineReference
p=Path(__file__).parent;baseline=np.load(p/'fixed_smooth_early.control_reference.npz');source=baseline['source_q'];base=np.load(p/'turn_recovery_y40.npz')['control_q'];d=np.load(p/'wrench_fixed_ten.npz');limits=d['joint_position_limits'][0];model=LegKinematics('src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf',d['final_joint_order']);plan=json.loads((p/'wrench_fixed_ten.steps_plan.json').read_text());points=np.array(plan['collision_points'][0]);radii=np.array(plan['collision_radii'][0])
def smooth(x):x=np.clip(x,0,1);return x**3*(10-15*x+6*x*x)
for drop in [.01,.02]:
 name='eleventh_flat_drop'+str(int(drop*1000));out=base.copy();records=[]
 for f in range(850,986):
  original=base[f].astype(float);q=original.copy();w=smooth((f-850)/40)*(1-smooth((f-955)/30));q[2]-=drop*w;errors=[]
  for side in [0,1]:
   pos,R,_=model.evaluate(original,side);target=pos.copy();targetR=R
   if side==0:
    yaw=np.arctan2(R[1,0],R[0,0]);flat=Rotation.from_euler('z',yaw).as_matrix();rotation=Rotation.from_matrix(flat@R.T).as_rotvec();targetR=Rotation.from_rotvec(w*rotation).as_matrix()@R;target[2]+=np.min((points@R.T)[:,2]-radii)-np.min((points@targetR.T)[:,2]-radii)
   q,pe,re=solve_leg_pose(model,q,side,target,targetR);errors.append((pe,re))
  out[f]=q;records.append([f,*np.max(errors,axis=0)])
 records=np.array(records);rmse=np.sqrt(np.mean((out[:,7:]-source[:,7:])**2,axis=1));root=np.linalg.norm(out[:,:3]-source[:,:3],axis=1);t=torch.tensor(out);curve=BlendedSplineReference(MotionInterpolator(t),t,stride=6,start_frame=130,ramp_frames=20);dense=np.array([curve.kinematics(f,240)[0].numpy()[0] for f in np.arange(844,993,.2)]);margin=float(np.minimum(dense[:,7:]-limits[:,0],limits[:,1]-dense[:,7:]).min());clearances=[]
 for q in dense:
  position,orientation,_=model.evaluate(q,0);clearances.append(position[2]+np.min((points@orientation.T)[:,2]-radii))
 landing_error=max(np.linalg.norm(model.evaluate(out[f],0)[0]-model.evaluate(source[f],0)[0]) for f in range(884,956))
 report=dict(minimum_dense_native_clearance=float(min(clearances)),maximum_landing_source_foot_error=float(landing_error),root_drop=drop,peak_position_residual=float(records[:,1].max()),peak_orientation_residual=float(records[:,2].max()),peak_source_joint_reference_rmse=float(rmse.max()),peak_root_position_change=float(root.max()),minimum_dense_joint_limit_margin=margin,physical_tracking_certified=False)
 report['preflight_passed']=bool(records[:,1].max()<.002 and records[:,2].max()<.01 and rmse.max()<.19 and root.max()<.19 and margin>=0 and min(clearances)>=-.002 and landing_error<.04);np.savez_compressed(p/(name+'.npz'),control_q=out,fps=200.,baseline_sha256=hashlib.sha256(baseline['control_q'].tobytes()).hexdigest(),source_sha256=hashlib.sha256(source.tobytes()).hexdigest());(p/(name+'.json')).write_text(json.dumps(report,indent=2)+'\n');print(name,report,flush=True)
