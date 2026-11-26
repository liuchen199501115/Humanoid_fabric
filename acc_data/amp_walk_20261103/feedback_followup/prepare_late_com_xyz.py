"""Smooth only the later COM height, retaining both foot poses and root XY."""
from pathlib import Path
import numpy as np,json,hashlib,torch
from scipy.signal import savgol_filter
from fabrics_sim.reference.interpolation import MotionInterpolator
from numpy_leg_kinematics import LegKinematics
from numpy_g1_com import CenterOfMassModel
from bounded_leg_ik import solve_leg_pose
from blended_spline import BlendedSplineReference
p=Path(__file__).parent;baseline=np.load(p/'fixed_smooth_early.control_reference.npz');source=baseline['source_q'];base=np.load(p/'turn_recovery_y40.npz')['control_q'];d=np.load(p/'wrench_fixed_ten.npz');limits=d['joint_position_limits'][0];urdf='src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf';model=LegKinematics(urdf,d['final_joint_order']);com=CenterOfMassModel(urdf,d['final_joint_order']);original_com=np.array([com.evaluate(q) for q in base]);filtered=savgol_filter(original_com,81,3,axis=0)
def smooth(x):x=np.clip(x,0,1);return x**3*(10-15*x+6*x*x)
for drop in [.005,.01]:
 name='late_com_xyz81_drop'+str(int(drop*1000));out=base.copy();records=[]
 for f in range(820,1001):
  original=base[f].astype(float);q=original.copy();w=smooth((f-820)/30)*(1-smooth((f-940)/60));target=original_com[f]+w*(filtered[f]-original_com[f]);target[2]-=drop*w;targets=[model.evaluate(original,side)[:2] for side in range(2)]
  for iteration in range(8):
   q[:3]+=1.25*(target-com.evaluate(q));errors=[]
   for side,(pos,R) in enumerate(targets):
    q,pe,re=solve_leg_pose(model,q,side,pos,R);errors.append((pe,re))
   ce=np.linalg.norm(com.evaluate(q)-target)
   if ce<1e-6:break
  out[f]=q;records.append([f,ce,*np.max(errors,axis=0)])
 records=np.array(records);rmse=np.sqrt(np.mean((out[:,7:]-source[:,7:])**2,axis=1));root=np.linalg.norm(out[:,:3]-source[:,:3],axis=1);t=torch.tensor(out);curve=BlendedSplineReference(MotionInterpolator(t),t,stride=6,start_frame=130,ramp_frames=20);dense=np.array([curve.kinematics(f,240)[0].numpy()[0] for f in np.arange(814,1007,.2)]);margin=float(np.minimum(dense[:,7:]-limits[:,0],limits[:,1]-dense[:,7:]).min());report=dict(root_drop=drop,peak_com_projection_error=float(records[:,1].max()),peak_position_residual=float(records[:,2].max()),peak_orientation_residual=float(records[:,3].max()),peak_source_joint_reference_rmse=float(rmse.max()),peak_root_position_change=float(root.max()),minimum_dense_joint_limit_margin=margin,physical_tracking_certified=False)
 report['preflight_passed']=bool(records[:,1].max()<.002 and records[:,2].max()<.002 and records[:,3].max()<.01 and rmse.max()<.19 and root.max()<.19 and margin>=0);np.savez_compressed(p/(name+'.npz'),control_q=out,fps=200.,baseline_sha256=hashlib.sha256(baseline['control_q'].tobytes()).hexdigest(),source_sha256=hashlib.sha256(source.tobytes()).hexdigest());(p/(name+'.json')).write_text(json.dumps(report,indent=2)+'\n');print(name,report,flush=True)
