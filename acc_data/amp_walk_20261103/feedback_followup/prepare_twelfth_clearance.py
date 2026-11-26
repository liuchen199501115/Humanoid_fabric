"""Add a bounded native swing clearance margin on the twelfth source step."""
from pathlib import Path
import json,hashlib
import numpy as np,torch
from fabrics_sim.reference.interpolation import MotionInterpolator
from numpy_leg_kinematics import LegKinematics
from bounded_leg_ik import solve_leg_pose
from blended_spline import BlendedSplineReference
p=Path(__file__).parent;baseline=np.load(p/'fixed_smooth_early.control_reference.npz');source=baseline['source_q'];base=np.load(p/'tenth_landing_forward15_right20.npz')['control_q'];d=np.load(p/'wrench_fixed_ten.npz');limits=d['joint_position_limits'][0];model=LegKinematics('src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf',d['final_joint_order'])
def smooth(x):x=np.clip(x,0,1);return x**3*(10-15*x+6*x*x)
for extra in [.01,.02]:
 name='twelfth_clearance'+str(int(extra*1000));out=base.copy();records=[]
 for f in range(908,953):
  q=base[f].astype(float);pos,R,_=model.evaluate(q,1);target=pos.copy();target[2]+=extra*smooth((f-908)/12)*(1-smooth((f-938)/14));q,pe,re=solve_leg_pose(model,q,1,target,R);out[f]=q;records.append([f,pe,re])
 records=np.array(records);rmse=np.sqrt(np.mean((out[:,7:]-source[:,7:])**2,axis=1));tensor=torch.tensor(out);curve=BlendedSplineReference(MotionInterpolator(tensor),tensor,stride=6,start_frame=130,ramp_frames=20);dense=np.array([curve.kinematics(f,240)[0].numpy()[0] for f in np.arange(902,960,.2)]);margin=float(np.minimum(dense[:,7:]-limits[:,0],limits[:,1]-dense[:,7:]).min());report=dict(extra_clearance_m=extra,peak_position_residual=float(records[:,1].max()),peak_orientation_residual=float(records[:,2].max()),peak_source_joint_reference_rmse=float(rmse.max()),minimum_dense_joint_limit_margin=margin,physical_tracking_certified=False)
 report['preflight_passed']=bool(records[:,1].max()<.002 and records[:,2].max()<.01 and rmse.max()<.19 and margin>=0);np.savez_compressed(p/(name+'.npz'),control_q=out,fps=200.,baseline_sha256=hashlib.sha256(baseline['control_q'].tobytes()).hexdigest(),source_sha256=hashlib.sha256(source.tobytes()).hexdigest());(p/(name+'.json')).write_text(json.dumps(report,indent=2)+'\n');print(name,report,flush=True)
