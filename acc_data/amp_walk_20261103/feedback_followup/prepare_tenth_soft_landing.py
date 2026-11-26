"""Smooth the native sole's final descent, keeping horizontal and angular targets."""
from pathlib import Path
import json,hashlib
import numpy as np,torch
from fabrics_sim.reference.interpolation import MotionInterpolator
from numpy_leg_kinematics import LegKinematics
from bounded_leg_ik import solve_leg_pose
from blended_spline import BlendedSplineReference
p=Path(__file__).parent;baseline=np.load(p/'fixed_smooth_early.control_reference.npz');source=baseline['source_q'];base=np.load(p/'tenth_landing_forward15_right20.npz')['control_q'];d=np.load(p/'wrench_fixed_ten.npz');limits=d['joint_position_limits'][0];model=LegKinematics('src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf',d['final_joint_order']);plan=json.load(open(p/'wrench_fixed_ten.steps_plan.json'));pts=np.array(plan['collision_points'][1]);radii=np.array(plan['collision_radii'][1])
def smooth(x):x=np.clip(x,0,1);return x**3*(10-15*x+6*x*x)
for end in [802,806]:
 name='tenth_soft_landing'+str(end);out=base.copy();records=[];pos,R,_=model.evaluate(base[790],1);height=pos[2]+np.min((pts@R.T)[:,2]-radii)
 for f in range(790,811):
  q=base[f].astype(float);pos,R,_=model.evaluate(q,1);clear=pos[2]+np.min((pts@R.T)[:,2]-radii);target=pos.copy();target[2]+=height*(1-smooth((f-790)/(end-790)))-clear;q,pe,re=solve_leg_pose(model,q,1,target,R);out[f]=q;records.append([f,pe,re])
 out[790:811,7:]=np.clip(out[790:811,7:],limits[:,0]+1e-5,limits[:,1]-1e-5);records=np.array(records);rmse=np.sqrt(np.mean((out[:,7:]-source[:,7:])**2,axis=1));tensor=torch.tensor(out);curve=BlendedSplineReference(MotionInterpolator(tensor),tensor,stride=6,start_frame=130,ramp_frames=20);dense=np.array([curve.kinematics(f,200)[0].numpy()[0] for f in np.arange(784,818.01,.1)]);margin=float(np.minimum(dense[:,7:]-limits[:,0],limits[:,1]-dense[:,7:]).min());clearances=[]
 for q in dense:
  pos,R,_=model.evaluate(q,1);clearances.append(pos[2]+np.min((pts@R.T)[:,2]-radii))
 report=dict(descent_end_frame=end,peak_position_residual=float(records[:,1].max()),peak_orientation_residual=float(records[:,2].max()),peak_source_joint_reference_rmse=float(rmse.max()),minimum_dense_joint_limit_margin=margin,minimum_dense_native_clearance=float(min(clearances)),physical_tracking_certified=False)
 report['preflight_passed']=bool(records[:,1].max()<.002 and records[:,2].max()<.01 and rmse.max()<.19 and margin>=0 and min(clearances)>=-.002);np.savez_compressed(p/(name+'.npz'),control_q=out,fps=200.,baseline_sha256=hashlib.sha256(baseline['control_q'].tobytes()).hexdigest(),source_sha256=hashlib.sha256(source.tobytes()).hexdigest());(p/(name+'.json')).write_text(json.dumps(report,indent=2)+'\n');print(name,report,flush=True)
