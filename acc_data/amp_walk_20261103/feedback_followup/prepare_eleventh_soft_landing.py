"""Reduce late swing vertical impact speed while retaining XY and orientation."""
from pathlib import Path
import json,hashlib
import numpy as np,torch
from fabrics_sim.reference.interpolation import MotionInterpolator
from numpy_leg_kinematics import LegKinematics
from bounded_leg_ik import solve_leg_pose
from blended_spline import BlendedSplineReference
p=Path(__file__).parent;baseline=np.load(p/'fixed_smooth_early.control_reference.npz');source=baseline['source_q'];base=np.load(p/'tenth_landing_forward15_right20.npz')['control_q'];d=np.load(p/'wrench_fixed_ten.npz');limits=d['joint_position_limits'][0];model=LegKinematics('src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf',d['final_joint_order']);plan=json.load(open(p/'wrench_fixed_ten.steps_plan.json'));pts=np.array(plan['collision_points'][0]);radii=np.array(plan['collision_radii'][0])
def smooth(x):x=np.clip(x,0,1);return x**3*(10-15*x+6*x*x)
for drop in [0.,.01]:
 name='eleventh_soft_landing884_drop'+str(int(drop*1000));out=base.copy();records=[];pos,R,_=model.evaluate(base[850],0);height=pos[2]+np.min((pts@R.T)[:,2]-radii)
 for f in range(850,941):
  q=base[f].astype(float);q[2]-=drop*smooth((f-850)/20)*(1-smooth((f-900)/40));errors=[]
  for side in [0,1]:
   pos,R,_=model.evaluate(base[f],side);target=pos.copy()
   if side==0 and f<=895:
    clear=pos[2]+np.min((pts@R.T)[:,2]-radii);target[2]+=height*(1-smooth((f-850)/34))-clear
   q,pe,re=solve_leg_pose(model,q,side,target,R);errors.append((pe,re))
  out[f]=q;records.append([f,*np.max(errors,axis=0)])
 records=np.array(records);rmse=np.sqrt(np.mean((out[:,7:]-source[:,7:])**2,axis=1));tensor=torch.tensor(out);curve=BlendedSplineReference(MotionInterpolator(tensor),tensor,stride=6,start_frame=130,ramp_frames=20);dense=np.array([curve.kinematics(f,240)[0].numpy()[0] for f in np.arange(844,947,.2)]);margin=float(np.minimum(dense[:,7:]-limits[:,0],limits[:,1]-dense[:,7:]).min());clearances=[]
 for q in dense[:int((896-844)/.2)]:
  pos,R,_=model.evaluate(q,0);clearances.append(pos[2]+np.min((pts@R.T)[:,2]-radii))
 report=dict(descent_end_frame=884,root_drop=drop,peak_position_residual=float(records[:,1].max()),peak_orientation_residual=float(records[:,2].max()),peak_source_joint_reference_rmse=float(rmse.max()),minimum_dense_joint_limit_margin=margin,minimum_dense_native_clearance=float(min(clearances)),physical_tracking_certified=False)
 report['preflight_passed']=bool(records[:,1].max()<.002 and records[:,2].max()<.01 and rmse.max()<.19 and margin>=0 and min(clearances)>=-.002);np.savez_compressed(p/(name+'.npz'),control_q=out,fps=200.,baseline_sha256=hashlib.sha256(baseline['control_q'].tobytes()).hexdigest(),source_sha256=hashlib.sha256(source.tobytes()).hexdigest());(p/(name+'.json')).write_text(json.dumps(report,indent=2)+'\n');print(name,report,flush=True)
