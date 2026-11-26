"""Keep the late swing above the native sole before the foot reaches its landing position."""
from pathlib import Path
import json,hashlib
import numpy as np,torch
from fabrics_sim.reference.interpolation import MotionInterpolator
from numpy_leg_kinematics import LegKinematics
from blended_spline import BlendedSplineReference
from bounded_leg_ik import solve_leg_pose
p=Path(__file__).parent;baseline=np.load(p/'fixed_smooth_early.control_reference.npz');source=baseline['source_q'];initial=np.load(p/'early_flat_lower15.npz')['control_q'];model=LegKinematics('src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf',np.load(p/'wrench_fixed_ten.npz')['final_joint_order']);plan=json.load(open(p/'wrench_fixed_ten.steps_plan.json'));points=np.array(plan['collision_points'][0]);radii=np.array(plan['collision_radii'][0])
def smooth(x):
 x=np.clip(x,0,1);return x**3*(10-15*x+6*x*x)
for height in [.015,.025,.035]:
 name='seventh_clearance'+str(int(height*1000));out=initial.copy();errors=[]
 for f in range(507,564):
  q=initial[f].astype(float);pos,R,_=model.evaluate(q,0);w=smooth((f-507)/17)*(1-smooth((f-549)/14));target=pos.copy();target[2]+=height*w
  if w:
   q,pe,re=solve_leg_pose(model,q,0,target,R);errors.append([f,pe,re]);out[f]=q
 errors=np.array(errors);tensor=torch.tensor(out);curve=BlendedSplineReference(MotionInterpolator(tensor),tensor,stride=6,start_frame=130,ramp_frames=20)
 frames=np.arange(507,563.01,.1);dense=np.array([curve.kinematics(f,200)[0].numpy()[0] for f in frames]);margin=float(np.minimum(dense[:,7:19]-model.lower,model.upper-dense[:,7:19]).min());clear=[]
 for q in dense:
  pos,R,_=model.evaluate(q,0);clear.append(float(pos[2]+np.min((points@R.T)[:,2]-radii)))
 rmse=float(np.sqrt(np.mean((out[:,7:]-source[:,7:])**2,axis=1)).max());report=dict(height=height,peak_position_residual=float(errors[:,1].max()),peak_orientation_residual=float(errors[:,2].max()),peak_source_joint_reference_rmse=rmse,minimum_dense_joint_limit_margin=margin,clearance_frame524_547_min=float(np.min(np.asarray(clear)[(frames>=524)&(frames<=547)])),physical_tracking_certified=False)
 report['preflight_passed']=bool(errors[:,1].max()<.002 and errors[:,2].max()<.01 and rmse<.19 and margin>=0)
 np.savez_compressed(p/(name+'.npz'),control_q=out,fps=200.,baseline_sha256=hashlib.sha256(baseline['control_q'].tobytes()).hexdigest(),source_sha256=hashlib.sha256(source.tobytes()).hexdigest());(p/(name+'.json')).write_text(json.dumps(report,indent=2)+'\n');print(name,report,flush=True)
