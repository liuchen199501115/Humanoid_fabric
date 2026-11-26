"""Consistent whole-body landing-tail clearance for source steps seven onward."""
from pathlib import Path
import json,hashlib
import numpy as np,torch
from fabrics_sim.reference.interpolation import MotionInterpolator
from numpy_leg_kinematics import LegKinematics
from blended_spline import BlendedSplineReference
from bounded_leg_ik import solve_leg_pose
p=Path(__file__).parent;baseline=np.load(p/'fixed_smooth_early.control_reference.npz');source=baseline['source_q'];initial=np.load(p/'early_flat_lower15.npz')['control_q'];d=np.load(p/'wrench_fixed_ten.npz');limits=d['joint_position_limits'][0];model=LegKinematics('src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf',d['final_joint_order']);plan=json.load(open(p/'wrench_fixed_ten.steps_plan.json'))
def smooth(x):x=np.clip(x,0,1);return x**3*(10-15*x+6*x*x)
for height in [.025,.035]:
 name='late_clearance'+str(int(height*1000));out=initial.copy();errors=[]
 for ei,event in enumerate(plan['events'][6:],6):
  side=1-event['support'];end=event['end_frame'];start=max(event['takeoff_frame'],end-42);rise=min(17.,.5*(end-start))
  for f in range(start,min(end+15,len(initial))):
   q=out[f].astype(float);pos,R,_=model.evaluate(q,side);w=smooth((f-start)/rise)*(1-smooth((f-end)/14));target=pos.copy();target[2]+=height*w
   if w:
    q,pe,re=solve_leg_pose(model,q,side,target,R);errors.append([ei+1,f,pe,re]);out[f]=q
 out[507:,7:]=np.clip(out[507:,7:],limits[:,0]+1e-5,limits[:,1]-1e-5)
 errors=np.array(errors);tensor=torch.tensor(out);curve=BlendedSplineReference(MotionInterpolator(tensor),tensor,stride=6,start_frame=130,ramp_frames=20);frames=np.arange(507,1999.01,.5);dense=np.array([curve.kinematics(f,200)[0].numpy()[0] for f in frames]);margin=float(np.minimum(dense[:,7:]-limits[:,0],limits[:,1]-dense[:,7:]).min());rmse=float(np.sqrt(np.mean((out[:,7:]-source[:,7:])**2,axis=1)).max())
 report=dict(height=height,peak_position_residual=float(errors[:,2].max()),peak_orientation_residual=float(errors[:,3].max()),peak_source_joint_reference_rmse=rmse,minimum_dense_joint_limit_margin=margin,physical_tracking_certified=False,worst_rows=errors[np.argsort(errors[:,2])[-5:]].tolist())
 report['preflight_passed']=bool(errors[:,2].max()<.002 and errors[:,3].max()<.01 and rmse<.19 and margin>=0)
 np.savez_compressed(p/(name+'.npz'),control_q=out,fps=200.,baseline_sha256=hashlib.sha256(baseline['control_q'].tobytes()).hexdigest(),source_sha256=hashlib.sha256(source.tobytes()).hexdigest());(p/(name+'.json')).write_text(json.dumps(report,indent=2)+'\n');print(name,report,flush=True)
