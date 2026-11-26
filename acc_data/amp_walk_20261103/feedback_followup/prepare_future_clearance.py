"""Bounded native-clearance corrections after the already verified ninth step."""
from pathlib import Path
import hashlib,json
import numpy as np,torch
from scipy.ndimage import gaussian_filter1d
from fabrics_sim.reference.interpolation import MotionInterpolator
from numpy_leg_kinematics import LegKinematics
from bounded_leg_ik import solve_leg_pose
from blended_spline import BlendedSplineReference
p=Path(__file__).parent;baseline=np.load(p/'fixed_smooth_early.control_reference.npz');source=baseline['source_q'];d=np.load(p/'wrench_fixed_ten.npz');limits=d['joint_position_limits'][0];model=LegKinematics('src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf',d['final_joint_order']);plan=json.load(open(p/'wrench_fixed_ten.steps_plan.json'))
def smooth(x):x=np.clip(x,0,1);return x**3*(10-15*x+6*x*x)
for parent,label in [('sixth_plant_roll_y30','plant6'),('eighth_plant_roll_y-30','plant8')]:
 base=np.load(p/(parent+'.npz'))['control_q'];out=base.copy();errors=[];corrections=[]
 for ei,event in enumerate(plan['events'][9:],9):
  side=1-event['support'];start=event['takeoff_frame'];end=event['end_frame'];frames=np.arange(start,end+1);edge=min(12.,(end-start)*.3);pts=np.array(plan['collision_points'][side]);radii=np.array(plan['collision_radii'][side]);height=[]
  for f in frames:
   pos,R,_=model.evaluate(base[f],side);clear=pos[2]+np.min((pts@R.T)[:,2]-radii);height.append(max(0.,.04-clear))
  height=gaussian_filter1d(height,2.,mode='nearest')*smooth((frames-start)/edge)*smooth((end-frames)/edge)
  for f,h in zip(frames,height):
   if h<=1e-9:continue
   q=out[f].astype(float);pos,R,_=model.evaluate(q,side);target=pos.copy();target[2]+=h;q,pe,re=solve_leg_pose(model,q,side,target,R);out[f]=q;errors.append([ei+1,int(f),pe,re]);corrections.append([int(f),side,float(h)])
 out[780:,7:]=np.clip(out[780:,7:],limits[:,0]+1e-5,limits[:,1]-1e-5);errors=np.array(errors);rmse=float(np.sqrt(np.mean((out[:,7:]-source[:,7:])**2,axis=1)).max());tensor=torch.tensor(out);curve=BlendedSplineReference(MotionInterpolator(tensor),tensor,stride=6,start_frame=130,ramp_frames=20);dense=np.array([curve.kinematics(f,200)[0].numpy()[0] for f in np.arange(780,1999.01,.5)]);margin=float(np.minimum(dense[:,7:]-limits[:,0],limits[:,1]-dense[:,7:]).min());report=dict(parent=parent,peak_position_residual=float(errors[:,2].max()),peak_orientation_residual=float(errors[:,3].max()),peak_source_joint_reference_rmse=rmse,minimum_dense_joint_limit_margin=margin,peak_height_correction=max(row[2] for row in corrections),physical_tracking_certified=False)
 report['preflight_passed']=bool(errors[:,2].max()<.002 and errors[:,3].max()<.01 and rmse<.19 and margin>=0);name=label+'_future_clearance40';np.savez_compressed(p/(name+'.npz'),control_q=out,fps=200.,baseline_sha256=hashlib.sha256(baseline['control_q'].tobytes()).hexdigest(),source_sha256=hashlib.sha256(source.tobytes()).hexdigest());(p/(name+'.json')).write_text(json.dumps(report,indent=2)+'\n');print(name,report,flush=True)
