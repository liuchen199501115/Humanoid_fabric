"""Smooth the root path while preserving source foot positions and native clearance."""
from pathlib import Path
import json,hashlib
import numpy as np,torch
from scipy.signal import savgol_filter
from fabrics_sim.reference.interpolation import MotionInterpolator
from numpy_leg_kinematics import LegKinematics
from bounded_leg_ik import solve_leg_pose
from blended_spline import BlendedSplineReference
p=Path(__file__).parent;baseline=np.load(p/'fixed_smooth_early.control_reference.npz');source=baseline['source_q'];base=np.load(p/'seventh_plant_roll_y25.npz')['control_q'];d=np.load(p/'wrench_fixed_ten.npz');limits=d['joint_position_limits'][0];model=LegKinematics('src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf',d['final_joint_order']);plan=json.load(open(p/'wrench_fixed_ten.steps_plan.json'));geometry=[(np.array(plan['collision_points'][s]),np.array(plan['collision_radii'][s])) for s in range(2)]
def smooth(x):x=np.clip(x,0,1);return x**3*(10-15*x+6*x*x)
for window,axes in [(61,2),(41,3),(61,3)]:
 name=f'root_plant_smooth{window}_{axes}';out=base.copy();target_root=savgol_filter(base[:,:3],window,3,axis=0);records=[]
 for f in range(470,len(base)):
  original=base[f].astype(float);q=original.copy();w=smooth((f-470)/40);q[:axes]+=w*(target_root[f,:axes]-original[:axes]);q[2]-=.3*np.linalg.norm(q[:2]-original[:2])+(w*.005 if axes==3 else 0.)
  errors=[]
  for side in range(2):
   pos,R,_=model.evaluate(original,side);target=pos.copy();pts,radii=geometry[side];clear=pos[2]+np.min((pts@R.T)[:,2]-radii)
   for correction in range(4):
    q,pe,re=solve_leg_pose(model,q,side,target,R);actual,actualR,_=model.evaluate(q,side);actual_clear=actual[2]+np.min((pts@actualR.T)[:,2]-radii)
    if abs(actual_clear-clear)<1e-6:break
    target[2]+=clear-actual_clear
   errors.append([np.linalg.norm(actual[:2]-pos[:2]),abs(actual_clear-clear),re,np.linalg.norm(actual-pos)])
  q[7:]=np.clip(q[7:],limits[:,0]+1e-5,limits[:,1]-1e-5);out[f]=q;records.append([f,*np.max(errors,axis=0)])
 records=np.array(records);rmse=np.sqrt(np.mean((out[:,7:]-source[:,7:])**2,axis=1));root_delta=np.linalg.norm(out[:,:3]-source[:,:3],axis=1);tensor=torch.tensor(out);sampler=BlendedSplineReference(MotionInterpolator(tensor),tensor,stride=6,start_frame=130,ramp_frames=20);dense=np.array([sampler.kinematics(f,200)[0].numpy()[0] for f in np.arange(470,1999.01,.5)]);margin=float(np.minimum(dense[:,7:]-limits[:,0],limits[:,1]-dense[:,7:]).min())
 report=dict(window_frames=window,root_axes_smoothed=axes,peak_foot_horizontal_error=float(records[:,1].max()),peak_native_clearance_error=float(records[:,2].max()),peak_foot_orientation_adjustment=float(records[:,3].max()),peak_foot_link_displacement=float(records[:,4].max()),peak_source_joint_reference_rmse=float(rmse.max()),peak_root_position_delta=float(root_delta.max()),minimum_dense_joint_limit_margin=margin,initial_qva_preserved=True,physical_tracking_certified=False)
 report['preflight_passed']=bool(records[:,1].max()<.002 and records[:,2].max()<.002 and records[:,3].max()<.1 and records[:,4].max()<.025 and rmse.max()<.19 and root_delta.max()<.19 and margin>=0);report['worst_rows']=records[np.argsort(records[:,1])[-5:]].tolist()
 np.savez_compressed(p/(name+'.npz'),control_q=out,fps=200.,baseline_sha256=hashlib.sha256(baseline['control_q'].tobytes()).hexdigest(),source_sha256=hashlib.sha256(source.tobytes()).hexdigest());(p/(name+'.json')).write_text(json.dumps(report,indent=2)+'\n');print(name,report,flush=True)
