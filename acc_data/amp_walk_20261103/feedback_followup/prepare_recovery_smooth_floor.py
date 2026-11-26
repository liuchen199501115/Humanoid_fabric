"""Remove small native-sole penetrations introduced by whole-body smoothing."""
from pathlib import Path
import numpy as np,json,torch
from scipy.ndimage import gaussian_filter1d
from fabrics_sim.reference.interpolation import MotionInterpolator
from numpy_leg_kinematics import LegKinematics
from bounded_leg_ik import solve_leg_pose
from blended_spline import BlendedSplineReference
p=Path(__file__).parent;D=np.load(p/'wrench_fixed_ten.npz');limits=D['joint_position_limits'][0];model=LegKinematics('src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf',D['final_joint_order']);source=np.load(p/'fixed_smooth_early.control_reference.npz')['source_q'];plan=json.loads((p/'wrench_fixed_ten.steps_plan.json').read_text());points=[np.array(x) for x in plan['collision_points']];radii=[np.array(x) for x in plan['collision_radii']]
for window in [51]:
 data=np.load(p/('recovery_wholebody_smooth'+str(window)+'.npz'));base=data['control_q'];out=base.copy();frames=np.arange(830,981);clear=[]
 for f in frames:
  row=[]
  for side in [0,1]:
   pos,R,_=model.evaluate(base[f],side);row.append(pos[2]+np.min((points[side]@R.T)[:,2]-radii[side]))
  clear.append(row)
 correction=gaussian_filter1d(np.maximum(.001-np.array(clear),0),2,axis=0);x=np.clip((frames-830)/10,0,1);blend=x**3*(10-15*x+6*x*x);x=np.clip((frames-970)/10,0,1);blend*=1-x**3*(10-15*x+6*x*x);correction*=blend[:,None];errors=[]
 for i,f in enumerate(frames):
  q=base[f].astype(float)
  for side in [0,1]:
   pos,R,_=model.evaluate(base[f],side);pos[2]+=correction[i,side];q,pe,re=solve_leg_pose(model,q,side,pos,R);errors.append((pe,re))
  out[f]=q
 name='recovery_wholebody_smooth'+str(window)+'_floor';t=torch.tensor(out);curve=BlendedSplineReference(MotionInterpolator(t),t,stride=6,start_frame=130,ramp_frames=20);dense=[];clearances=[]
 for f in np.arange(824,987,.2):
  q=curve.kinematics(f,240)[0].numpy()[0];dense.append(q)
  for side in [0,1]:
   pos,R,_=model.evaluate(q,side);clearances.append(pos[2]+np.min((points[side]@R.T)[:,2]-radii[side]))
 dense=np.array(dense);margin=float(np.minimum(dense[:,7:]-limits[:,0],limits[:,1]-dense[:,7:]).min());joint=float(np.sqrt(np.mean((out[:,7:]-source[:,7:])**2,axis=1)).max());foot=max(np.linalg.norm(model.evaluate(out[f],side)[0]-model.evaluate(source[f],side)[0]) for a,b,side in [(884,909,0),(952,970,1)] for f in range(a,b));errors=np.array(errors);report=dict(peak_position_residual=float(errors[:,0].max()),peak_orientation_residual=float(errors[:,1].max()),peak_source_joint_reference_rmse=joint,minimum_dense_joint_limit_margin=margin,minimum_dense_native_clearance=float(min(clearances)),maximum_landing_source_foot_error=float(foot),maximum_native_floor_correction=float(correction.max()),physical_tracking_certified=False);report['preflight_passed']=bool(errors[:,0].max()<.002 and errors[:,1].max()<.01 and joint<.19 and margin>=0 and min(clearances)>=-.002 and foot<.04);np.savez_compressed(p/(name+'.npz'),control_q=out,fps=200.,baseline_sha256=data['baseline_sha256'],source_sha256=data['source_sha256']);(p/(name+'.json')).write_text(json.dumps(report,indent=2)+'\n');print(name,report,flush=True)
