"""Repair interpolation-induced native sole penetration in a bounded reference."""
from pathlib import Path
import numpy as np,json,torch
from scipy.ndimage import maximum_filter1d,gaussian_filter1d
from fabrics_sim.reference.interpolation import MotionInterpolator
from numpy_leg_kinematics import LegKinematics
from bounded_leg_ik import solve_leg_pose
from blended_spline import BlendedSplineReference
p=Path(__file__).parent;import sys;parent=sys.argv[1];data=np.load(p/(parent+'.npz'));base=data['control_q'];out=base.copy();d=np.load(p/'wrench_fixed_ten.npz');limits=d['joint_position_limits'][0];m=LegKinematics('src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf',d['final_joint_order']);m.lower+=.01;m.upper-=.01;plan=json.load(open(p/'wrench_fixed_ten.steps_plan.json'));geometry=[(np.array(plan['collision_points'][j]),np.array(plan['collision_radii'][j])) for j in [0,1]];frames=np.arange(754,847,.25);errors=[];max_fix=0.;repairs=[]
def sample(out):
 t=torch.tensor(out);curve=BlendedSplineReference(MotionInterpolator(t),t,stride=6,start_frame=130,ramp_frames=20);values=[curve.kinematics(float(f),240) for f in frames];q=np.array([x[0].numpy()[0] for x in values]);clear=[]
 for pose in q:
  row=[]
  for j in [0,1]:
   pos,R,_=m.evaluate(pose,j);pts,r=geometry[j];row.append(pos[2]+min((pts@R.T)[:,2]-r))
  clear.append(row)
 return values,q,np.array(clear)
for iteration in range(3):
 values,dense,clear=sample(out);repairs.append(dict(iteration=iteration,minimum_clearance=float(clear.min())))
 if clear.min()>-.0015:break
 corrections=np.zeros((len(out),2))
 for j in [0,1]:
  np.maximum.at(corrections[:,j],np.rint(frames).astype(int),np.where(clear[:,j]<-.0008,.0002-clear[:,j],0.))
 corrections=1.3*gaussian_filter1d(maximum_filter1d(corrections,size=11,axis=0,mode='nearest'),2,axis=0)
 for f in np.flatnonzero(corrections.max(1)>1e-7):
  q=out[f].astype(float)
  for j in [0,1]:
   if corrections[f,j]<=1e-7:continue
   pos,R,_=m.evaluate(q,j);pos[2]+=corrections[f,j];q,pe,re=solve_leg_pose(m,q,j,pos,R,orientation_weight=.05);errors.append([pe,re])
  out[f]=q
 max_fix+=float(corrections.max())
values,dense,clear=sample(out);margin=float(np.minimum(dense[:,7:]-limits[:,0],limits[:,1]-dense[:,7:]).min());source=np.load(p/'fixed_smooth_early.control_reference.npz')['source_q'];rmse=float(np.sqrt(np.mean((out[:,7:]-source[:,7:])**2,axis=1)).max());vel=float(max(abs(x[1].numpy()[0,6:]).max() for x in values));report=json.load(open(p/(parent+'.json')));report.update(parent=parent,maximum_additional_clearance_target=max_fix,clearance_repair_iterations=repairs,native_floor_repair_required=bool(clear.min()<-.0015),minimum_dense_native_clearance=float(clear.min()),minimum_dense_joint_limit_margin=margin,peak_source_joint_reference_rmse=rmse,maximum_dense_joint_velocity=vel,floor_ik_peak_residual=np.max(errors,axis=0).tolist() if errors else [0.,0.],audited_source_frame_interval=[float(frames[0]),float(frames[-1])]);report['preflight_passed']=bool(report.get('peak_foot_xy_error',report['peak_position_residual'])<.002 and report.get('peak_foot_orientation_adjustment',report['peak_orientation_residual'])<.15 and rmse<.19 and margin>.005 and clear.min()>-.0015 and vel<20 and max_fix<.012 and report['floor_ik_peak_residual'][0]<.002 and report['floor_ik_peak_residual'][1]<.01);name=parent+'_floor';np.savez_compressed(p/(name+'.npz'),control_q=out,fps=200.,baseline_sha256=data['baseline_sha256'],source_sha256=data['source_sha256']);(p/(name+'.json')).write_text(json.dumps(report,indent=2)+'\n');print(name,report)
