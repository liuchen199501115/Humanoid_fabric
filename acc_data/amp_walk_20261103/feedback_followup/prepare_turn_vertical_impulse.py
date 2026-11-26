"""Redistribute vertical support acceleration across the short tenth transfer.
A compact C2 pelvis-height correction preserves both world foot poses.
"""
from pathlib import Path
import json,numpy as np,torch
from scipy.interpolate import CubicSpline
from numpy_leg_kinematics import LegKinematics
from numpy_g1_com import CenterOfMassModel
from bounded_leg_ik import solve_leg_pose
from blended_spline import BlendedSplineReference
from fabrics_sim.reference.interpolation import MotionInterpolator
p=Path(__file__).parent;data=np.load(p/'turn_recovery_y40.npz');base=data['control_q'];saved=np.load(p/'wrench_fixed_ten.npz');model=LegKinematics('src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf',saved['final_joint_order']);com=CenterOfMassModel('src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf',saved['final_joint_order']);source=np.load(p/'fixed_smooth_early.control_reference.npz')['source_q'];limits=saved['joint_position_limits'][0];plan=json.load(open(p/'wrench_fixed_ten.steps_plan.json'));geometry=[(np.array(plan['collision_points'][j]),np.array(plan['collision_radii'][j])) for j in [0,1]]
for drop in [.025,.035,.045]:
 out=base.copy();errors=[]
 for f in range(761,840):
  u=(f-760)/80.;bump=64*u**3*(1-u)**3;q=base[f].astype(float);targets=[model.evaluate(q,j)[:2] for j in [0,1]];q[2]-=drop*bump
  for j,(pos,R) in enumerate(targets):q,pe,re=solve_leg_pose(model,q,j,pos,R);errors.append([pe,re])
  out[f]=q
 t=torch.tensor(out);curve=BlendedSplineReference(MotionInterpolator(t),t,stride=6,start_frame=130,ramp_frames=20);frames=np.arange(754,847,.25);values=[curve.kinematics(float(f),240.) for f in frames];dense=np.array([x[0].numpy()[0] for x in values]);momentum=np.array([com.momentum(x[0].numpy()[0],x[1].numpy()[0],return_state=True) for x in values]);times=frames/240.;a=CubicSpline(times,momentum[:,1])(times,1);forces=com.mass*(a+[0,0,9.81]);clear=[]
 for q in dense:
  for j in [0,1]:
   pos,R,_=model.evaluate(q,j);pts,r=geometry[j];clear.append(pos[2]+min((pts@R.T)[:,2]-r))
 margin=float(np.minimum(dense[:,7:]-limits[:,0],limits[:,1]-dense[:,7:]).min());rmse=float(np.sqrt(np.mean((out[:,7:]-source[:,7:])**2,axis=1)).max());err=np.max(errors,axis=0);report=dict(parent='turn_recovery_y40',root_drop=drop,source_interval=[760,840],peak_position_residual=float(err[0]),peak_orientation_residual=float(err[1]),minimum_dense_native_clearance=float(min(clear)),minimum_dense_joint_limit_margin=margin,peak_source_joint_reference_rmse=rmse,minimum_required_normal_force=float(forces[:,2].min()),maximum_required_normal_force=float(forces[:,2].max()),negative_force_samples=int((forces[:,2]<0).sum()),physical_tracking_certified=False)
 report['preflight_passed']=bool(err[0]<.002 and err[1]<.01 and min(clear)>-.0015 and margin>.005 and rmse<.19)
 name='turn_vertical_impulse'+str(round(drop*1000));np.savez_compressed(p/(name+'.npz'),control_q=out,fps=200.,baseline_sha256=data['baseline_sha256'],source_sha256=data['source_sha256']);np.savez_compressed(p/(name+'.forces.npz'),frames=frames,forces=forces,com=momentum[:,0],velocity=momentum[:,1]);(p/(name+'.json')).write_text(json.dumps(report,indent=2)+'\n');print(name,report,flush=True)
