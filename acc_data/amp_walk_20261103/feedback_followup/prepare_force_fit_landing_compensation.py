"""Correct measured tenth-landing bias while preserving the fitted COM path."""
from pathlib import Path
import json,numpy as np,torch
from numpy_leg_kinematics import LegKinematics
from numpy_g1_com import CenterOfMassModel
from bounded_leg_ik import solve_leg_pose
from blended_spline import BlendedSplineReference
from fabrics_sim.reference.interpolation import MotionInterpolator
p=Path(__file__).parent;data=np.load(p/'turn_force_feasible_com_floor.npz');base=data['control_q'];source=np.load(p/'fixed_smooth_early.control_reference.npz')['source_q'];d=np.load(p/'wrench_fixed_ten.npz');urdf='src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf';leg=LegKinematics(urdf,d['final_joint_order']);com=CenterOfMassModel(urdf,d['final_joint_order']);limits=d['joint_position_limits'][0];plan=json.load(open(p/'wrench_fixed_ten.steps_plan.json'));geometry=[(np.array(plan['collision_points'][j]),np.array(plan['collision_radii'][j])) for j in [0,1]]
def smooth(x):x=np.clip(x,0,1);return x**3*(10-15*x+6*x*x)
for delta in [np.array([-.04,-.05,0.]),np.array([-.05,-.06,0.])]:
 out=base.copy();errors=[]
 for f in range(781,884):
  q=base[f].astype(float);target=com.evaluate(q);feet=[leg.evaluate(q,j)[:2] for j in [0,1]];w=smooth((f-780)/22)*(1-smooth((f-840)/44));feet[1]=(feet[1][0]+w*delta,feet[1][1])
  for _ in range(12):
   q[:3]+=1.25*(target-com.evaluate(q));peaks=[]
   for j,(pos,R) in enumerate(feet):q,pe,re=solve_leg_pose(leg,q,j,pos,R);peaks.append([pe,re])
   ce=np.linalg.norm(com.evaluate(q)-target)
   if ce<1e-7:break
  out[f]=q;errors.append([ce,*np.max(peaks,axis=0)])
 t=torch.tensor(out);curve=BlendedSplineReference(MotionInterpolator(t),t,stride=6,start_frame=130,ramp_frames=20);frames=np.arange(774,891,.25);values=[curve.kinematics(float(f),240.) for f in frames];dense=np.array([x[0].numpy()[0] for x in values]);clear=[]
 for q in dense:
  for j in [0,1]:
   pos,R,_=leg.evaluate(q,j);pts,r=geometry[j];clear.append(pos[2]+min((pts@R.T)[:,2]-r))
 margin=float(np.minimum(dense[:,7:]-limits[:,0],limits[:,1]-dense[:,7:]).min());rmse=float(np.sqrt(np.mean((out[:,7:]-source[:,7:])**2,axis=1)).max());err=np.max(errors,axis=0);report=dict(parent='turn_force_feasible_com_floor',additional_landing_offset=delta.tolist(),peak_com_projection_error=float(err[0]),peak_position_residual=float(err[1]),peak_orientation_residual=float(err[2]),minimum_dense_joint_limit_margin=margin,minimum_dense_native_clearance=float(min(clear)),peak_source_joint_reference_rmse=rmse,physical_tracking_certified=False);report['preflight_passed']=bool(err[0]<.002 and err[1]<.002 and err[2]<.01 and margin>.005 and rmse<.19 and min(clear)>-.0015);name=f'force_fit_landing_x{round(delta[0]*1000)}_y{round(delta[1]*1000)}';np.savez_compressed(p/(name+'.npz'),control_q=out,fps=200.,baseline_sha256=data['baseline_sha256'],source_sha256=data['source_sha256']);(p/(name+'.json')).write_text(json.dumps(report,indent=2)+'\n');print(name,report,flush=True)
