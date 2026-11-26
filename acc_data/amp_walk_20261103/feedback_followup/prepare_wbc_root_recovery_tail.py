"""Remove the turn root correction gradually while preserving both foot poses."""
from pathlib import Path
import json,numpy as np,torch
from fabrics_sim.reference.interpolation import MotionInterpolator
from numpy_leg_kinematics import LegKinematics
from bounded_leg_ik import solve_leg_pose
from blended_spline import BlendedSplineReference
p=Path(__file__).parent;parent=np.load(p/'wbc_convex_catch15_25_clear35.npz');base=parent['control_q'];source=np.load(p/'fixed_smooth_early.control_reference.npz')['source_q'];d=np.load(p/'wrench_fixed_ten.npz');limits=d['joint_position_limits'][0];speed=d['history_joint_velocity_limits'][-1].reshape(-1);plan=json.loads((p/'wrench_fixed_ten.steps_plan.json').read_text());model=LegKinematics('src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf',d['final_joint_order'])
def smooth(x):
 x=np.clip(x,0,1);return x**3*(10-15*x+6*x*x)
for end in [1120,1240]:
 out=base.copy();errors=[]
 for f in range(941,end):
  q=base[f].astype(float);feet=[model.evaluate(q,side)[:2] for side in [0,1]]
  q[1]+=.04*(smooth((f-940)/60)-smooth((f-940)/(end-940)))
  for side,(pos,R) in enumerate(feet):
   q,pe,re=solve_leg_pose(model,q,side,pos,R);errors.append([pe,re])
  out[f]=q
 tensor=torch.tensor(out);curve=BlendedSplineReference(MotionInterpolator(tensor),tensor,stride=6,start_frame=130,ramp_frames=20);values=[curve.kinematics(float(f),240.) for f in np.arange(934,end+7,.2)];dense=np.array([v[0].numpy()[0] for v in values]);vel=np.array([v[1].numpy()[0,6:] for v in values]);margin=np.minimum(dense[:,7:]-limits[:,0],limits[:,1]-dense[:,7:]).min();clear=[]
 for q in dense:
  for side in [0,1]:
   pos,R,_=model.evaluate(q,side);clear.append(pos[2]+np.min((np.array(plan['collision_points'][side])@R.T)[:,2]-np.array(plan['collision_radii'][side])))
 err=np.max(errors,axis=0);rmse=np.sqrt(np.mean((out[:,7:]-source[:,7:])**2,axis=1)).max();fraction=(abs(vel)/speed).max();report=dict(root_recovery_end_frame=end,peak_ik_position_error=float(err[0]),peak_ik_orientation_error=float(err[1]),minimum_joint_margin=float(margin),minimum_native_clearance=float(min(clear)),peak_source_joint_reference_rmse=float(rmse),maximum_joint_velocity_fraction=float(fraction),physical_tracking_certified=False)
 report['preflight_passed']=bool(err[0]<.002 and err[1]<.01 and margin>=0 and min(clear)>=-.002 and rmse<.19 and fraction<=1)
 name=f'wbc_root_tail{end}';np.savez_compressed(p/(name+'.npz'),control_q=out,fps=200.,baseline_sha256=parent['baseline_sha256'],source_sha256=parent['source_sha256']);(p/(name+'.json')).write_text(json.dumps(report,indent=2)+'\n');print(name,report,flush=True)
