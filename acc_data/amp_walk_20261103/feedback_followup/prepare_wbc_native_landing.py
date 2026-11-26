"""Geometric reference correction for native touchdown; no online planner."""
from pathlib import Path
import numpy as np,torch,json,hashlib
from numpy_leg_kinematics import LegKinematics
from bounded_leg_ik import solve_leg_pose
from blended_spline import BlendedSplineReference
from fabrics_sim.reference.interpolation import MotionInterpolator
p=Path(__file__).parent;baseline=np.load(p/'fixed_smooth_early.control_reference.npz');source=baseline['source_q'];base=np.load(p/'turn_recovery_y40.npz')['control_q'];d=np.load(p/'wrench_fixed_ten.npz');limits=d['joint_position_limits'][0];model=LegKinematics('src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf',d['final_joint_order']);plan=json.loads((p/'wrench_fixed_ten.steps_plan.json').read_text());points=np.array(plan['collision_points'][0]);radii=np.array(plan['collision_radii'][0])
def smooth(x):
 x=np.clip(x,0,1);return x**3*(10-15*x+6*x*x)
initial,R,_=model.evaluate(base[850],0);height=initial[2]+np.min((points@R.T)[:,2]-radii)
for drop in [.01,.02]:
 out=base.copy();records=[]
 for f in range(830,981):
  q=base[f].astype(float);weight=smooth((f-830)/30)*(1-smooth((f-940)/40));q[2]-=drop*weight
  errors=[]
  for side in [0,1]:
   pos,R,_=model.evaluate(base[f],side);target=pos.copy()
   if side==0 and 850<=f<=925:
    clearance=pos[2]+np.min((points@R.T)[:,2]-radii)
    target_clearance=height*(1-smooth((f-850)/30))
    recovery=smooth((f-900)/25)
    target_clearance=(1-recovery)*target_clearance+recovery*clearance
    target[2]+=target_clearance-clearance
   q,pe,re=solve_leg_pose(model,q,side,target,R);errors.append((pe,re))
  out[f]=q;records.append([f,*np.max(errors,axis=0)])
 records=np.array(records);t=torch.tensor(out);curve=BlendedSplineReference(MotionInterpolator(t),t,stride=6,start_frame=130,ramp_frames=20)
 dense=np.array([curve.kinematics(f,240)[0].numpy()[0] for f in np.arange(824,988,.2)])
 margin=float(np.minimum(dense[:,7:]-limits[:,0],limits[:,1]-dense[:,7:]).min());clearances=[];landing_errors=[]
 for f in np.arange(870.,908.,.2):
  dq=curve.kinematics(f,240)[0].numpy()[0];pos,R,_=model.evaluate(dq,0);clearances.append([float(f),float(pos[2]+np.min((points@R.T)[:,2]-radii))])
  if f>=884:
   sq=MotionInterpolator(torch.tensor(source))(float(f)).numpy()[0];landing_errors.append(float(np.linalg.norm(pos-model.evaluate(sq,0)[0])))
 report=dict(root_drop=drop,landing_frame=880,peak_position_residual=float(records[:,1].max()),peak_orientation_residual=float(records[:,2].max()),peak_source_joint_reference_rmse=float(np.sqrt(np.mean((out[:,7:]-source[:,7:])**2,axis=1)).max()),minimum_dense_joint_limit_margin=margin,minimum_dense_clearance=float(np.min(np.array(clearances)[:,1])),maximum_source_landing_foot_error=max(landing_errors),native_clearance_at884=next(row[1] for row in clearances if abs(row[0]-884)<1e-6),physical_tracking_certified=False)
 report['preflight_passed']=bool(report['peak_position_residual']<.002 and report['peak_orientation_residual']<.01 and report['peak_source_joint_reference_rmse']<.19 and margin>=0 and report['minimum_dense_clearance']>=-.002 and max(landing_errors)<.04)
 name='wbc_native_landing880_drop'+str(round(drop*1000));np.savez_compressed(p/(name+'.npz'),control_q=out,fps=200.,baseline_sha256=hashlib.sha256(baseline['control_q'].tobytes()).hexdigest(),source_sha256=hashlib.sha256(source.tobytes()).hexdigest());(p/(name+'.json')).write_text(json.dumps(report,indent=2)+'\n');print(name,report,flush=True)
