"""After measured-transfer preparation, advance the next left swing smoothly."""
from pathlib import Path
import json
import numpy as np,torch
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation,Slerp
from numpy_leg_kinematics import LegKinematics
from bounded_leg_ik import solve_leg_pose
from blended_spline import BlendedSplineReference
from fabrics_sim.reference.interpolation import MotionInterpolator
p=Path(__file__).parent;data=np.load(p/'wbc_tenth_transfer_bridge808.npz');base=data['control_q'];source=np.load(p/'fixed_smooth_early.control_reference.npz')['source_q'];d=np.load(p/'wrench_fixed_ten.npz');limits=d['joint_position_limits'][0];speed=d['history_joint_velocity_limits'][-1].ravel();plan=json.loads((p/'wrench_fixed_ten.steps_plan.json').read_text());model=LegKinematics('src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf',d['final_joint_order']);frames=np.arange(808,891);poses=[model.evaluate(base[f],0)[:2] for f in frames];path=CubicSpline(frames,np.array([r[0] for r in poses]));rotation=Slerp(frames,Rotation.from_matrix(np.array([r[1] for r in poses])))
def smooth(x):x=np.clip(x,0,1);return x**3*(10-15*x+6*x*x)
for lead in [8.]:
 out=base.copy();errors=[]
 for f in range(815,885):
  target_frame=f+lead*smooth((f-814)/16)*(1-smooth((f-860)/24));q,pe,re=solve_leg_pose(model,base[f].astype(float),0,path(target_frame),rotation(target_frame).as_matrix());out[f]=q;errors.append((pe,re))
 t=torch.tensor(out);curve=BlendedSplineReference(MotionInterpolator(t),t,stride=6,start_frame=130,ramp_frames=20);samples=[curve.kinematics(float(f),240.) for f in np.arange(808,891,.2)];qs=np.array([v[0].numpy()[0] for v in samples]);vs=np.array([v[1].numpy()[0] for v in samples]);margin=np.minimum(qs[:,7:]-limits[:,0],limits[:,1]-qs[:,7:]).min();clear=[]
 for q in qs:
  for side in [0,1]:
   pos,R,_=model.evaluate(q,side);clear.append(pos[2]+np.min((np.array(plan['collision_points'][side])@R.T)[:,2]-np.array(plan['collision_radii'][side])))
 err=np.max(errors,axis=0);rmse=np.sqrt(np.mean((out[:,7:]-source[:,7:])**2,axis=1)).max();fraction=(abs(vs[:,6:])/speed).max();report=dict(parent='wbc_tenth_transfer_bridge808',lead_frames=lead,peak_ik_position_error=float(err[0]),peak_ik_orientation_error=float(err[1]),minimum_joint_margin=float(margin),minimum_native_clearance=float(min(clear)),peak_source_joint_reference_rmse=float(rmse),maximum_joint_velocity_fraction=float(fraction),physical_tracking_certified=False)
 report['preflight_passed']=bool(err[0]<.002 and err[1]<.01 and margin>=0 and min(clear)>=-.002 and rmse<.19 and fraction<=1)
 name='wbc_transfer_bridge_left_lead8';np.savez_compressed(p/(name+'.npz'),control_q=out,fps=200.,baseline_sha256=data['baseline_sha256'],source_sha256=data['source_sha256']);(p/(name+'.json')).write_text(json.dumps(report,indent=2)+'\n');print(report,flush=True)
 if not report['preflight_passed']:raise RuntimeError('Swing catch-up preflight failed')
