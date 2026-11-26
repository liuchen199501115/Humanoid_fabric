"""Align the short tenth swing's yaw before loaded stance locks it.

Only the right foot's world yaw is advanced. Both foot-link positions and
native vertical clearance are retained by bounded IK; original audit unchanged.
"""
from pathlib import Path
import json,numpy as np,torch
from scipy.spatial.transform import Rotation
from fabrics_sim.reference.interpolation import MotionInterpolator
from numpy_leg_kinematics import LegKinematics
from bounded_leg_ik import solve_leg_pose
from blended_spline import BlendedSplineReference
p=Path(__file__).parent;parent=np.load(p/'wbc_tenth_held_out33.npz');base=parent['control_q'];source=np.load(p/'fixed_smooth_early.control_reference.npz')['source_q'];d=np.load(p/'wrench_fixed_ten.npz');limits=d['joint_position_limits'][0];speed=d['history_joint_velocity_limits'][-1].reshape(-1);plan=json.loads((p/'wrench_fixed_ten.steps_plan.json').read_text());model=LegKinematics('src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf',d['final_joint_order']);raw=MotionInterpolator(torch.tensor(base))
def smooth(x):
 x=np.clip(x,0,1);return x**3*(10-15*x+6*x*x)
for lead in [10.]:
 out=base.copy();errors=[]
 for f in range(781,845):
  q=base[f].astype(float);pos,R,_=model.evaluate(q,1);phase=f+lead*smooth((f-780)/12)*(1-smooth((f-820)/24));sample=raw.kinematics(float(phase),200.)[0].numpy()[0];later=model.evaluate(sample,1)[1];delta=np.arctan2(later[1,0],later[0,0])-np.arctan2(R[1,0],R[0,0]);delta=np.arctan2(np.sin(delta),np.cos(delta));target_R=Rotation.from_euler('z',delta).as_matrix()@R;q,pe,re=solve_leg_pose(model,q,1,pos,target_R);out[f]=q;errors.append([pe,re])
 tensor=torch.tensor(out);curve=BlendedSplineReference(MotionInterpolator(tensor),tensor,stride=6,start_frame=130,ramp_frames=20);values=[curve.kinematics(float(f),240.) for f in np.arange(774,852,.2)];dense=np.array([v[0].numpy()[0] for v in values]);vel=np.array([v[1].numpy()[0,6:] for v in values]);margin=np.minimum(dense[:,7:]-limits[:,0],limits[:,1]-dense[:,7:]).min();clear=[]
 for q in dense:
  for side in [0,1]:
   pos,R,_=model.evaluate(q,side);clear.append(pos[2]+np.min((np.array(plan['collision_points'][side])@R.T)[:,2]-np.array(plan['collision_radii'][side])))
 err=np.max(errors,axis=0);rmse=np.sqrt(np.mean((out[:,7:]-source[:,7:])**2,axis=1)).max();fraction=(abs(vel)/speed).max();report=dict(yaw_lead_frames=lead,peak_ik_position_error=float(err[0]),peak_ik_orientation_error=float(err[1]),minimum_joint_margin=float(margin),minimum_native_clearance=float(min(clear)),peak_source_joint_reference_rmse=float(rmse),maximum_joint_velocity_fraction=float(fraction),physical_tracking_certified=False)
 report['preflight_passed']=bool(err[0]<.002 and err[1]<.01 and margin>=0 and min(clear)>=-.002 and rmse<.19 and fraction<=1)
 name='wbc_tenth_yaw_lead10';np.savez_compressed(p/(name+'.npz'),control_q=out,fps=200.,baseline_sha256=parent['baseline_sha256'],source_sha256=parent['source_sha256']);(p/(name+'.json')).write_text(json.dumps(report,indent=2)+'\n');print(name,report,flush=True)
