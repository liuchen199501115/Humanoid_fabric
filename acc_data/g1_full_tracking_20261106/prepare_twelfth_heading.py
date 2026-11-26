"""Finish incoming foot yaw before touchdown, retaining its link path."""
from pathlib import Path
import json,numpy as np,torch
from scipy.spatial.transform import Rotation
from fabrics_sim.reference.interpolation import MotionInterpolator
from numpy_leg_kinematics import LegKinematics
from bounded_leg_ik import solve_leg_pose
from blended_spline import BlendedSplineReference
from reference_joint_extrema import joint_extrema
p=Path(__file__).parent;parent=np.load('acc_data/amp_walk_20261103/feedback_followup/wbc_convex_catch15_25_clear35.npz');base=parent['control_q'];source=np.load(p/'rolling_contact.control_reference.npz')['source_q'];d=np.load(p/'rolling_contact.npz')
model=LegKinematics('src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf',d['final_joint_order']);plan=json.loads((p/'rolling_contact.steps_plan.json').read_text())
def smooth(x):
 x=np.clip(x,0,1);return x**3*(10-15*x+6*x*x)
def curve(q):
 t=torch.tensor(q,dtype=torch.float64);return BlendedSplineReference(MotionInterpolator(t),t,stride=6,start_frame=130,ramp_frames=20)
R=model.evaluate(base[980],1)[1];heading=np.arctan2(R[1,0],R[0,0]);out=base.copy();errors=[]
for f in range(901,1030):
 q=base[f].astype(float);pos,R,_=model.evaluate(q,1);delta=heading-np.arctan2(R[1,0],R[0,0]);delta=np.arctan2(np.sin(delta),np.cos(delta));delta*=smooth((f-900)/40)*smooth((1030-f)/30)
 target=Rotation.from_euler('z',delta).as_matrix()@R;q,pe,re=solve_leg_pose(model,q,1,pos,target,position_tolerance=1e-9,orientation_tolerance=1e-8);out[f]=q;errors.append([pe,re])
r=curve(out);old=curve(base);geometry=[];clear=[];oldclear=[]
for f in np.arange(894,1036.01,.25):
 q=r.kinematics(float(f),240.)[0].numpy()[0];original=old.kinematics(float(f),240.)[0].numpy()[0]
 for side in (0,1):
  pos,R,_=model.evaluate(q,side);op,oR,_=model.evaluate(original,side);points=np.array(plan['collision_points'][side]);radii=np.array(plan['collision_radii'][side]);geometry.append(np.linalg.norm(pos-op));clear.append(pos[2]+min((points@R.T)[:,2]-radii));oldclear.append(op[2]+min((points@oR.T)[:,2]-radii))
continuous=joint_extrema(r.smooth,894.,1036.,240.,d['joint_position_limits'][0],d['history_joint_velocity_limits'][-1].reshape(-1));errors=np.array(errors)
report=dict(heading_target_rad=float(heading),blend_begin=900,blend_full=940,fade_begin=1000,fade_end=1030,peak_ik_position=float(errors[:,0].max()),peak_ik_orientation=float(errors[:,1].max()),maximum_dense_foot_change=float(max(geometry)),minimum_native_clearance=float(min(clear)),minimum_parent_clearance=float(min(oldclear)),peak_source_joint_reference_rmse=float(np.sqrt(((out[:,7:]-source[:,7:])**2).mean(1)).max()),physical_tracking_certified=False,**continuous)
report['preflight_passed']=bool(errors[:,0].max()<.002 and errors[:,1].max()<.01 and max(geometry)<.002 and min(clear)>=min(oldclear)-.0003 and continuous['minimum_continuous_joint_margin']>=0 and continuous['maximum_continuous_joint_velocity_fraction']<=1 and report['peak_source_joint_reference_rmse']<.19)
path=p/'twelfth_heading_hold';path.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if not k.startswith('joint_')},indent=2),flush=True)
if not report['preflight_passed']:raise RuntimeError('Heading candidate failed kinematic preflight')
np.savez_compressed(path.with_suffix('.npz'),control_q=out,fps=200.,baseline_sha256=parent['baseline_sha256'],source_sha256=parent['source_sha256'])
