"""Level only the incoming right sole roll using bounded IK and native clearance.
Original-source audit and contact timing are retained; this is not a dynamics certificate.
"""
from pathlib import Path
import json, hashlib
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from fabrics_sim.reference.interpolation import MotionInterpolator
from numpy_leg_kinematics import LegKinematics
from bounded_leg_ik import solve_leg_pose
from blended_spline import BlendedSplineReference
from reference_joint_extrema import joint_extrema
p=Path(__file__).parent
parent_path=Path('acc_data/amp_walk_20261103/feedback_followup/wbc_convex_catch15_25_clear35.npz')
parent=np.load(parent_path);base=parent['control_q'];out=base.copy()
d=np.load(p/'rolling_contact.npz');source=np.load(p/'rolling_contact.control_reference.npz')['source_q']
model=LegKinematics('src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf',d['final_joint_order'])
plan=json.loads((p/'rolling_contact.steps_plan.json').read_text())
points=[np.array(x) for x in plan['collision_points']];radii=[np.array(x) for x in plan['collision_radii']]
def smooth(x):
 x=np.clip(x,0,1);return x**3*(10-15*x+6*x*x)
def curve(q):
 t=torch.tensor(q,dtype=torch.float64);return BlendedSplineReference(MotionInterpolator(t),t,stride=6,start_frame=130,ramp_frames=20)
def height(pos,R,side):return pos[2]+np.min((points[side]@R.T)[:,2]-radii[side])
heading_R=model.evaluate(base[980],1)[1];heading=float(np.arctan2(heading_R[1,0],heading_R[0,0]))
errors=[]
for f in range(909,1060):
 pos,R,_=model.evaluate(base[f],1);rpy=Rotation.from_matrix(R).as_euler('xyz');flat=Rotation.from_euler('xyz',[0.,rpy[1],heading])
 weight=smooth((f-908)/32)*smooth((1060-f)/30)
 target=(Rotation.from_rotvec(weight*(flat*Rotation.from_matrix(R).inv()).as_rotvec())*Rotation.from_matrix(R)).as_matrix()
 target_pos=pos.copy();target_pos[2]+=height(pos,R,1)-height(pos,target,1)+.001*weight
 q=base[f].astype(float);q[2]-=.012*weight
 left_pos,left_R,_=model.evaluate(base[f],0)
 q,lpe,lre=solve_leg_pose(model,q,0,left_pos,left_R,position_tolerance=1e-9,orientation_tolerance=1e-8)
 errors.append([lpe,lre])
 q,pe,re=solve_leg_pose(model,q,1,target_pos,target,position_tolerance=1e-9,orientation_tolerance=1e-8)
 out[f]=q;errors.append([pe,re])
old=curve(base);new=curve(out);max_xy=0.;max_z=0.;clearance_delta=0.;dense_joint_rmse=0.
for f in np.arange(902.,1066.01,.25):
 q=new.kinematics(float(f),240.)[0].numpy()[0];oldq=old.kinematics(float(f),240.)[0].numpy()[0]
 for side in [0,1]:
  pos,R,_=model.evaluate(q,side);op,oR,_=model.evaluate(oldq,side)
  max_xy=max(max_xy,float(np.linalg.norm(pos[:2]-op[:2])));max_z=max(max_z,float(abs(pos[2]-op[2])))
  clearance_delta=min(clearance_delta,float(height(pos,R,side)-height(op,oR,side)))
 ref=(1-(f%1))*source[int(f)]+(f%1)*source[int(f)+1]
 dense_joint_rmse=max(dense_joint_rmse,float(np.sqrt(((q[7:]-ref[7:])**2).mean())))
extrema=joint_extrema(new.smooth,902.,1066.,240.,d['joint_position_limits'][0],d['history_joint_velocity_limits'][-1].reshape(-1))
err=np.max(errors,axis=0);rmse=float(np.sqrt(((out[:,7:]-source[:,7:])**2).mean(1)).max())
report=dict(parent_sha256=hashlib.sha256(parent_path.read_bytes()).hexdigest(),blend_frames=[908,940,1030,1060],foot='right',heading_target_rad=heading,root_drop_m=.012,clearance_reserve_m=.001,peak_ik_position_m=float(err[0]),peak_ik_orientation_rad=float(err[1]),maximum_dense_foot_xy_change_m=max_xy,maximum_dense_foot_z_change_m=max_z,minimum_clearance_change_m=clearance_delta,peak_source_joint_reference_rmse_rad=rmse,dense_source_joint_reference_rmse_rad=dense_joint_rmse,physical_tracking_certified=False,source_contact_plan_unchanged=True,**extrema)
report['preflight_passed']=bool(err[0]<.002 and err[1]<.01 and max_xy<.002 and max_z<.03 and clearance_delta>=-.0005 and max(rmse,dense_joint_rmse)<.19 and extrema['minimum_continuous_joint_margin']>=0 and extrema['maximum_continuous_joint_velocity_fraction']<=1)
path=p/'twelfth_flat_roll_heading_crouch12';path.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if not k.startswith('joint_')},indent=2),flush=True)
if not report['preflight_passed']:raise RuntimeError('Flat support candidate rejected by kinematic preflight')
np.savez_compressed(path.with_suffix('.npz'),control_q=out,fps=200.,baseline_sha256=parent['baseline_sha256'],source_sha256=parent['source_sha256'])
