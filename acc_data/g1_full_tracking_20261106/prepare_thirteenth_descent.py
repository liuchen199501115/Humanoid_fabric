"""Lower only step-13 swing height after its verified predecessor.
Foot XY/orientation, original PKL, source clock and contact plan remain intact.
"""
from pathlib import Path
import json,numpy as np,torch
from fabrics_sim.reference.interpolation import MotionInterpolator
from numpy_leg_kinematics import LegKinematics
from bounded_leg_ik import solve_leg_pose
from blended_spline import BlendedSplineReference
from reference_joint_extrema import joint_extrema
p=Path(__file__).parent;parent=np.load('acc_data/amp_walk_20261103/feedback_followup/wbc_convex_catch15_25_clear35.npz');base=parent['control_q'];r=np.load(p/'rolling_contact.control_reference.npz');d=np.load(p/'rolling_contact.npz');source=r['source_q'];limits=d['joint_position_limits'][0];speeds=d['history_joint_velocity_limits'].reshape(-1,29)[0];plan=json.load(open(p/'rolling_contact.steps_plan.json'));model=LegKinematics('src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf',d['final_joint_order']);geometry=[(np.array(plan['collision_points'][side]),np.array(plan['collision_radii'][side])) for side in [0,1]]
def smooth(x):
 x=np.clip(x,0,1);return x**3*(10-15*x+6*x*x)
def curve(q):
 t=torch.tensor(q,dtype=torch.float64);return BlendedSplineReference(MotionInterpolator(t),t,stride=6,start_frame=130,ramp_frames=20)
original=curve(base)
for drop in [.035,.05]:
 out=base.copy();errors=[]
 for frame in range(971,1031):
  q=base[frame].astype(float);pos,R,_=model.evaluate(q,0);points,radii=geometry[0];height=pos[2]+np.min((points@R.T)[:,2]-radii)
  weight=smooth((frame-970)/20)*(1-smooth((frame-1008)/22));target=pos.copy();target[2]-=min(drop*weight,max(0.,height-.0005))
  q,pe,re=solve_leg_pose(model,q,0,target,R);out[frame]=q;errors.append([pe,re])
 c=curve(out);frames=np.arange(964,1037,.25);poses=np.array([c.kinematics(float(f),240.)[0].numpy()[0] for f in frames]);clear=[];horizontal=[];angles=[];source_foot_errors=[]
 for frame,q in zip(frames,poses):
  parent_pose=original.kinematics(float(frame),240.)[0].numpy()[0]
  row=[]
  for side in [0,1]:
   pos,R,_=model.evaluate(q,side);op,OR,_=model.evaluate(parent_pose,side);pts,radii=geometry[side];row.append(pos[2]+np.min((pts@R.T)[:,2]-radii));horizontal.append(np.linalg.norm(pos[:2]-op[:2]));angles.append(np.linalg.norm(R-OR))
  clear.append(row)
 clear=np.array(clear);extrema=joint_extrema(c.smooth,964,1037,240.,limits,speeds);err=np.max(errors,axis=0);rmse=float(np.sqrt(np.mean((out[:,7:]-source[:,7:])**2,axis=1)).max())
 report=dict(drop_m=drop,peak_ik_position_error=float(err[0]),peak_ik_orientation_error=float(err[1]),maximum_dense_horizontal_foot_change_m=float(max(horizontal)),minimum_native_clearance_m=float(clear.min()),peak_source_joint_reference_rmse=rmse,maximum_swing_clearance_before_descent_m=float(clear[(frames>=965)&(frames<=980),0].max()),physical_tracking_certified=False,source_contact_plan_unchanged=True,**extrema)
 report['preflight_passed']=bool(err[0]<.002 and err[1]<.01 and max(horizontal)<.002 and clear.min()>=-.002 and rmse<.19 and extrema['minimum_continuous_joint_margin']>=0 and extrema['maximum_continuous_joint_velocity_fraction']<=1 and report['maximum_swing_clearance_before_descent_m']>=.02)
 name=f'thirteenth_descent{round(drop*1000)}';np.savez_compressed(p/(name+'.npz'),control_q=out,fps=200.,baseline_sha256=parent['baseline_sha256'],source_sha256=parent['source_sha256']);np.savez_compressed(p/(name+'.geometry.npz'),frame=frames,clearance=clear);(p/(name+'.json')).write_text(json.dumps(report,indent=2)+'\n');print(name,json.dumps({k:v for k,v in report.items() if not isinstance(v,list)}),flush=True)
 if not report['preflight_passed']:raise RuntimeError('Candidate geometry or joint preflight failed')
