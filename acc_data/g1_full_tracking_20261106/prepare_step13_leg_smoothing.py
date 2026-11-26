"""Bounded local leg-reference smoothing, with unchanged raw-source acceptance."""
from pathlib import Path
import json,numpy as np,torch
from scipy.ndimage import gaussian_filter1d
from numpy_leg_kinematics import LegKinematics
from blended_spline import BlendedSplineReference
from reference_joint_extrema import joint_extrema
from fabrics_sim.reference.interpolation import MotionInterpolator
p=Path(__file__).parent;parent=np.load(p/'thirteenth_force_fit150.npz');base=parent['control_q'].astype(float);source=np.load(p/'rolling_contact.control_reference.npz')['source_q'];record=np.load(p/'rolling_contact.npz');limits=record['joint_position_limits'][0];speed=record['history_joint_velocity_limits'][-1].reshape(-1);plan=json.loads((p/'rolling_contact.steps_plan.json').read_text());geometry=[(np.array(plan['collision_points'][j]),np.array(plan['collision_radii'][j])) for j in [0,1]];leg=LegKinematics('src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf',record['final_joint_order'])
def curve(q):
 t=torch.tensor(q,dtype=torch.float64);return BlendedSplineReference(MotionInterpolator(t),t,stride=6,start_frame=130,ramp_frames=20)
frames=np.arange(894,1068.01,.5);old=curve(base);oldq=np.array([old.kinematics(float(f),240.)[0].numpy()[0] for f in frames]);x=np.arange(len(base));up=np.clip((x-900)/30,0,1);down=np.clip((1062-x)/30,0,1);blend=(up**3*(10-15*up+6*up*up))*(down**3*(10-15*down+6*down*down))
for sigma in [5.]:
 name=f'step13_leg_smooth{int(sigma)}_clearance_audit';out=base.copy();filtered=gaussian_filter1d(base[:,7:13],sigma,axis=0,mode='nearest');out[:,7:13]+=blend[:,None]*(filtered-base[:,7:13]);sampler=curve(out);qs=np.array([sampler.kinematics(float(f),240.)[0].numpy()[0] for f in frames]);delta=[];clear=[];oldclear=[];landing=[]
 for f,q,original in zip(frames,qs,oldq):
  for side in [0,1]:
   pos,R,_=leg.evaluate(q,side);op,oR,_=leg.evaluate(original,side);pts,radius=geometry[side];delta.append(float(np.linalg.norm(pos-op)));clear.append(pos[2]+min((pts@R.T)[:,2]-radius));oldclear.append(op[2]+min((pts@oR.T)[:,2]-radius))
   if 1012<=f<=1014:landing.append(float(np.linalg.norm(pos-op)))
 report=dict(parent='thirteenth_force_fit150',sigma_source_frames=sigma,scope='Local six left-leg joint smoothing; original source is unchanged. These are kinematic checks only.',peak_reference_foot_change_m=max(delta),peak_landing_reference_foot_change_m=max(landing),minimum_clearance_change_m=float(np.min(np.array(clear)-oldclear)),minimum_clearance_m=float(min(clear)),peak_raw_joint_rmse=float(np.sqrt(np.mean((out[:,7:]-source[:,7:])**2,axis=1)).max()))
 ci=int(np.argmin(np.array(clear)-oldclear));left=np.array(clear).reshape(-1,2)[:,0];swing=(frames>=955)&(frames<=1013)
 report.update(minimum_parent_clearance_m=float(min(oldclear)),worst_clearance_change_frame=float(frames[ci//2]),clearance_at_worst_change_m=float(clear[ci]),parent_clearance_at_worst_change_m=float(oldclear[ci]),peak_left_swing_clearance_m=float(left[swing].max()),clearance_rule='Do not add ground penetration beyond the parent by more than 0.5 mm; retain at least 25 mm reference swing clearance. Airborne height need not match the parent pointwise.')
 report.update(joint_extrema(sampler.smooth,894,1068,240.,limits,speed));report['kinematic_preflight_passed']=bool(max(delta)<=.02 and max(landing)<=.01 and min(clear)>=min(oldclear)-.0005 and left[swing].max()>=.025 and report['peak_raw_joint_rmse']<.19 and report['minimum_continuous_joint_margin']>0 and report['maximum_continuous_joint_velocity_fraction']<=1)
 (p/(name+'.json')).write_text(json.dumps(report,indent=2)+'\n')
 if report['kinematic_preflight_passed']:np.savez_compressed(p/(name+'.npz'),control_q=out,fps=200.,baseline_sha256=parent['baseline_sha256'],source_sha256=parent['source_sha256'])
 print(json.dumps({k:v for k,v in report.items() if not k.startswith('joint_')}),flush=True)
