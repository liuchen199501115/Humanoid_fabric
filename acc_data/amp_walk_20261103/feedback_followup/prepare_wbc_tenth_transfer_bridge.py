"""Briefly retain the departing left foot during step-10 load transfer.

A modified q/v/a guide requests physical contact; measured forces still decide
runtime support. Original labels and all independent acceptance gates stay intact.
"""
from pathlib import Path
import json
import numpy as np,torch
from scipy.spatial.transform import Rotation
from numpy_leg_kinematics import LegKinematics
from bounded_leg_ik import solve_leg_pose
from blended_spline import BlendedSplineReference
from fabrics_sim.reference.interpolation import MotionInterpolator
p=Path(__file__).parent;data=np.load(p/'wbc_tenth_held_out33.npz');base=data['control_q'];source=np.load(p/'fixed_smooth_early.control_reference.npz')['source_q'];d=np.load(p/'wrench_fixed_ten.npz');limits=d['joint_position_limits'][0];speed=d['history_joint_velocity_limits'][-1].ravel();plan=json.loads((p/'wrench_fixed_ten.steps_plan.json').read_text());model=LegKinematics('src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf',d['final_joint_order'])
def smooth(x):
 x=np.clip(x,0,1);return x**3*(10-15*x+6*x*x)
anchor,Ranchor,_=model.evaluate(base[798],0);points=np.array(plan['collision_points'][0]);radii=np.array(plan['collision_radii'][0]);anchor[2]-=anchor[2]+np.min((points@Ranchor.T)[:,2]-radii)
for hold_end in [808,812]:
 out=base.copy();errors=[]
 for f in range(793,hold_end+17):
  q=base[f].astype(float);pos,R,_=model.evaluate(q,0);w=smooth((f-792)/6)*(1-smooth((f-hold_end)/16));target=pos+w*(anchor-pos);targetR=Rotation.from_rotvec(w*Rotation.from_matrix(Ranchor@R.T).as_rotvec()).as_matrix()@R;q,pe,re=solve_leg_pose(model,q,0,target,targetR);out[f]=q;errors.append((pe,re))
 t=torch.tensor(out);curve=BlendedSplineReference(MotionInterpolator(t),t,stride=6,start_frame=130,ramp_frames=20);samples=[curve.kinematics(float(f),240.) for f in np.arange(786,hold_end+23,.2)];qs=np.array([x[0].numpy()[0] for x in samples]);vs=np.array([x[1].numpy()[0] for x in samples]);margin=np.minimum(qs[:,7:]-limits[:,0],limits[:,1]-qs[:,7:]).min();clear=[]
 for q in qs:
  for side in [0,1]:
   pos,R,_=model.evaluate(q,side);clear.append(pos[2]+np.min((np.array(plan['collision_points'][side])@R.T)[:,2]-np.array(plan['collision_radii'][side])))
 rmse=np.sqrt(np.mean((out[:,7:]-source[:,7:])**2,axis=1)).max();err=np.max(errors,axis=0);fraction=(abs(vs[:,6:])/speed).max();report=dict(parent='wbc_tenth_held_out33',anchor_frame=798,hold_end_frame=hold_end,fade_end_frame=hold_end+16,peak_ik_position_error=float(err[0]),peak_ik_orientation_error=float(err[1]),minimum_joint_margin=float(margin),minimum_native_clearance=float(min(clear)),peak_source_joint_reference_rmse=float(rmse),maximum_joint_velocity_fraction=float(fraction),runtime_contact_evidence_required=True,physical_tracking_certified=False)
 report['preflight_passed']=bool(err[0]<.002 and err[1]<.01 and margin>=0 and min(clear)>=-.002 and rmse<.19 and fraction<=1)
 name=f'wbc_tenth_transfer_bridge{hold_end}';np.savez_compressed(p/(name+'.npz'),control_q=out,fps=200.,baseline_sha256=data['baseline_sha256'],source_sha256=data['source_sha256']);(p/(name+'.json')).write_text(json.dumps(report,indent=2)+'\n');print(name,report,flush=True)
 if not report['preflight_passed']:raise RuntimeError('Transfer bridge preflight failed')
