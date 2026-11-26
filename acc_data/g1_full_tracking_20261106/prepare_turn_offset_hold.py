"""Delay removal of the existing 4 cm root correction, preserving guide feet.
This is a kinematic preflight only; physical and raw-source audits remain required.
"""
from pathlib import Path
import json, numpy as np, torch
from fabrics_sim.reference.interpolation import MotionInterpolator
from numpy_leg_kinematics import LegKinematics
from bounded_leg_ik import solve_leg_pose
from blended_spline import BlendedSplineReference
from reference_joint_extrema import joint_extrema
outdir=Path('acc_data/g1_full_tracking_20261106')
parent=np.load('acc_data/amp_walk_20261103/feedback_followup/wbc_convex_catch15_25_clear35.npz')
base=parent['control_q']; source=np.load(outdir/'rolling_contact.control_reference.npz')['source_q']
record=np.load(outdir/'rolling_contact.npz')
model=LegKinematics('src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf',record['final_joint_order'])
limits=record['joint_position_limits'][0]; speeds=record['history_joint_velocity_limits'][-1].reshape(-1)
plan=json.loads((outdir/'rolling_contact.steps_plan.json').read_text())
def smooth(x):
    x=np.clip(x,0,1); return x**3*(10-15*x+6*x*x)
def curve(q):
    t=torch.tensor(q); return BlendedSplineReference(MotionInterpolator(t),t,stride=6,start_frame=130,ramp_frames=20)
out=base.copy(); errors=[]
for frame in range(941,1150):
    q=base[frame].astype(float).copy()
    q[1]+=.04*(smooth((frame-940)/60)-smooth((frame-1070)/80))
    for side in (0,1):
        position,rotation,_=model.evaluate(base[frame],side)
        q,pe,re=solve_leg_pose(model,q,side,position,rotation)
        errors.append([pe,re])
    out[frame]=q
reference=curve(out); previous=curve(base)
continuous=joint_extrema(reference.smooth,934.,1156.,240.,limits,speeds)
minimum_clearance=float('inf'); peak_foot_change=0.
for frame in np.arange(934.,1156.01,.25):
    q=reference.kinematics(float(frame),240.)[0].numpy()[0]
    old=previous.kinematics(float(frame),240.)[0].numpy()[0]
    for side in (0,1):
        position,rotation,_=model.evaluate(q,side)
        old_position,_,_=model.evaluate(old,side)
        peak_foot_change=max(peak_foot_change,float(np.linalg.norm(position-old_position)))
        height=position[2]+np.min((np.array(plan['collision_points'][side])@rotation.T)[:,2]-np.array(plan['collision_radii'][side]))
        minimum_clearance=min(minimum_clearance,float(height))
errors=np.array(errors)
report=dict(parent='wbc_convex_catch15_25_clear35.npz',root_y_hold_until_frame=1070,root_y_return_end_frame=1150,
    peak_ik_position_error=float(errors[:,0].max()),peak_ik_orientation_error=float(errors[:,1].max()),
    maximum_dense_guide_foot_change_m=peak_foot_change,minimum_dense_native_clearance_m=minimum_clearance,
    peak_raw_source_joint_rmse_rad=float(np.sqrt(((out[:,7:]-source[:,7:])**2).mean(1)).max()),
    peak_raw_source_root_error_m=float(np.linalg.norm(out[:,:3]-source[:,:3],axis=1).max()),
    physical_tracking_certified=False,**continuous)
report['preflight_passed']=bool(errors[:,0].max()<.002 and errors[:,1].max()<.01
    and continuous['minimum_continuous_joint_margin']>=0 and continuous['maximum_continuous_joint_velocity_fraction']<=1
    and minimum_clearance>=-.002 and peak_foot_change<.002 and report['peak_raw_source_joint_rmse_rad']<.19)
path=outdir/'turn_offset_hold1070'
path.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if not k.startswith('joint_')},indent=2),flush=True)
if not report['preflight_passed']:raise RuntimeError('Candidate rejected by kinematic preflight')
np.savez_compressed(path.with_suffix('.npz'),control_q=out,fps=200.,baseline_sha256=parent['baseline_sha256'],source_sha256=parent['source_sha256'])
