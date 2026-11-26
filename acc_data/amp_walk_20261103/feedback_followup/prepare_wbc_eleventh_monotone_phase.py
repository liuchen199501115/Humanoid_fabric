"""Delay only the eleventh swing onset, preserving its original end pose.

The bounded IK output is sampled as one q/v/a curve; the original source audit
and actual measured flight/landing requirements remain unchanged.
"""
from pathlib import Path
import json
import numpy as np
import torch
from blended_spline import BlendedSplineReference
from bounded_leg_ik import solve_leg_pose
from numpy_leg_kinematics import LegKinematics
from fabrics_sim.reference.interpolation import MotionInterpolator
p=Path(__file__).parent
parent=np.load(p/'turn_recovery_y40.npz');base=parent['control_q'];d=np.load(p/'wrench_fixed_ten.npz')
source=np.load(p/'fixed_smooth_early.control_reference.npz')['source_q']
model=LegKinematics('src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf',d['final_joint_order'])
limits=d['joint_position_limits'][0];speed=d['history_joint_velocity_limits'][-1].reshape(-1)
plan=json.loads((p/'wrench_fixed_ten.steps_plan.json').read_text())
raw=MotionInterpolator(torch.tensor(base))
def smooth(x):
    x=np.clip(x,0.,1.);return x**3*(10-15*x+6*x*x)
for delay in [16.]:
    out=base.copy();errors=[]
    for frame in range(803,884):
        weight=smooth((frame-802)/32)*(1-smooth((frame-836)/48))
        phase=frame-delay*weight
        pose=raw.kinematics(float(phase),200.)[0].numpy()[0]
        target,R,_=model.evaluate(pose,0)
        q,pe,re=solve_leg_pose(model,base[frame].astype(float),0,target,R)
        out[frame]=q;errors.append([pe,re])
    curve=BlendedSplineReference(MotionInterpolator(torch.tensor(out)),torch.tensor(out),stride=6,start_frame=130,ramp_frames=20)
    values=[curve.kinematics(float(frame),240.) for frame in np.arange(796,891,.2)]
    poses=np.array([x[0].numpy()[0] for x in values]);vel=np.array([x[1].numpy()[0,6:] for x in values])
    clear=np.full(2,np.inf)
    for pose in poses:
        for side in [0,1]:
            pos,R,_=model.evaluate(pose,side)
            z=pos[2]+np.min((np.array(plan['collision_points'][side])@R.T)[:,2]-np.array(plan['collision_radii'][side]))
            clear[side]=min(clear[side],z)
    err=np.max(errors,axis=0);margin=np.minimum(poses[:,7:]-limits[:,0],limits[:,1]-poses[:,7:]).min()
    rmse=np.sqrt(np.mean((out[:,7:]-source[:,7:])**2,axis=1)).max()
    report=dict(delay_frames=delay,peak_ik_position_error=float(err[0]),peak_ik_orientation_error=float(err[1]),minimum_joint_margin=float(margin),minimum_native_clearance_per_foot=clear.tolist(),peak_source_joint_reference_rmse=float(rmse),maximum_joint_velocity_fraction=float((abs(vel)/speed).max()),physical_tracking_certified=False)
    phase_grid=np.linspace(802,884,8201)
    phase=phase_grid-delay*smooth((phase_grid-802)/32)*(1-smooth((phase_grid-836)/48))
    report['minimum_phase_speed']=float(np.min(np.diff(phase)/np.diff(phase_grid)))
    report['preflight_passed']=bool(report['minimum_phase_speed']>=0 and err[0]<.002 and err[1]<.01 and margin>=0 and rmse<.19 and clear.min()>=-.002 and report['maximum_joint_velocity_fraction']<=1)
    name=f'wbc_eleventh_monotone_delay{int(delay)}'
    np.savez_compressed(p/(name+'.npz'),control_q=out,fps=200.,baseline_sha256=parent['baseline_sha256'],source_sha256=parent['source_sha256'])
    (p/(name+'.json')).write_text(json.dumps(report,indent=2)+'\n');print(name,json.dumps(report),flush=True)
