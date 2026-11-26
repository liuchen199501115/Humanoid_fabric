"""Necessary net-contact-force checks for the actual nominal reference curve.
This does not certify that bounded-error physical tracking is impossible.
"""
import json, argparse
from pathlib import Path
import numpy as np
import torch
from fabrics_sim.reference.interpolation import MotionInterpolator
from blended_spline import BlendedSplineReference
from smooth_clock import smooth_clock
from mujoco_reference_dynamics import ReferenceDynamics
parser=argparse.ArgumentParser();parser.add_argument('--second-start',type=float,default=100.);parser.add_argument('--second-ramp',type=float,default=.3);parser.add_argument('--second-speed',type=float,default=None);parser.add_argument('--name',default='reference');args=parser.parse_args()
out=Path(__file__).parent
r=np.load(out/'rolling_contact.control_reference.npz');d=np.load(out/'rolling_contact.npz')
q=torch.tensor(r['control_q'],dtype=torch.float64)
curve=BlendedSplineReference(MotionInterpolator(q),q,stride=int(r['smooth_stride']),start_frame=float(r['smooth_start'])*200,ramp_frames=float(r['smooth_ramp'])*200)
model=ReferenceDynamics('src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf',d['final_joint_order'])
rows=[]
for time in np.arange(0., 18., .004):
 phase,rate,acc=smooth_clock(float(time),.9,.35,1.2,args.second_start,args.second_ramp,args.second_speed);frame=phase*200
 if frame>1999:break
 state=curve.kinematics(frame,rate*200,acc*200)
 pose,vel,a=[x.numpy()[0] for x in state]
 force=model.evaluate(pose,vel,a)['generalized_force'][:3]
 rows.append([float(time),frame,*force])
rows=np.array(rows);fz=rows[:,4];mu=np.linalg.norm(rows[:,2:4],axis=1)/np.maximum(fz,1e-9)
report=dict(configured_ground_material=dict(static_friction=1.,dynamic_friction=.8),clock=dict(first_start=.9,first_ramp=.35,first_speed=1.2,second_start=args.second_start,second_ramp=args.second_ramp,second_speed=args.second_speed),scope='Negative total normal force is infeasible on unassisted level ground. The horizontal-force ratio uses a diagnostic circular 0.6 budget, not the exact configuration-dependent QP cones or effective simulator friction. This does not prove impossibility inside the original tracking error bounds.',samples=len(rows),negative_normal_samples=int((fz<0).sum()),minimum_required_normal_N=float(fz.min()),worst_frame=float(rows[fz.argmin(),1]),circular_budget_06_exceeded_samples=int(((fz<=0)|(mu>.6)).sum()),turn_samples=[])
for start,end in [(740,880),(880,955),(955,1073),(1073,1999)]:
 mask=(rows[:,1]>=start)&(rows[:,1]<=end)
 report['turn_samples'].append(dict(frame_range=[start,end],samples=int(mask.sum()),negative_normal_samples=int(((fz<0)&mask).sum()),minimum_required_normal_N=float(fz[mask].min()),circular_budget_06_exceeded_samples=int((((fz<=0)|(mu>.6))&mask).sum())))
np.savez_compressed(out/(args.name+'_required_force.npz'),time=rows[:,0],frame=rows[:,1],force=rows[:,2:5])
(out/(args.name+'_force_audit.json')).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
