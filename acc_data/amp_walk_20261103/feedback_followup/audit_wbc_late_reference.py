from pathlib import Path
import json,numpy as np,torch
from fabrics_sim.reference.interpolation import MotionInterpolator
from blended_spline import BlendedSplineReference
from late_tube_smoothing import transition_tube_spline
from numpy_leg_kinematics import LegKinematics
p=Path(__file__).parent;data=np.load(p/'wrench_fixed_ten.npz');q=torch.tensor(np.load(p/'turn_recovery_y40.npz')['control_q']);base=BlendedSplineReference(MotionInterpolator(q),q,stride=6,start_frame=130,ramp_frames=20);model=LegKinematics('src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf',data['final_joint_order']);limits=data['joint_position_limits'][0];speed=data['history_joint_velocity_limits'][-1].reshape(-1);plan=json.loads((p/'wrench_fixed_ten.steps_plan.json').read_text());records=[]
for stride in [10,12]:
 curve=transition_tube_spline(base,q,stride=stride,start_frame=830,ramp_frames=30);samples=[curve.kinematics(float(f),240.) for f in np.arange(824,1010,.25)];poses=np.array([s[0].numpy()[0] for s in samples]);vel=np.array([s[1].numpy()[0] for s in samples]);acc=np.array([s[2].numpy()[0] for s in samples]);clear=[]
 for pose in poses:
  for side in [0,1]:
   pos,R,_=model.evaluate(pose,side);pts=np.array(plan['collision_points'][side]);radii=np.array(plan['collision_radii'][side]);clear.append(pos[2]+min((pts@R.T)[:,2]-radii))
 row=dict(stride=stride,minimum_native_clearance=float(min(clear)),minimum_joint_margin=float(np.minimum(poses[:,7:]-limits[:,0],limits[:,1]-poses[:,7:]).min()),max_velocity_fraction=float((abs(vel[:,6:])/speed).max()),max_root_angular_acceleration=float(abs(acc[:,3:6]).max()),physical_tracking_certified=False)
 row['preflight_passed']=bool(row['minimum_native_clearance']>=-.002 and row['minimum_joint_margin']>=0 and row['max_velocity_fraction']<=1);records.append(row)
print(json.dumps(records,indent=2));(p/'wbc_late_reference_preflight.json').write_text(json.dumps(records,indent=2)+'\n')
