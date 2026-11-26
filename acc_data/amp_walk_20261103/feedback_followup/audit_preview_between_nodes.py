from pathlib import Path
import sys,json,numpy as np,torch
from scipy.spatial.transform import Rotation
from full_body_preview import integrate
from mujoco_reference_dynamics import ReferenceDynamics
from blended_spline import BlendedSplineReference
from fabrics_sim.reference.interpolation import MotionInterpolator
p=Path(__file__).parent;saved=np.load(p/'wrench_fixed_ten.npz');model=ReferenceDynamics('src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf',saved['final_joint_order']);base=torch.tensor(np.load(p/'turn_recovery_y40.npz')['control_q']);reference=BlendedSplineReference(MotionInterpolator(base),base,stride=6,start_frame=130,ramp_frames=20);geometry=json.load(open(p/'wrench_fixed_ten.steps_plan.json'));reports=[]
for run in ['fixed_clock120_body10_full_body_preview42_contact24','fixed_clock120_body10_full_body_preview42_wrench_warm24']:
 file=p/(run+'.full_body_preview.first_plan.npz')
 if not file.exists():continue
 d=np.load(file,allow_pickle=True);rate=float(d['rate']);worst=dict(clearance=float('inf'))
 for k,control in enumerate(d['controls']):
  frame=float(d['frames'][k]);rq,rv,_=[x.numpy().ravel() for x in reference.kinematics(frame,rate)];q=integrate(rq,d['states'][k,:35]);v=rv+d['states'][k,35:]
  for fraction in [0.,.25,.5,.75,1.]:
   dt=.025*fraction;mq=integrate(q,dt*v+.5*dt**2*control[:35]);s=model.evaluate(mq,v+dt*control[:35])
   for side in [0,1]:
    foot=s['feet'][side];R=Rotation.from_quat(foot[3:]).as_matrix();points=np.array(geometry['collision_points'][side]);radii=np.array(geometry['collision_radii'][side]);clearance=float(foot[2]+np.min((points@R.T)[:,2]-radii))
    if clearance<worst['clearance']:worst=dict(clearance=clearance,stage=k,fraction=fraction,foot=side,frame=frame+dt*rate)
 reports.append(dict(run=run,minimum_predicted_native_clearance=worst));print(reports[-1],flush=True)
(p/'full_body_preview_between_nodes_audit.json').write_text(json.dumps(reports,indent=2)+'\n')
