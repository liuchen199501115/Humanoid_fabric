from pathlib import Path
import numpy as np,json,torch
from scipy.spatial.transform import Rotation
from numpy_g1_com import CenterOfMassModel
from numpy_leg_kinematics import LegKinematics
from bounded_leg_ik import solve_leg_pose
from fabrics_sim.reference.interpolation import MotionInterpolator
from blended_spline import BlendedSplineReference
from centroidal_pose_linearization import pose_variations
from mujoco_reference_dynamics import ReferenceDynamics
from full_effort_linearization import effort_variations,inverse_output
p=Path(__file__).parent;d=np.load(p/'wrench_fixed_ten.npz');urdf='src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf';com=CenterOfMassModel(urdf,d['final_joint_order']);leg=LegKinematics(urdf,d['final_joint_order']);model=ReferenceDynamics(urdf,d['final_joint_order']);t=torch.tensor(np.load(p/'turn_recovery_y40.npz')['control_q'],dtype=torch.float64);curve=BlendedSplineReference(MotionInterpolator(t),t,stride=6,start_frame=130,ramp_frames=20)
def project(frame,u):
 q=curve.kinematics(frame,240.)[0].numpy()[0];target=com.evaluate(q)+u[:3];feet=[leg.evaluate(q,j)[:2] for j in [0,1]];q[3:7]=(Rotation.from_rotvec(u[3:6])*Rotation.from_quat(q[3:7])).as_quat();q[19:]+=u[6:]
 for _ in range(35):
  q[:3]+=1.2*(target-com.evaluate(q))
  for j,(pos,R) in enumerate(feet):q,pe,re=solve_leg_pose(leg,q,j,pos,R,position_tolerance=1e-12,orientation_tolerance=1e-11)
  if np.linalg.norm(com.evaluate(q)-target)<1e-12:break
 return q
h=.0005;u=np.r_[[.2,-.3,-.2,.3,-.2,.1],np.sin(np.arange(17))*.1]*1e-4;ud=.3*u;ua=-.7*u;reports=[]
def state(frame,scale):
 q=project(frame,scale*u);qm=project(frame-240*h,scale*(u-h*ud+.5*h*h*ua));qp=project(frame+240*h,scale*(u+h*ud+.5*h*h*ua));rp,r,rm=[Rotation.from_quat(x[3:7]) for x in [qp,q,qm]]
 v=np.r_[(qp[:3]-qm[:3])/(2*h),(rp*rm.inv()).as_rotvec()/(2*h),(qp[7:]-qm[7:])/(2*h)]
 a=np.r_[(qp[:3]-2*q[:3]+qm[:3])/h**2,((rp*r.inv()).as_rotvec()-(r*rm.inv()).as_rotvec())/h**2,(qp[7:]-2*q[7:]+qm[7:])/h**2];return q,v,a
for frame,foot in [(850.,1),(870.,1),(930.,0)]:
 frames=np.arange(frame-1,frame+1.01,.125);times=frames/240.;values=[curve.kinematics(float(f),240.) for f in frames];qs=np.array([x[0].numpy()[0] for x in values]);vs=np.array([x[1].numpy()[0] for x in values]);ac=np.array([x[2].numpy()[0] for x in values]);T,Td,*_=pose_variations(com,leg,qs,vs,times,upper=True);k=8;q0,v0,a0=state(frame,0);q1,v1,a1=state(frame,1);qs[k],vs[k],ac[k]=q0,v0,a0;raw,P,V,A=effort_variations(model,qs,vs,ac,T,Td,times,[k],[foot])[k];actual=inverse_output(model.evaluate(q1,v1,a1),foot)-inverse_output(model.evaluate(q0,v0,a0),foot);predicted=P@u+V@ud+A@ua;report=dict(frame=frame,relative_error=float(np.linalg.norm(actual-predicted)/np.linalg.norm(actual)),maximum_absolute_error=float(abs(actual-predicted).max()),maximum_force_torque_variation=float(abs(actual).max()));reports.append(report);print(report,flush=True)
report=dict(samples=reports,passed=all(r['relative_error']<.03 for r in reports));(p/'full_effort_linearization_audit.json').write_text(json.dumps(report,indent=2)+'\n');assert report['passed']
