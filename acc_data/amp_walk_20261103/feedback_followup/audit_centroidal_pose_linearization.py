from pathlib import Path
import json,numpy as np,torch
from scipy.spatial.transform import Rotation
from numpy_g1_com import CenterOfMassModel
from numpy_leg_kinematics import LegKinematics
from bounded_leg_ik import solve_leg_pose
from fabrics_sim.reference.interpolation import MotionInterpolator
from blended_spline import BlendedSplineReference
from centroidal_pose_linearization import pose_variations,skew
p=Path(__file__).parent;d=np.load(p/'wrench_fixed_ten.npz');urdf='src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf';com=CenterOfMassModel(urdf,d['final_joint_order']);leg=LegKinematics(urdf,d['final_joint_order']);t=torch.tensor(np.load(p/'turn_force_feasible_com_floor.npz')['control_q'],dtype=torch.float64);curve=BlendedSplineReference(MotionInterpolator(t),t,stride=6,start_frame=130,ramp_frames=20);frames=np.arange(802,842.01,.25);times=frames/240;values=[curve.kinematics(float(f),240.) for f in frames];qs=np.array([x[0].numpy()[0] for x in values]);vs=np.array([x[1].numpy()[0] for x in values]);T,Td,P,Q,Pd,Qd=pose_variations(com,leg,qs,vs,times);contact_residual=0.;com_residual=0.
for k,q in enumerate(qs):
 for side in [0,1]:
  pos,R,J=leg.evaluate(q,side);base=np.block([[np.eye(3),-skew(pos-q[:3])],[np.zeros((3,3)),np.eye(3)]]);contact_residual=max(contact_residual,float(abs(base@T[k,:6]+J@T[k,6+6*side:12+6*side]).max()))
 S=np.stack([com.momentum(q,T[k,:,j],return_state=True)[1] for j in range(6)],axis=1);com_residual=max(com_residual,float(abs(S-np.c_[np.eye(3),np.zeros((3,3))]).max()))
def project(frame,u):
 q=curve.kinematics(frame,240.)[0].numpy()[0];target=com.evaluate(q)+u[:3];feet=[leg.evaluate(q,j)[:2] for j in [0,1]];q[3:7]=(Rotation.from_rotvec(u[3:])*Rotation.from_quat(q[3:7])).as_quat()
 for _ in range(16):
  q[:3]+=1.25*(target-com.evaluate(q))
  for j,(pos,R) in enumerate(feet):q,pe,re=solve_leg_pose(leg,q,j,pos,R,position_tolerance=1e-11,orientation_tolerance=1e-10)
  if np.linalg.norm(com.evaluate(q)-target)<1e-10:break
 return q
errors=[]
u=np.array([.3,-.2,-.1,.4,.2,-.3])*1e-4;h=.00025
for f in [806.,820.,838.]:
 k=np.argmin(abs(frames-f));q=project(f,u);qm=project(f-240*h,u);qp=project(f+240*h,u);omega=(Rotation.from_quat(qp[3:7])*Rotation.from_quat(qm[3:7]).inv()).as_rotvec()/(2*h);v=np.r_[(qp[:3]-qm[:3])/(2*h),omega,(qp[7:]-qm[7:])/(2*h)];qm0=project(f-240*h,np.zeros(6));qp0=project(f+240*h,np.zeros(6));q0=project(f,np.zeros(6));omega0=(Rotation.from_quat(qp0[3:7])*Rotation.from_quat(qm0[3:7]).inv()).as_rotvec()/(2*h);v0=np.r_[(qp0[:3]-qm0[:3])/(2*h),omega0,(qp0[7:]-qm0[7:])/(2*h)];actual=com.momentum(q,v)-com.momentum(q0,v0);prediction=P[k]@u;errors.append(dict(frame=f,observed=actual.tolist(),predicted=prediction.tolist(),relative_error=float(np.linalg.norm(actual-prediction)/max(np.linalg.norm(actual),1e-12))))
report=dict(maximum_foot_velocity_constraint_residual=contact_residual,maximum_com_velocity_map_residual=com_residual,momentum_variation=errors);report['passed']=bool(contact_residual<1e-8 and com_residual<1e-8 and max(x['relative_error'] for x in errors)<.03);(p/'centroidal_pose_linearization_audit.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
