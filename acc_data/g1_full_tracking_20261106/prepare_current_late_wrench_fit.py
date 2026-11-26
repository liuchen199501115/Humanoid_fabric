"""Minimal offline COM correction of the current reference, preserving foot poses.
Exact runtime q/v/a and rigid-body net forces are re-evaluated after every IK fit.
This necessary force check does not certify contact moments or physical tracking.
"""
from pathlib import Path
import argparse,json,pickle
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from solve_upper_fit_qp import solve
import numpy as np
import torch,osqp
from scipy import sparse
from scipy.interpolate import BSpline
from numpy_leg_kinematics import LegKinematics
from numpy_g1_com import CenterOfMassModel
from bounded_leg_ik import solve_leg_pose
from blended_spline import BlendedSplineReference
from reference_smoothing_basis import runtime_smoothing_basis
from mujoco_reference_dynamics import ReferenceDynamics
from fabrics_sim.reference.interpolation import MotionInterpolator
parser=argparse.ArgumentParser();parser.add_argument('--normal-min',type=float,default=150.);parser.add_argument('--name',default='current_late_wrench_fit');args=parser.parse_args()
p=Path(__file__).parent
parent=np.load('acc_data/amp_walk_20261103/feedback_followup/wbc_convex_catch15_25_clear35.npz')
r=np.load(p/'rolling_contact.control_reference.npz');d=np.load(p/'rolling_contact.npz')
base=parent['control_q'].astype(float);source=r['source_q'];out=base.copy()
urdf='src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf'
leg=LegKinematics(urdf,d['final_joint_order']);com=CenterOfMassModel(urdf,d['final_joint_order']);dynamics=ReferenceDynamics(urdf,d['final_joint_order'])
limits=d['joint_position_limits'][0];plan=json.load(open(p/'rolling_contact.steps_plan.json'));geometry=[(np.array(plan['collision_points'][j]),np.array(plan['collision_radii'][j])) for j in [0,1]]
start,end=840,1170;frames=np.arange(start,end+.01,2.);active=(frames>=890)&(frames<=1110)
contacts=np.asarray(pickle.load(open('acc_data/amp_walk_20261102/amp_walk_native_contacts.pkl','rb'))['contact_sequence']);selected=np.flatnonzero(active);ns=len(selected);nc=None
knots=np.r_[np.full(4,start),np.arange(start+12,end,12),np.full(4,end)];K=len(knots)-4;raw_basis=BSpline(knots,np.eye(K),3)
basis=runtime_smoothing_basis(raw_basis,len(base),6);B=basis(frames);V=basis(frames,1)*240;A=basis(frames,2)*240**2;J=basis(frames,3)*240**3
B3=sparse.kron(B,np.eye(3),format='csc');V3=sparse.kron(V,np.eye(3),format='csc');A3=sparse.kron(A,np.eye(3),format='csc');J3=sparse.kron(J,np.eye(3),format='csc')
nc=3*K
raw_com=np.array([com.evaluate(q) for q in base]);coeff=np.zeros((K,3));history=[]
def curve(q):
 t=torch.tensor(q,dtype=torch.float64)
 return BlendedSplineReference(MotionInterpolator(t),t,stride=6,start_frame=130,ramp_frames=20)
def sample(q,fs):
 sampler=curve(q);poses=[];forces=[]
 for f in fs:
  pose,vel,acc=[x.numpy()[0] for x in sampler.kinematics(float(f),240.)]
  poses.append(pose);forces.append(dynamics.evaluate(pose,vel,acc)['generalized_force'][:6])
 return np.array(poses),np.array(forces)
for iteration in range(6):
 poses,force=sample(out,frames);effective_acc=force[:,:3]/com.mass-[0.,0.,9.81]-A@coeff
 P=2*(100000*(B3.T@B3)+50*(V3.T@V3)+.00002*(J3.T@J3)+A3.T@A3)+sparse.eye(3*K)*1e-6
 P=sparse.block_diag((P,sparse.eye(ns)*2e9),format="csc")
 linear=np.r_[2*np.asarray(A3.T@effective_acc.ravel()).ravel(),np.zeros(ns)];matrices=[];low=[];high=[]
 def add(matrix,lo,hi):
  matrix=sparse.csc_matrix(matrix)
  if matrix.shape[1]==nc:matrix=sparse.hstack((matrix,sparse.csc_matrix((matrix.shape[0],ns))),format="csc")
  matrices.append(matrix);low.extend(np.broadcast_to(lo,(matrix.shape[0],)));high.extend(np.broadcast_to(hi,(matrix.shape[0],)))
 selector=np.zeros((18,3*K));indices=np.r_[np.arange(9),np.arange(3*K-9,3*K)];selector[np.arange(18),indices]=1;add(selector,0.,0.)
 add(B3,np.tile([-.035,-.035,-.045],len(frames)),np.tile([.035,.035,0.],len(frames)))
 for axis in [0,1]:
  for sign in [-1,1]:
   direction=np.array([0.,0.,1.]);direction[axis]=.15*sign;add(sparse.kron(B,direction),-np.inf,0.)
 az=A3[3*np.flatnonzero(active)+2];add(az,args.normal_min/com.mass-9.81-effective_acc[active,2],800/com.mass-9.81-effective_acc[active,2])
 for sx in [-1.,1.]:
  for sy in [-1.,1.]:
   direction=np.array([sx,sy,-.5]);add(sparse.kron(A[active],direction),-np.inf,.5*9.81-effective_acc[active]@direction)
 add(sparse.hstack((sparse.csc_matrix((ns,nc)),sparse.eye(ns))),0.,np.inf)
 add(sparse.eye(nc),coeff.ravel()-.015,coeff.ravel()+.015)
 for node,k in enumerate(selected):
  q=poses[k];c=com.evaluate(q);F=force[k,:3];M=force[k,3:6]+np.cross(q[:3],F);vertices=[]
  for foot in np.flatnonzero(contacts[int(frames[k])]):
   fp,R,_=leg.evaluate(q,foot);vertices.extend((fp+np.array([[x,y,-.035] for x in [-.025,.085] for y in [-.02,.02]])@R.T)[:,:2])
  for nx,ny,b in ConvexHull(vertices).equations:
   b+=.003
   cp=np.array([nx*F[2],ny*F[2],-nx*F[0]-ny*F[1]])
   ca=com.mass*np.array([-nx*c[2],-ny*c[2],nx*c[0]+ny*c[1]+b])
   row=(np.kron(B[k],cp)+np.kron(A[k],ca))/(com.mass*9.81)
   residual=(-nx*M[1]+ny*M[0]+b*F[2])/(com.mass*9.81)
   full=np.r_[row,np.zeros(ns)];full[nc+node]=-1.
   add(full[None],-np.inf,-residual+row@coeff.ravel())
 solution,solver_report=solve(P,linear,sparse.vstack(matrices,format='csc'),np.array(low),np.array(high))
 if not solver_report['accepted']:raise RuntimeError(solver_report)
 coeff=solution[:nc].reshape(K,3);offset=BSpline(knots,coeff,3)(np.arange(len(base)));offset[:start]=0;offset[end+1:]=0
 out=base.copy();errors=[]
 for f in range(start+1,end):
  q=base[f].copy();target=raw_com[f]+offset[f];feet=[leg.evaluate(q,j)[:2] for j in [0,1]]
  for _ in range(15):
   q[:3]+=1.25*(target-com.evaluate(q));peaks=[]
   for j,(pos,R) in enumerate(feet):q,pe,re=solve_leg_pose(leg,q,j,pos,R);peaks.append([pe,re])
   ce=np.linalg.norm(com.evaluate(q)-target)
   if ce<1e-7:break
  out[f]=q;errors.append([ce,*np.max(peaks,axis=0)])
 poses,force=sample(out,frames);err=np.max(errors,axis=0);record=dict(iteration=iteration,solver=solver_report,peak_linearized_cop_slack=float(solution[nc:].max()),projection_errors=err.tolist(),maximum_com_offset=np.max(abs(offset),axis=0).tolist(),minimum_normal_force=float(force[active,2].min()),maximum_normal_force=float(force[active,2].max()),maximum_friction_l1=float((np.sum(abs(force[active,:2]),axis=1)/force[active,2]).max()))
 history.append(record);print(json.dumps(record),flush=True)
 if np.any(err>[.002,.002,.01]):raise RuntimeError('IK did not preserve COM and feet')
fs=np.arange(start-6,end+6.01,.5);dense,forces=sample(out,fs);parent_dense,_=sample(base,fs);clear=[];parent_clear=[];foot_delta=[]
for q,original in zip(dense,parent_dense):
 for j in [0,1]:
  pos,R,_=leg.evaluate(q,j);oldpos,oldR,_=leg.evaluate(original,j);pts,radius=geometry[j];clear.append(pos[2]+min((pts@R.T)[:,2]-radius));parent_clear.append(oldpos[2]+min((pts@oldR.T)[:,2]-radius));foot_delta.append(np.linalg.norm(pos-oldpos))
mask=(fs>=890)&(fs<=1110);friction=np.sum(abs(forces[mask,:2]),axis=1)/forces[mask,2]
report=dict(parent='wbc_convex_catch15_25_clear35',iterations=history,minimum_dense_joint_limit_margin=float(np.minimum(dense[:,7:]-limits[:,0],limits[:,1]-dense[:,7:]).min()),minimum_dense_native_clearance=float(min(clear)),minimum_parent_native_clearance=float(min(parent_clear)),peak_dense_foot_change=float(max(foot_delta)),peak_source_joint_reference_rmse=float(np.sqrt(np.mean((out[:,7:]-source[:,7:])**2,axis=1)).max()),peak_joint_correction_rmse=float(np.sqrt(np.mean((out[:,7:]-base[:,7:])**2,axis=1)).max()),minimum_normal_force=float(forces[mask,2].min()),maximum_friction_l1=float(friction.max()),force_audit_frame_range=[890,1110],physical_tracking_certified=False)
cop_violation=[]
for q,Ffull,f in zip(dense,forces,fs):
 if not 890<=f<=1110:continue
 F=Ffull[:3];M=Ffull[3:6]+np.cross(q[:3],F);vertices=[]
 for foot in np.flatnonzero(contacts[int(f)]):
  fp,R,_=leg.evaluate(q,foot);vertices.extend((fp+np.array([[x,y,-.035] for x in [-.025,.085] for y in [-.02,.02]])@R.T)[:,:2])
 hull=ConvexHull(vertices).equations;cop=np.array([-M[1],M[0]])/max(F[2],1e-9)
 cop_violation.append(float(np.maximum(0,hull[:,:2]@cop+hull[:,2]).max()))
report['maximum_nonlinear_cop_violation_m']=max(cop_violation)
report['preflight_passed']=bool(report['maximum_nonlinear_cop_violation_m']<.005 and report['minimum_dense_joint_limit_margin']>0 and report['minimum_dense_native_clearance']>=min(parent_clear)-.0005 and report['peak_dense_foot_change']<.002 and report['peak_source_joint_reference_rmse']<.19 and report['minimum_normal_force']>0 and report['maximum_friction_l1']<.6)
np.savez_compressed(p/(args.name+'.npz'),control_q=out,fps=200.,baseline_sha256=parent['baseline_sha256'],source_sha256=parent['source_sha256'])
np.savez_compressed(p/(args.name+'.forces.npz'),frame=fs,force=forces)
(p/(args.name+'.json')).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
if not report['preflight_passed']:raise RuntimeError('Reference preflight failed')
