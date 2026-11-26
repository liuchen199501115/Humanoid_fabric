"""Fit a local C2 COM correction subject to unilateral and friction forces.
This centroidal preflight does not certify whole-body or closed-loop feasibility.
"""
from pathlib import Path
import json,numpy as np,torch,osqp
from scipy import sparse
from scipy.interpolate import BSpline,CubicSpline
from numpy_leg_kinematics import LegKinematics
from numpy_g1_com import CenterOfMassModel
from bounded_leg_ik import solve_leg_pose
from blended_spline import BlendedSplineReference
from fabrics_sim.reference.interpolation import MotionInterpolator
p=Path(__file__).parent;data=np.load(p/'turn_recovery_y40.npz');base=data['control_q'];source=np.load(p/'fixed_smooth_early.control_reference.npz')['source_q'];d=np.load(p/'wrench_fixed_ten.npz');urdf='src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf';leg=LegKinematics(urdf,d['final_joint_order']);com=CenterOfMassModel(urdf,d['final_joint_order']);limits=d['joint_position_limits'][0];plan=json.load(open(p/'wrench_fixed_ten.steps_plan.json'));geometry=[(np.array(plan['collision_points'][j]),np.array(plan['collision_radii'][j])) for j in [0,1]]
start,end=700,1080;frames=np.arange(start,end+.01,1.);active=(frames>=740)&(frames<=1000);knots=np.r_[np.full(4,start),np.arange(start+6,end,6),np.full(4,end)];K=len(knots)-4;eye=np.eye(K);spline=BSpline(knots,eye,3);B=spline(frames);V=spline(frames,1)*240;A=spline(frames,2)*240**2;J=spline(frames,3)*240**3
B3=sparse.kron(sparse.csc_matrix(B),sparse.eye(3),format='csc');V3=sparse.kron(sparse.csc_matrix(V),sparse.eye(3),format='csc');A3=sparse.kron(sparse.csc_matrix(A),sparse.eye(3),format='csc');J3=sparse.kron(sparse.csc_matrix(J),sparse.eye(3),format='csc')
raw_com=np.array([com.evaluate(q) for q in base]);basecoef=np.zeros((K,3));out=base.copy();history=[]
def sample(q):
 tensor=torch.tensor(q);curve=BlendedSplineReference(MotionInterpolator(tensor),tensor,stride=6,start_frame=130,ramp_frames=20);values=[curve.kinematics(float(f),240.) for f in frames];states=np.array([com.momentum(x[0].numpy()[0],x[1].numpy()[0],return_state=True) for x in values]);acc=CubicSpline(frames/240,states[:,1])(frames/240,1);return values,states,acc
for iteration in range(5):
 values,states,acc=sample(out);effective_acc=acc-A@basecoef;effective_pos=states[:,0]-B@basecoef
 P=2*(100000*(B3.T@B3)+50*(V3.T@V3)+.00002*(J3.T@J3))+sparse.eye(3*K)*1e-6
 # Prefer small pose offsets and reduce unnecessary force peaks.
 P+=2*(A3.T@A3);linear=2*np.asarray(A3.T@effective_acc.ravel()).ravel()
 matrices=[];low=[];high=[]
 def add(matrix,lo,hi):
  matrix=sparse.csc_matrix(matrix);matrices.append(matrix);low.extend(np.broadcast_to(lo,(matrix.shape[0],)));high.extend(np.broadcast_to(hi,(matrix.shape[0],)))
 selector=np.zeros((18,3*K));indices=np.r_[np.arange(9),np.arange(3*K-9,3*K)];selector[np.arange(18),indices]=1;add(selector,0.,0.)
 add(B3,np.tile([-.065,-.065,-.065],len(frames)),np.tile([.065,.065,0.],len(frames)))
 # Small downward motion retains leg reach when lateral corrections grow.
 for axis in [0,1]:
  for sign in [-1,1]:
   row=np.array([0.,0.,1.]);row[axis]=.15*sign;add(sparse.kron(B,row),-np.inf,0.)
 az=A3[3*np.flatnonzero(active)+2];add(az,120/com.mass-9.81-effective_acc[active,2],850/com.mass-9.81-effective_acc[active,2])
 for sx in [-1.,1.]:
  for sy in [-1.,1.]:
   direction=np.array([sx,sy,-.43]);row=sparse.kron(A[active],direction);rhs=.43*9.81-effective_acc[active]@direction;add(row,-np.inf,rhs)
 C=sparse.vstack(matrices,format='csc');solver=osqp.OSQP();solver.setup(P=sparse.triu(P,format='csc'),q=linear,A=C,l=np.array(low),u=np.array(high),eps_abs=1e-7,eps_rel=1e-7,max_iter=100000,polish=True,verbose=False);solution=solver.solve()
 if solution.x is None or solution.info.status_val not in (1,2):raise RuntimeError('COM force fit failed: '+solution.info.status)
 coefficients=solution.x.reshape(K,3);offset=BSpline(knots,coefficients,3)(np.arange(len(base)));offset[:start]=0;offset[end+1:]=0;out=base.copy();errors=[]
 for f in range(start+1,end):
  q=base[f].astype(float);target=raw_com[f]+offset[f];feet=[leg.evaluate(q,j)[:2] for j in [0,1]]
  for _ in range(12):
   q[:3]+=1.25*(target-com.evaluate(q));peaks=[]
   for j,(pos,R) in enumerate(feet):q,pe,re=solve_leg_pose(leg,q,j,pos,R);peaks.append([pe,re])
   ce=np.linalg.norm(com.evaluate(q)-target)
   if ce<1e-7:break
  out[f]=q;errors.append([ce,*np.max(peaks,axis=0)])
 basecoef=coefficients;values,states,acc=sample(out);forces=com.mass*(acc+[0,0,9.81]);ratio=np.sum(abs(forces[active,:2]),axis=1)/forces[active,2];err=np.max(errors,axis=0)
 record=dict(iteration=iteration,status=solution.info.status,maximum_com_offset=np.max(abs(offset),axis=0).tolist(),projection_errors=err.tolist(),minimum_normal_force=float(forces[active,2].min()),maximum_normal_force=float(forces[active,2].max()),maximum_friction_l1=float(ratio.max()));history.append(record);print(record,flush=True)
 if err[0]>.002 or err[1]>.002 or err[2]>.01:break
# Audit geometry and original-source reference errors separately from forces.
clear=[];dense=np.array([x[0].numpy()[0] for x in values])
for q in dense:
 for j in [0,1]:
  pos,R,_=leg.evaluate(q,j);pts,r=geometry[j];clear.append(pos[2]+min((pts@R.T)[:,2]-r))
margin=float(np.minimum(dense[:,7:]-limits[:,0],limits[:,1]-dense[:,7:]).min());rmse=float(np.sqrt(np.mean((out[:,7:]-source[:,7:])**2,axis=1)).max());report=dict(parent='turn_recovery_y40',iterations=history,peak_com_projection_error=float(err[0]),peak_position_residual=float(err[1]),peak_orientation_residual=float(err[2]),minimum_dense_joint_limit_margin=margin,minimum_dense_native_clearance=float(min(clear)),peak_source_joint_reference_rmse=rmse,physical_tracking_certified=False)
report['preflight_passed']=bool(err[0]<.002 and err[1]<.002 and err[2]<.01 and margin>.005 and rmse<.19 and min(clear)>-.0015)
name='turn_force_feasible_com';np.savez_compressed(p/(name+'.npz'),control_q=out,fps=200.,baseline_sha256=data['baseline_sha256'],source_sha256=data['source_sha256']);np.savez_compressed(p/(name+'.forces.npz'),frames=frames,force=forces,com=states[:,0],velocity=states[:,1],knots=knots,coefficients=coefficients);(p/(name+'.json')).write_text(json.dumps(report,indent=2)+'\n');print(report,flush=True)
