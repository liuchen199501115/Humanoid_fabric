"""Local six-coordinate reference SQP for feasible net force and ground moment.
Ground-moment slack and nonlinear reconstruction are reported, never hidden.
"""
from pathlib import Path
import json,pickle,numpy as np,torch,osqp
from scipy import sparse
from scipy.interpolate import BSpline,CubicSpline
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from numpy_leg_kinematics import LegKinematics
from numpy_g1_com import CenterOfMassModel
from bounded_leg_ik import solve_leg_pose
from blended_spline import BlendedSplineReference
from fabrics_sim.reference.interpolation import MotionInterpolator
from centroidal_pose_linearization import pose_variations
p=Path(__file__).parent;data=np.load(p/'turn_force_feasible_com_floor.npz');base=data['control_q'];source=np.load(p/'fixed_smooth_early.control_reference.npz')['source_q'];d=np.load(p/'wrench_fixed_ten.npz');urdf='src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf';leg=LegKinematics(urdf,d['final_joint_order']);com=CenterOfMassModel(urdf,d['final_joint_order']);limits=d['joint_position_limits'][0];plan=json.load(open(p/'wrench_fixed_ten.steps_plan.json'));geometry=[(np.array(plan['collision_points'][j]),np.array(plan['collision_radii'][j])) for j in [0,1]];contacts=np.asarray(pickle.load(open('acc_data/amp_walk_20261102/amp_walk_native_contacts.pkl','rb'))['contact_sequence'])
start,end=700,1080;frames=np.arange(start,end+.01,2.);times=frames/240.;selected=np.flatnonzero((frames>=740)&(frames<=1000));knots=np.r_[np.full(4,start),np.arange(start+6,end,6),np.full(4,end)];K=len(knots)-4;spline=BSpline(knots,np.eye(K),3);B=spline(frames);V=spline(frames,1)*240;A=spline(frames,2)*240**2;J=spline(frames,3)*240**3;B6=sparse.kron(B,np.eye(6),format='csc');V6=sparse.kron(V,np.eye(6),format='csc');A6=sparse.kron(A,np.eye(6),format='csc');J6=sparse.kron(J,np.eye(6),format='csc');nc=6*K;ns=len(selected);n=nc+ns
raw_com=np.array([com.evaluate(q) for q in base]);coeff=np.zeros((K,6));out=base.copy();history=[];corners=np.array([[-.05,-.025,-.035],[-.05,.025,-.035],[.12,-.025,-.035],[.12,.025,-.035]])
def sample(q):
 t=torch.tensor(q);curve=BlendedSplineReference(MotionInterpolator(t),t,stride=6,start_frame=130,ramp_frames=20);values=[curve.kinematics(float(f),240.) for f in frames];qs=np.array([x[0].numpy()[0] for x in values]);vs=np.array([x[1].numpy()[0] for x in values]);states=np.array([com.momentum(q,v,return_state=True) for q,v in zip(qs,vs)]);der=CubicSpline(times,states[:,1:])(times,1);return values,qs,vs,states,der[:,0],der[:,1]
for iteration in range(4):
 values,qs,vs,states,acc,hdot=sample(out);T,Td,HP,HQ,HPd,HQd=pose_variations(com,leg,qs,vs,times);forces=com.mass*(acc+[0,0,9.81]);effective_acc=acc-A@coeff[:,:3]
 W=sparse.diags(np.tile([100000.]*3+[10000.]*3,len(frames)));WV=sparse.diags(np.tile([50.]*3+[5.]*3,len(frames)))
 H=2*(B6.T@W@B6+V6.T@WV@V6+.00002*(J6.T@J6))+sparse.eye(nc)*1e-5;H=sparse.block_diag((H,sparse.eye(ns)*2e6),format='csc');linear=np.zeros(n);matrices=[];low=[];high=[]
 def add(matrix,lo,hi,slack=None):
  matrix=sparse.csc_matrix(matrix)
  if matrix.shape[1]==nc:matrix=sparse.hstack((matrix,sparse.csc_matrix((matrix.shape[0],ns))),format='csc')
  matrices.append(matrix);low.extend(np.broadcast_to(lo,(matrix.shape[0],)));high.extend(np.broadcast_to(hi,(matrix.shape[0],)))
 selector=np.zeros((36,nc));indices=np.r_[np.arange(18),np.arange(nc-18,nc)];selector[np.arange(36),indices]=1;add(selector,0.,0.)
 add(B6,np.tile([-.065]*3+[-.18]*3,len(frames)),np.tile([.065,.065,0.]+[.18]*3,len(frames)))
 # Trust the local orientation map; COM force corrections remain unconstrained within the global tube.
 trust=np.tile([.065]*3+[.06]*3,K);add(sparse.eye(nc),coeff.ravel()-trust,coeff.ravel()+trust)
 add(sparse.hstack((sparse.csc_matrix((ns,nc)),sparse.eye(ns))),0.,np.inf)
 for axis in [0,1]:
  for sign in [-1,1]:
   direction=np.zeros(6);direction[2]=1;direction[axis]=.15*sign;add(sparse.kron(B,direction),-np.inf,0.)
 for k in range(len(frames)):
  position=np.kron(B[k],T[k,6:18]);current=qs[k,7:19];previous=position@coeff.ravel();add(position,limits[:12,0]+.003-current+previous,limits[:12,1]-.003-current+previous)
 for node,k in enumerate(selected):
  direction=np.array([0.,0.,1.,0.,0.,0.]);add(np.kron(A[k:k+1],direction),120/com.mass-9.81-effective_acc[k,2],850/com.mass-9.81-effective_acc[k,2])
  for sx in [-1.,1.]:
   for sy in [-1.,1.]:
    direction=np.array([sx,sy,-.34,0.,0.,0.]);add(np.kron(A[k:k+1],direction),-np.inf,.34*9.81-effective_acc[k]@direction[:3])
  moment=np.kron(B[k],HPd[k])+np.kron(V[k],HP[k]+HQd[k])+np.kron(A[k],HQ[k]);vertices=[]
  for side in [0,1]:
   if contacts[int(frames[k]),side]:
    pos,R,_=leg.evaluate(qs[k],side);vertices.extend((pos+corners@R.T)[:,:2])
  if len(vertices)<3:continue
  c=states[k,0];force=forces[k];Rxy=c[:2]*force[2]-c[2]*force[:2]+np.array([-hdot[k,1],hdot[k,0]])
  for nx,ny,offset in ConvexHull(vertices).equations:
   offset+=.003
   cp=np.array([nx*force[2],ny*force[2],-nx*force[0]-ny*force[1],0.,0.,0.]);ca=com.mass*np.array([-nx*c[2],-ny*c[2],nx*c[0]+ny*c[1]+offset,0.,0.,0.]);row=(np.kron(B[k],cp)+np.kron(A[k],ca)+np.array([ny,-nx,0.])@moment)/(com.mass*9.81);residual=(np.array([nx,ny])@Rxy+offset*force[2])/(com.mass*9.81);full=np.r_[row,np.zeros(ns)];full[nc+node]=-1;add(full[None],-np.inf,-residual+row@coeff.ravel())
 C=sparse.vstack(matrices,format='csc');solver=osqp.OSQP();solver.setup(P=sparse.triu(H,format='csc'),q=linear,A=C,l=np.array(low),u=np.array(high),eps_abs=1e-7,eps_rel=1e-7,max_iter=150000,polish=True,verbose=False);sol=solver.solve()
 if sol.x is None or sol.info.status_val not in (1,2):raise RuntimeError('Centroidal pose fit failed: '+sol.info.status)
 newcoeff=sol.x[:nc].reshape(K,6);offsets=BSpline(knots,newcoeff,3)(np.arange(len(base)));offsets[:start]=0;offsets[end+1:]=0;out=base.copy();errors=[]
 for f in range(start+1,end):
  q=base[f].astype(float);target=raw_com[f]+offsets[f,:3];feet=[leg.evaluate(q,j)[:2] for j in [0,1]];q[3:7]=(Rotation.from_rotvec(offsets[f,3:])*Rotation.from_quat(q[3:7])).as_quat()
  for _ in range(12):
   q[:3]+=1.25*(target-com.evaluate(q));peaks=[]
   for j,(pos,R) in enumerate(feet):q,pe,re=solve_leg_pose(leg,q,j,pos,R);peaks.append([pe,re])
   ce=np.linalg.norm(com.evaluate(q)-target)
   if ce<1e-7:break
  out[f]=q;errors.append([ce,*np.max(peaks,axis=0)])
 coeff=newcoeff;err=np.max(errors,axis=0);record=dict(iteration=iteration,status=sol.info.status,peak_linearized_cop_slack_m=float(sol.x[nc:].max()),maximum_coordinate_offset=np.max(abs(offsets),axis=0).tolist(),projection_errors=err.tolist());history.append(record);print(record,flush=True)
 if err[0]>.002 or err[1]>.002 or err[2]>.01:break
values,qs,vs,states,acc,hdot=sample(out);clear=[]
for q in qs:
 for side in [0,1]:
  pos,R,_=leg.evaluate(q,side);pts,r=geometry[side];clear.append(pos[2]+min((pts@R.T)[:,2]-r))
margin=float(np.minimum(qs[:,7:]-limits[:,0],limits[:,1]-qs[:,7:]).min());rmse=float(np.sqrt(np.mean((out[:,7:]-source[:,7:])**2,axis=1)).max());report=dict(parent='turn_force_feasible_com_floor',iterations=history,peak_com_projection_error=float(err[0]),peak_position_residual=float(err[1]),peak_orientation_residual=float(err[2]),minimum_dense_joint_limit_margin=margin,minimum_dense_native_clearance=float(min(clear)),peak_source_joint_reference_rmse=rmse,physical_tracking_certified=False);report['preflight_passed']=bool(err[0]<.002 and err[1]<.002 and err[2]<.01 and margin>0 and rmse<.19 and min(clear)>-.0015)
name='turn_centroidal_pose_fit';np.savez_compressed(p/(name+'.npz'),control_q=out,fps=200.,baseline_sha256=data['baseline_sha256'],source_sha256=data['source_sha256']);np.savez_compressed(p/(name+'.centroidal.npz'),frames=frames,com=states[:,0],force=com.mass*(acc+[0,0,9.81]),momentum_rate=hdot,knots=knots,coefficients=coeff);(p/(name+'.json')).write_text(json.dumps(report,indent=2)+'\n');print(report,flush=True)
