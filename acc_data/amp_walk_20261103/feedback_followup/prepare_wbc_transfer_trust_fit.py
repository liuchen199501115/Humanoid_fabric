"""Offline COM, root orientation and upper-joint reference SQP.
Ground-moment slack and nonlinear reconstruction are reported, never hidden.
"""
from pathlib import Path
import json,pickle,numpy as np,torch,osqp
from scipy import sparse
from scipy.interpolate import BSpline,CubicSpline
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from numpy_leg_kinematics import LegKinematics
from mujoco_centroidal_model import CentroidalModel as CenterOfMassModel
from bounded_leg_ik import solve_leg_pose
from blended_spline import BlendedSplineReference
from fabrics_sim.reference.interpolation import MotionInterpolator
from centroidal_pose_linearization import pose_variations
p=Path(__file__).parent;data=np.load(p/'wbc_transfer_left_landing_forward30.npz');base=data['control_q'];source=np.load(p/'fixed_smooth_early.control_reference.npz')['source_q'];d=np.load(p/'wrench_fixed_ten.npz');urdf='src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf';leg=LegKinematics(urdf,d['final_joint_order']);com=CenterOfMassModel(urdf,d['final_joint_order']);limits=d['joint_position_limits'][0];plan=json.load(open(p/'wrench_fixed_ten.steps_plan.json'));geometry=[(np.array(plan['collision_points'][j]),np.array(plan['collision_radii'][j])) for j in [0,1]];contacts=np.asarray(pickle.load(open('acc_data/amp_walk_20261102/amp_walk_native_contacts.pkl','rb'))['contact_sequence'])
# These are interpolation coefficients, not commands. The executed stride-6
# curve is constrained using the original URDF bounds and audited at its exact
# polynomial extrema. Padding prevents clipping intermediate IK coefficients.
import copy
control_leg=copy.copy(leg);control_leg.lower=leg.lower-.04;control_leg.upper=leg.upper+.04
# This offline plan follows the prepared geometry during the short transfer.
# Runtime support is still gated solely by measured physical contact.
contacts=contacts.copy();geometry_contact_rows=[]
tbase=torch.tensor(base,dtype=torch.float64);base_curve=BlendedSplineReference(MotionInterpolator(tbase),tbase,stride=6,start_frame=130,ramp_frames=20)
for frame in range(800,821):
 pose=base_curve.kinematics(float(frame),240.)[0].numpy()[0];clear=[]
 for side in [0,1]:
  position,rotation,_=leg.evaluate(pose,side);points,radii=geometry[side];clear.append(float(position[2]+np.min((points@rotation.T)[:,2]-radii)))
 contact=np.array(clear)<=.002
 if not contact.any():raise RuntimeError('Prepared transfer has unsupported flight')
 geometry_contact_rows.append(dict(frame=frame,original=contacts[frame].tolist(),prepared=contact.tolist(),native_clearance=clear));contacts[frame]=contact
(p/'wbc_transfer_trust_fit.contact_plan.json').write_text(json.dumps(dict(runtime_contacts_unmodified=True,source_labels_unmodified=True,geometry_transfer=geometry_contact_rows),indent=2)+'\n')
dimension=23; start,end=780,1100;switches=np.flatnonzero(np.any(contacts[1:]!=contacts[:-1],axis=1))+1;switches=switches[(switches>start)&(switches<end)];frames=np.unique(np.r_[np.arange(start,end+.01,.5),switches-1e-4]);times=frames/240.;selected=np.flatnonzero((frames>=800)&(frames<=1060));knots=np.r_[np.full(4,start),np.arange(start+8,end,8),np.full(4,end)];K=len(knots)-4;spline=BSpline(knots,np.eye(K),3)
# Linearize the same stride-6 smoothing operator used by the runtime q/v/a.
from reference_smoothing_basis import runtime_smoothing_basis
effective_spline=runtime_smoothing_basis(spline,len(base),6)
B=effective_spline(frames);V=effective_spline(frames,1)*240;A=effective_spline(frames,2)*240**2;J=effective_spline(frames,3)*240**3;B6=sparse.kron(B,np.eye(dimension),format='csc');V6=sparse.kron(V,np.eye(dimension),format='csc');A6=sparse.kron(A,np.eye(dimension),format='csc');J6=sparse.kron(J,np.eye(dimension),format='csc');nc=dimension*K;ns=len(selected);n=nc+ns
raw_com=np.array([com.evaluate(q) for q in base]);coeff=np.zeros((K,dimension));out=base.copy();history=[];corners=np.array([[-.025,-.02,-.035],[-.025,.02,-.035],[.085,-.02,-.035],[.085,.02,-.035]])
def sample(q):
 t=torch.tensor(q,dtype=torch.float64);curve=BlendedSplineReference(MotionInterpolator(t),t,stride=6,start_frame=130,ramp_frames=20);values=[curve.kinematics(float(f),240.) for f in frames];qs=np.array([x[0].numpy()[0] for x in values]);vs=np.array([x[1].numpy()[0] for x in values]);states=np.array([com.momentum(q,v,return_state=True) for q,v in zip(qs,vs)]);der=CubicSpline(times,states[:,1:])(times,1);return values,qs,vs,states,der[:,0],der[:,1]
checkpoint=np.load(p/'wbc_transfer_curve_fit.iteration_3.npz');coeff=checkpoint['coefficients'];out=checkpoint['control_q'].astype(float);history=[]
for iteration in range(6):
 print('linearizing iteration',iteration,flush=True)
 values,qs,vs,states,acc,hdot=sample(out)
 from runtime_centroidal_linearization import linearize
 tcurve=torch.tensor(out,dtype=torch.float64);curve=BlendedSplineReference(MotionInterpolator(tcurve),tcurve,stride=6,start_frame=130,ramp_frames=20)
 maps=linearize(com,leg,curve,spline,coeff,frames)
 # Cubic interpolation has exponentially decaying global tails; remove only
 # sub-picometre derivative coefficients to preserve sparse support.
 for array in maps.values():array[abs(array)<1e-12]=0.
 print('composed maps ready',flush=True)
 forces=com.mass*(acc+[0,0,9.81]);effective_acc=acc-maps['acceleration']@coeff.ravel()
 W=sparse.diags(np.tile([100000.]*3+[10000.]*3+[1000.]*17,len(frames)));WV=sparse.diags(np.tile([50.]*3+[5.]*3+[.5]*17,len(frames)))
 H=2*(B6.T@W@B6+V6.T@WV@V6+.00002*(J6.T@J6))+sparse.eye(nc)*1e-5;H=sparse.block_diag((H,sparse.eye(ns)*2e8),format='csc');linear=np.zeros(n);matrices=[];low=[];high=[]
 def add(matrix,lo,hi,slack=None):
  matrix=sparse.csc_matrix(matrix)
  if matrix.shape[1]==nc:matrix=sparse.hstack((matrix,sparse.csc_matrix((matrix.shape[0],ns))),format='csc')
  matrices.append(matrix);low.extend(np.broadcast_to(lo,(matrix.shape[0],)));high.extend(np.broadcast_to(hi,(matrix.shape[0],)))
 selector=np.zeros((6*dimension,nc));indices=np.r_[np.arange(3*dimension),np.arange(nc-3*dimension,nc)];selector[np.arange(6*dimension),indices]=1;add(selector,0.,0.)
 add(B6,np.tile([-.065]*3+[-.15]*3+[-.22]*17,len(frames)),np.tile([.065,.065,0.]+[.15]*3+[.22]*17,len(frames)))
 # Trust the local orientation map; COM force corrections remain unconstrained within the global tube.
 trust=np.tile([.005]*3+[.01]*3+[.015]*17,K);add(sparse.eye(nc),coeff.ravel()-trust,coeff.ravel()+trust)
 add(sparse.hstack((sparse.csc_matrix((ns,nc)),sparse.eye(ns))),0.,np.inf)
 for axis in [0,1]:
  for sign in [-1,1]:
   direction=np.zeros(dimension);direction[2]=1;direction[axis]=.15*sign;add(sparse.kron(B,direction),-np.inf,0.)
 # Bound the actual unsmoothed leg change, not only COM coordinates. Near
 # straight knees the IK sensitivity can amplify a small COM step by 10x.
 from runtime_centroidal_linearization import pose_map
 raw_nodes=np.array([np.mean(curve.smooth.t[i+1:i+4]) for i in range(len(curve.smooth.c))])
 for raw_frame in raw_nodes[(raw_nodes>start)&(raw_nodes<end)]:
  raw_q=out[int(round(raw_frame))].astype(float);raw_B=spline(raw_frame)
  raw_T=pose_map(com,leg,raw_q,raw_B@coeff[:,3:6])
  raw_J=np.kron(raw_B,raw_T[6:18]);previous=raw_J@coeff.ravel()
  add(raw_J,previous-.025,previous+.025)
 for k in range(len(frames)):
  position=maps['joints'][k];current=qs[k,7:];previous=position@coeff.ravel()
  lower=limits[:,0]+.010
  if 800<=frames[k]<=1060:
   lower=lower.copy();lower[[3,9]]=np.maximum(lower[[3,9]],.25)
  add(position,lower-current+previous,limits[:,1]-.010-current+previous)
 for node,k in enumerate(selected):
  add(maps['acceleration'][k,2:3],140/com.mass-9.81-effective_acc[k,2],780/com.mass-9.81-effective_acc[k,2])
  for sx in [-1.,1.]:
   for sy in [-1.,1.]:
    direction=np.array([sx,sy,-.5]);add((direction@maps['acceleration'][k])[None],-np.inf,.5*9.81-effective_acc[k]@direction)
  moment=maps['momentum_rate'][k];vertices=[]
  for side in [0,1]:
   if contacts[int(frames[k]),side]:
    pos,R,_=leg.evaluate(qs[k],side);vertices.extend((pos+corners@R.T)[:,:2])
  if len(vertices)<3:continue
  c=states[k,0];force=forces[k];Rxy=c[:2]*force[2]-c[2]*force[:2]+np.array([-hdot[k,1],hdot[k,0]])
  for nx,ny,offset in ConvexHull(vertices).equations:
   offset+=.003
   cp=np.array([nx*force[2],ny*force[2],-nx*force[0]-ny*force[1]]);ca=com.mass*np.array([-nx*c[2],-ny*c[2],nx*c[0]+ny*c[1]+offset]);row=(cp@maps['com'][k]+ca@maps['acceleration'][k]+np.array([ny,-nx,0.])@moment)/(com.mass*9.81);residual=(np.array([nx,ny])@Rxy+offset*force[2])/(com.mass*9.81);full=np.r_[row,np.zeros(ns)];full[nc+node]=-1;add(full[None],-np.inf,-residual+row@coeff.ravel())
 C=sparse.vstack(matrices,format='csc');row_scale=np.maximum(np.sqrt(C.multiply(C).sum(axis=1)).A.ravel(),1e-8);objective_scale=max(float(H.diagonal().max()),1.)
 np.savez_compressed(p/'wbc_transfer_trust_fit.problem.npz',H_data=H.data,H_indices=H.indices,H_indptr=H.indptr,H_shape=H.shape,C_data=C.data,C_indices=C.indices,C_indptr=C.indptr,C_shape=C.shape,low=low,high=high,linear=linear,row_scale=row_scale,objective_scale=objective_scale)
 from solve_upper_fit_qp import solve
 print('solving',C.shape,'nonzeros',C.nnz,flush=True)
 solution,solver_report=solve(H,linear,C,np.array(low),np.array(high))
 print('iteration',iteration,'solver',solver_report,flush=True)
 if not solver_report['accepted']:
  (p/'wbc_transfer_trust_fit.json').write_text(json.dumps(dict(preflight_passed=False,physical_tracking_certified=False,solver=solver_report,iterations=history),indent=2)+'\n')
  raise RuntimeError('Centroidal pose fit failed exact constraint checks')
 newcoeff=solution[:nc].reshape(K,dimension);offsets=BSpline(knots,newcoeff,3)(np.arange(len(base)));offsets[:start]=0;offsets[end+1:]=0;out=base.astype(float).copy();errors=[]
 for f in range(start+1,end):
  q=base[f].astype(float);target=raw_com[f]+offsets[f,:3];feet=[leg.evaluate(q,j)[:2] for j in [0,1]];q[3:7]=(Rotation.from_rotvec(offsets[f,3:6])*Rotation.from_quat(q[3:7])).as_quat()
  q[19:]+=offsets[f,6:]
  for _ in range(20):
   q[:3]+=1.25*(target-com.evaluate(q));peaks=[]
   for j,(pos,R) in enumerate(feet):q,pe,re=solve_leg_pose(control_leg,q,j,pos,R);peaks.append([pe,re])
   ce=np.linalg.norm(com.evaluate(q)-target)
   if ce<1e-7:break
  out[f]=q;errors.append([ce,*np.max(peaks,axis=0)])
 coeff=newcoeff
 np.savez_compressed(p/f'wbc_transfer_trust_fit.iteration_{iteration}.npz',control_q=out,knots=knots,coefficients=coeff,fps=200.,baseline_sha256=data['baseline_sha256'],source_sha256=data['source_sha256'])
 err=np.max(errors,axis=0);record=dict(iteration=iteration,status=solver_report['status'],peak_linearized_cop_slack_m=float(solution[nc:].max()),maximum_coordinate_offset=np.max(abs(offsets),axis=0).tolist(),projection_errors=err.tolist());history.append(record);print(record,flush=True)
 if err[0]>.002 or err[1]>.002 or err[2]>.01:break
values,qs,vs,states,acc,hdot=sample(out);clear=[]
for q in qs:
 for side in [0,1]:
  pos,R,_=leg.evaluate(q,side);pts,r=geometry[side];clear.append(pos[2]+min((pts@R.T)[:,2]-r))
margin=float(np.minimum(qs[:,7:]-limits[:,0],limits[:,1]-qs[:,7:]).min());rmse=float(np.sqrt(np.mean((out[:,7:]-source[:,7:])**2,axis=1)).max());report=dict(parent='wbc_transfer_left_landing_forward30',iterations=history,peak_com_projection_error=float(err[0]),peak_position_residual=float(err[1]),peak_orientation_residual=float(err[2]),minimum_dense_joint_limit_margin=margin,minimum_dense_native_clearance=float(min(clear)),peak_source_joint_reference_rmse=rmse,physical_tracking_certified=False);report['kinematic_preflight_passed']=bool(err[0]<.002 and err[1]<.002 and err[2]<.01 and margin>0 and rmse<.19 and min(clear)>-.002 and history[-1]['peak_linearized_cop_slack_m']<.002)
report['raw_leg_trust_rad']=.025;report['linearization']='raw fixed-foot IK composed with executed normalized spline';report['warm_start']='wbc_transfer_curve_fit.iteration_3';report['control_point_bound_padding_rad']=.04;report['minimum_raw_control_point_margin']=float(np.minimum(out[:,7:]-limits[:,0],limits[:,1]-out[:,7:]).min());report['requires_exact_curve_joint_audit']=True
report['preflight_passed']=False;report['requires_nonlinear_dynamic_audit']=True
name='wbc_transfer_trust_fit';np.savez_compressed(p/(name+'.npz'),control_q=out,fps=200.,baseline_sha256=data['baseline_sha256'],source_sha256=data['source_sha256']);np.savez_compressed(p/(name+'.centroidal.npz'),frames=frames,planning_contacts=contacts,com=states[:,0],force=com.mass*(acc+[0,0,9.81]),momentum_rate=hdot,knots=knots,coefficients=coeff);(p/(name+'.json')).write_text(json.dumps(report,indent=2)+'\n');print(report,flush=True)

if not report['kinematic_preflight_passed']:raise RuntimeError('Offline full-body guide did not pass preflight; not approved for simulation')
