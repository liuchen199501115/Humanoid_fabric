"""Offline local reference fit. Passing nominal full-sole dynamics is not tracking.
Root orientation is fixed; root translation and 29 joints may move while both
foot paths and original-source joint fidelity remain constrained.
"""
from pathlib import Path
import argparse, json, pickle
import numpy as np
import torch
from scipy import sparse
from scipy.interpolate import BSpline
from scipy.spatial.transform import Rotation
import clarabel
import warp as wp
from fabrics_sim.fabrics.g1_pose_fabric import G1PoseFabric
from blended_spline import BlendedSplineReference
from reference_smoothing_basis import runtime_smoothing_basis
from mujoco_reference_dynamics import ReferenceDynamics
from full_effort_linearization import effort_variations, inverse_output, perturb
from g1_model_effort import model_effort_limits
from fabrics_sim.reference.interpolation import MotionInterpolator
from fabrics_sim.solver.AccelerationQP import build_single_support_wrench_constraints
from reference_joint_extrema import joint_extrema

parser=argparse.ArgumentParser()
parser.add_argument('--name',required=True)
parser.add_argument('--initial-fit')
parser.add_argument('--iterations',type=int,default=12)
parser.add_argument('--effort-fraction',type=float,default=1.)
parser.add_argument('--trust',type=float,default=.02)
parser.add_argument('--parent',default='acc_data/g1_full_tracking_20261106/step13_leg_smooth5_clearance_audit.npz')
args=parser.parse_args()
assert 1<=args.iterations<=20 and .8<=args.effort_fraction<=1 and 0<args.trust<=.04
p=Path(__file__).parent
if (p/(args.name+'.json')).exists():raise RuntimeError('Existing report must not be overwritten')
parent_path=Path(args.parent)
parent=np.load(parent_path);base=parent['control_q'].astype(float);out=base.copy()
record=np.load(p/'rolling_contact.npz');source=np.load(p/'rolling_contact.control_reference.npz')['source_q']
names=record['final_joint_order'];limits=record['joint_position_limits'][0]
urdf='src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf'
model=ReferenceDynamics(urdf,names)
effort=args.effort_fraction*model_effort_limits(urdf,names,record['effort_limits'].reshape(-1))
speed=record['history_joint_velocity_limits'][-1].reshape(-1)
contacts=np.asarray(pickle.load(open('acc_data/amp_walk_20261102/amp_walk_native_contacts.pkl','rb'))['contact_sequence'])
start,end=924,1092;frames=np.unique(np.r_[np.arange(start,end+.01,.5),1013.-1e-5]);times=frames/240.;U=32
indices=np.flatnonzero((frames>=960)&(frames<=1035));ns=len(indices)
assert contacts[frames[indices].astype(int),1].all()
knots=np.r_[np.full(4,start),np.arange(start+12,end,12),np.full(4,end)];K=len(knots)-4
raw_basis=BSpline(knots,np.eye(K),3);basis=runtime_smoothing_basis(raw_basis,len(base),6)
B=basis(frames);V=basis(frames,1)*240;A=basis(frames,2)*240**2;J=basis(frames,3)*240**3
BM=sparse.kron(B,np.eye(U),format='csc');VM=sparse.kron(V,np.eye(U),format='csc');AM=sparse.kron(A,np.eye(U),format='csc');JM=sparse.kron(J,np.eye(U),format='csc')
nc=K*U;coeff=np.zeros((K,U));history=[]
double_nodes=[node for node,k in enumerate(indices) if contacts[int(frames[k])].all()]
left_slices={node:slice(nc+ns+6*i,nc+ns+6*(i+1)) for i,node in enumerate(double_nodes)}
nf=6*len(double_nodes);ng=len(frames);total=nc+ns+nf+ng;x=np.zeros(total)
wp.config.kernel_cache_dir="/tmp/g1_warp_cache";wp.init()
fabric=G1PoseFabric(1,"cpu",robot_name="g1_29dof_rev_1_0",task_profile="dynamics-aware")
collision=fabric.dynamics_tasks;collision.enable_wrist_hip_avoidance(include_upper_body=True,convex=True)
position_columns=np.r_[0:3,7:36];velocity_columns=np.r_[0:3,6:35]
T=np.zeros((len(frames),35,U));T[:,velocity_columns,np.arange(U)]=1;Td=np.zeros_like(T)
if args.initial_fit:
 initial=np.load(args.initial_fit);c=initial['coefficients']
 if c.shape==(K,17):coeff[:,15:]=c
 elif c.shape==(K,U):coeff=c
 else:raise ValueError('Initial coefficient dimensions')
 offset=raw_basis(np.arange(len(base)))@coeff;offset[:start]=0;offset[end+1:]=0
 out[:,position_columns]+=offset
 if not np.allclose(out,initial['rejected_or_unexecuted_control_q'],atol=1e-10,rtol=0):raise ValueError('Initial fit parent differs')
 if 'left_wrenches' in initial:
  for node,w in zip(double_nodes,initial['left_wrenches']):x[left_slices[node]]=w

def sampler(trajectory):
 tensor=torch.tensor(trajectory,dtype=torch.float64)
 return BlendedSplineReference(MotionInterpolator(tensor),tensor,stride=6,start_frame=130,ramp_frames=20)
def sample(trajectory,fs):
 curve=sampler(trajectory);data=[curve.kinematics(float(f),240.) for f in fs]
 return tuple(np.array([v[j].numpy()[0] for v in data]) for j in range(3))
base_q,_,_=sample(base,frames);target_feet=np.array([model.evaluate(q,np.zeros(35))['feet'] for q in base_q])
source_joint=np.stack([np.interp(frames,np.arange(len(source)),source[:,j]) for j in range(7,36)],axis=1)

def wrench_rows(q,foot):
 state=model.evaluate(q,np.zeros(35));R=Rotation.from_quat(state['feet'][:,3:]).as_matrix()
 G,b=build_single_support_wrench_constraints(contact_mode=torch.tensor([foot+1]),foot_rotation_world=torch.tensor(R[None]),friction_coefficient=.6,coupled_yaw=True,toe_length=.085,heel_length=.025,half_width=.02,sole_depth=.035,minimum_normal_force=20. if foot==0 else 100.,maximum_normal_force=800.)
 G=G.numpy()[0,:,6*foot:6*foot+6];b=b.numpy()[0];active=np.any(abs(G)>0,axis=1);G,b=G[active],b[active]
 scale=np.linalg.norm(G*np.array([350.,350.,350.,35.,35.,35.]),axis=1)
 return G/scale[:,None],b/scale

def left_response(state):
 L,R=state['jacobian'];mapping=np.linalg.solve(R[:,:6].T,L[:,:6].T)
 return np.vstack((-mapping,R[:,6:].T@mapping-L[:,6:].T))

def evaluate(q,v,a,node,solution):
 state=model.evaluate(q,v,a);value=inverse_output(state,1);left=np.zeros(6)
 if node in left_slices:left=solution[left_slices[node]];value+=left_response(state)@left
 G,b=wrench_rows(q,1);violation=float(np.maximum(G@value[:6]-b,0).max())
 if node in left_slices:
  G,b=wrench_rows(q,0);violation=max(violation,float(np.maximum(G@left-b,0).max()))
 reconstructed=state['jacobian'][1].T@value[:6]+state['jacobian'][0].T@left+np.r_[np.zeros(6),value[6:]]
 return value,violation,float(abs(reconstructed-state['generalized_force']).max())

def solve_soc(H,linear,C,lo,hi,soc):
 row_scale=np.maximum(np.sqrt(C.multiply(C).sum(axis=1)).A.ravel(),1e-8)
 Cs=sparse.diags(1/row_scale)@C;lo=lo/row_scale;hi=hi/row_scale
 eq=np.isfinite(lo)&np.isfinite(hi)&(lo==hi);upper=np.isfinite(hi)&~eq;lower=np.isfinite(lo)&~eq
 matrices=[Cs.tocsr()[eq],Cs.tocsr()[upper],-Cs.tocsr()[lower]];rhs=[hi[eq],hi[upper],-lo[lower]]
 cones=[clarabel.ZeroConeT(int(eq.sum())),clarabel.NonnegativeConeT(int(upper.sum()+lower.sum()))]
 for matrix,constant,radius in soc:
  matrices.append(sparse.vstack((sparse.csc_matrix((1,total)),-matrix),format='csc'));rhs.append(np.r_[radius,constant]);cones.append(clarabel.SecondOrderConeT(len(constant)+1))
 settings=clarabel.DefaultSettings();settings.verbose=False;settings.max_threads=1;settings.max_iter=200;settings.tol_feas=1e-9;settings.tol_gap_abs=1e-9;settings.tol_gap_rel=1e-9
 scale=max(H.diagonal().max(),1.)
 result=clarabel.DefaultSolver(sparse.triu(H/scale,format='csc'),linear/scale,sparse.vstack(matrices,format='csc'),np.concatenate(rhs),cones,settings).solve()
 value=np.asarray(result.x);linear_error=max(0.,float((lo-Cs@value).max()),float((Cs@value-hi).max()))
 cone_error=max(0.,max(float(np.linalg.norm(matrix@value+constant)-radius) for matrix,constant,radius in soc))
 report=dict(status=str(result.status),linear_residual=linear_error,cone_residual=cone_error,iterations=result.iterations)
 if str(result.status)!='Solved' or max(linear_error,cone_error)>1e-6:raise RuntimeError(report)
 return value,report

for iteration in range(args.iterations):
 qs,vs,acc=sample(out,frames);maps=effort_variations(model,qs,vs,acc,T,Td,times,indices,[1]*ns)
 weights=np.r_[np.full(3,1e5),np.full(29,2000.)]
 weighted=sparse.kron(sparse.eye(len(frames)),sparse.diags(weights),format='csc')
 H=2*(BM.T@weighted@BM+2*VM.T@VM+.00001*JM.T@JM+.01*AM.T@AM)+sparse.eye(nc)*1e-6
 H=sparse.block_diag((H,sparse.eye(ns)*2e8,sparse.eye(nf)*1e-6,sparse.eye(ng)*2e8),format='csc');linear=np.zeros(total)
 desired=base_q[:,position_columns].copy();desired[:,3:]=source_joint
 error=qs[:,position_columns]-B@coeff-desired;linear[:nc]=2*np.asarray(BM.T@weighted@error.ravel()).ravel()
 mats=[];lows=[];highs=[];soc=[]
 def padded(matrix):
  matrix=sparse.csc_matrix(matrix)
  return sparse.hstack((matrix,sparse.csc_matrix((matrix.shape[0],total-matrix.shape[1]))),format='csc') if matrix.shape[1]<total else matrix
 def add(matrix,lo,hi):
  matrix=padded(matrix);mats.append(matrix);lows.extend(np.broadcast_to(lo,(matrix.shape[0],)));highs.extend(np.broadcast_to(hi,(matrix.shape[0],)))
 endpoint=np.r_[np.arange(3*U),np.arange(nc-3*U,nc)];add(sparse.eye(nc,format='csc')[endpoint],0.,0.)
 box=np.r_[.025,.025,.02,np.full(12,.15),np.full(17,.25)]
 add(BM,np.tile(-box,len(frames)),np.tile(box,len(frames)))
 trust=np.r_[np.full(3,min(args.trust,.005)),np.full(29,args.trust)]
 add(sparse.eye(nc),coeff.ravel()-np.tile(trust,K),coeff.ravel()+np.tile(trust,K))
 rows=np.array([U*k+j for k in range(len(frames)) for j in range(3,U)])
 joint_B=BM[rows];joint_V=VM[rows]
 add(joint_B,(limits[:,0]+.01-qs[:,7:]+(B@coeff)[:,3:]).ravel(),(limits[:,1]-.01-qs[:,7:]+(B@coeff)[:,3:]).ravel())
 add(joint_V,(-speed-vs[:,6:]+(V@coeff)[:,3:]).ravel(),(speed-vs[:,6:]+(V@coeff)[:,3:]).ravel())
 add(sparse.hstack((sparse.csc_matrix((ns,nc)),sparse.eye(ns))),0.,np.inf)
 add(sparse.hstack((sparse.csc_matrix((ng,nc+ns+nf)),sparse.eye(ng))),0.,np.inf)
 for k in range(len(frames)):
  collision_state=collision.model.snapshot(torch.tensor(qs[k:k+1],dtype=torch.float32))
  distance,jacobian=collision.collision_geometry(collision_state)
  distance=distance[0].detach().numpy();jacobian=jacobian[0].detach().numpy()
  selected=distance<.10;M=np.kron(B[k],jacobian[selected][:,velocity_columns])
  collision_rows=np.zeros((len(M),total));collision_rows[:,:nc]=-M/.1;collision_rows[:,nc+ns+nf+k]=-1
  add(collision_rows,-np.inf,(distance[selected]-.02-M@coeff.ravel())/.1)
  soc.append((padded(joint_B[29*k:29*(k+1)]),qs[k,7:]-(B[k]@coeff)[3:]-source_joint[k],.18*np.sqrt(29)))
  state=model.evaluate(qs[k],vs[k])
  for foot in (0,1):
   pose=state['feet'][foot];target=target_feet[k,foot]
   delta=np.r_[pose[:3]-target[:3],(Rotation.from_quat(pose[3:])*Rotation.from_quat(target[3:]).inv()).as_rotvec()]
   mapping=np.kron(B[k],state['jacobian'][foot][:,velocity_columns]);constant=delta-mapping@coeff.ravel()
   tolerance=np.r_[np.full(3,.0005),np.full(3,.005)]
   add(mapping,-tolerance-constant,tolerance-constant)
 # Raw integer reference also stays inside the original-source posture budget.
 for f in range(start,end+1):
  mapping=np.kron(raw_basis(f),np.c_[np.zeros((29,3)),np.eye(29)])
  soc.append((padded(mapping),base[f,7:]-source[f,7:],.18*np.sqrt(29)))
 for node,k in enumerate(indices):
  raw,P,W,Q=maps[k]
  state=model.evaluate(qs[k],vs[k],acc[k]);response=left_response(state)
  if node in left_slices:
   left=x[left_slices[node]]
   for j in range(U):
    plus=model.evaluate(perturb(qs[k],1e-5*T[k,:,j]),vs[k]);minus=model.evaluate(perturb(qs[k],-1e-5*T[k,:,j]),vs[k])
    P[:,j]+=(left_response(plus)-left_response(minus))@left/(2e-5)
  mapping=np.kron(B[k],P)+np.kron(V[k],W)+np.kron(A[k],Q);constant=raw-mapping@coeff.ravel();G,b=wrench_rows(qs[k],1)
  D=np.zeros((len(G)+58,35));D[:len(G),:6]=G;D[len(G):len(G)+29,6:]=np.diag(1/effort);D[len(G)+29:,6:]=-np.diag(1/effort)
  rows=np.c_[D@mapping,np.zeros((len(D),ns+nf+ng))];rows[:,nc+node]=-1
  if node in left_slices:
   rows[:,left_slices[node]]=D@response;L,c=wrench_rows(qs[k],0);left_rows=np.zeros((len(L),total));left_rows[:,left_slices[node]]=L;left_rows[:,nc+node]=-1;add(left_rows,-np.inf,c)
  add(rows,-np.inf,np.r_[b,np.ones(58)]-D@constant)
 x,status=solve_soc(H,linear,sparse.vstack(mats,format='csc'),np.array(lows),np.array(highs),soc)
 coeff=x[:nc].reshape(K,U);offset=raw_basis(np.arange(len(base)))@coeff;offset[:start]=0;offset[end+1:]=0
 out=base.copy();out[:,position_columns]+=offset
 qs,vs,acc=sample(out,frames);details=[]
 for node,k in enumerate(indices):
  value,wrench_error,equality_error=evaluate(qs[k],vs[k],acc[k],node,x);ratios=abs(value[6:])/effort;j=int(np.argmax(ratios))
  if equality_error>1e-7:raise RuntimeError('Rigid-body dynamics equality failed')
  details.append(dict(frame=float(frames[k]),wrench_violation=wrench_error,effort_fraction=float(ratios[j]),joint=str(names[j]),torque=float(value[6+j])))
 foot_errors=[]
 for k in range(len(frames)):
  feet=model.evaluate(qs[k],vs[k])['feet']
  for foot in (0,1):
   foot_errors.append([np.linalg.norm(feet[foot,:3]-target_feet[k,foot,:3]),(Rotation.from_quat(feet[foot,3:])*Rotation.from_quat(target_feet[k,foot,3:]).inv()).magnitude()])
 peak=max(max(r['wrench_violation'],r['effort_fraction']-1,0.) for r in details)
 minimum_collision=float('inf')
 for pose in qs:
  collision_state=collision.model.snapshot(torch.tensor(pose[None],dtype=torch.float32));distance,_=collision.collision_geometry(collision_state)
  minimum_collision=min(minimum_collision,float(distance.min()))
 row=dict(minimum_collision_distance_m=minimum_collision,peak_collision_slack=float(x[nc+ns+nf:].max()),iteration=iteration,solver=status,peak_linear_slack=float(x[nc:nc+ns].max()),peak_exact_violation=peak,foot_error=np.max(foot_errors,axis=0).tolist(),peak_raw_joint_rmse=float(np.sqrt(np.mean((out[:,7:]-source[:,7:])**2,axis=1)).max()))
 history.append(row);print(json.dumps(row),flush=True)
 np.savez_compressed(p/(args.name+'.checkpoint.npz'),rejected_or_unexecuted_control_q=out,coefficients=coeff,frames=frames,left_wrenches=x[nc+ns:nc+ns+nf].reshape(-1,6),left_wrench_frames=[float(frames[indices[n]]) for n in double_nodes])
 if minimum_collision>=.01999 and peak<5e-5 and max(np.asarray(foot_errors)[:,0])<.001:break
report=dict(peak_runtime_offset=np.max(abs(B@coeff),axis=0).tolist(),details=details,parent=str(parent_path),search_parameters=vars(args),scope='Offline nominal full-sole dynamics with original sphere and added convex wrist/body distances (20 mm), source contact labels, fixed root orientation, preserved foot paths; no physical tracking certification.',iterations=history,worst_wrench=max(details,key=lambda r:r['wrench_violation']),worst_effort=max(details,key=lambda r:r['effort_fraction']),physical_tracking_certified=False)
report.update(joint_extrema(sampler(out).smooth,start-6,end+6,240.,limits,speed))
report['preflight_passed']=bool(minimum_collision>=.01999 and peak<1e-4 and max(np.asarray(foot_errors)[:,0])<.001 and max(np.asarray(foot_errors)[:,1])<.01 and row['peak_raw_joint_rmse']<.19 and report['minimum_continuous_joint_margin']>0 and report['maximum_continuous_joint_velocity_fraction']<=1)
(p/(args.name+'.json')).write_text(json.dumps(report,indent=2)+'\n')
np.savez_compressed(p/(args.name+'.diagnostic.npz'),rejected_or_unexecuted_control_q=out,coefficients=coeff,frames=frames,left_wrenches=x[nc+ns:nc+ns+nf].reshape(-1,6),left_wrench_frames=[float(frames[indices[n]]) for n in double_nodes])
if report['preflight_passed']:np.savez_compressed(p/(args.name+'.npz'),control_q=out,fps=200.,baseline_sha256=parent['baseline_sha256'],source_sha256=parent['source_sha256'])
print(json.dumps(report,indent=2),flush=True)
