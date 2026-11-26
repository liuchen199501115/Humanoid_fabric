"""Offline upper-body-only correction using exact runtime derivatives and RNEA.
A full-sole wrench check is optimistic; passing it is not physical tracking.
"""
from pathlib import Path
import json,pickle,argparse
import numpy as np
import torch
from scipy import sparse
from scipy.interpolate import BSpline
from scipy.spatial.transform import Rotation
from blended_spline import BlendedSplineReference
from reference_smoothing_basis import runtime_smoothing_basis
from mujoco_reference_dynamics import ReferenceDynamics
from full_effort_linearization import effort_variations,inverse_output
from solve_upper_fit_qp import solve
from g1_model_effort import model_effort_limits
from fabrics_sim.reference.interpolation import MotionInterpolator
from fabrics_sim.solver.AccelerationQP import build_single_support_wrench_constraints
parser=argparse.ArgumentParser();parser.add_argument('--grid-spacing',type=float,default=3.);parser.add_argument('--check-end',type=float,default=1011.);parser.add_argument('--initial-fit');parser.add_argument('--effort-fraction',type=float,default=.8);parser.add_argument('--parent',default='acc_data/g1_full_tracking_20261106/thirteenth_force_fit150.npz');parser.add_argument('--offset-limit',type=float,default=.12);parser.add_argument('--iterations',type=int,default=4);parser.add_argument('--name',default='step13_upper_wrench_fit');args=parser.parse_args()
if not .5 <= args.grid_spacing <= 3 or not 1011 <= args.check_end <= 1035 or not .8 <= args.effort_fraction <= 1 or not 0 < args.offset_limit <= .3 or not 1 <= args.iterations <= 12:raise ValueError('Invalid search parameters')
if args.grid_spacing!=.5:raise ValueError('Contact-aware fitting requires the half-frame grid')
p=Path(__file__).parent;name=args.name;parent=np.load(args.parent)
if (p/(name+'.json')).exists():raise RuntimeError('Refuse to overwrite an existing fit report')
base=parent['control_q'].astype(float);out=base.copy();record=np.load(p/'rolling_contact.npz')
source=np.load(p/'rolling_contact.control_reference.npz')['source_q'];names=record['final_joint_order']
urdf='src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf'
model=ReferenceDynamics(urdf,names);limits=record['joint_position_limits'][0]
effort=args.effort_fraction*model_effort_limits(urdf,names,record['effort_limits'].reshape(-1))
speed=record['history_joint_velocity_limits'][-1].reshape(-1)
contacts=np.asarray(pickle.load(open('acc_data/amp_walk_20261102/amp_walk_native_contacts.pkl','rb'))['contact_sequence'])
start,end=924,1092;frames=np.arange(start,end+.01,args.grid_spacing);times=frames/240.;U=17
indices=[k for k,f in enumerate(frames) if 960<=f<=args.check_end]
assert all(contacts[int(frames[k]),1] for k in indices)
feet=[1]*len(indices)  # Eliminate the right wrench; allocate a left wrench in planned double support.
knots=np.r_[np.full(4,start),np.arange(start+12,end,12),np.full(4,end)];K=len(knots)-4;raw_basis=BSpline(knots,np.eye(K),3)
basis=runtime_smoothing_basis(raw_basis,len(base),6);B=basis(frames);V=basis(frames,1)*240;A=basis(frames,2)*240**2;J=basis(frames,3)*240**3
BM=sparse.kron(B,np.eye(U),format='csc');VM=sparse.kron(V,np.eye(U),format='csc');AM=sparse.kron(A,np.eye(U),format='csc');JM=sparse.kron(J,np.eye(U),format='csc')
nc=K*U;ns=len(indices);coeff=np.zeros((K,U));history=[]
double_nodes=[node for node,k in enumerate(indices) if contacts[int(frames[k])].all()]
left_slices={node:slice(nc+ns+6*i,nc+ns+6*(i+1)) for i,node in enumerate(double_nodes)}
nf=6*len(double_nodes);total=nc+ns+nf
frame_nodes={round(float(frames[k]),6):node for node,k in enumerate(indices)}
if args.initial_fit:
 initial=np.load(args.initial_fit);coeff=initial['coefficients'];out=initial['rejected_or_unexecuted_control_q']
 if coeff.shape!=(K,U) or out.shape!=base.shape or not np.array_equal(out[:,:19],base[:,:19]):raise ValueError('Initial fit does not preserve this parent root and legs')
 offset=BSpline(knots,coeff,3)(np.arange(len(base)));offset[:start]=0;offset[end+1:]=0
 if not np.allclose(out[:,19:],base[:,19:]+offset,atol=1e-10,rtol=0):raise ValueError('Initial fit belongs to a different upper-body parent')
source_upper=np.stack([np.interp(frames,np.arange(len(source)),source[:,j]) for j in range(19,36)],axis=1)
T=np.zeros((len(frames),35,U));T[:,18:]=np.eye(U);Td=np.zeros_like(T)
def sample(trajectory,fs):
    tensor=torch.tensor(trajectory,dtype=torch.float64);sampler=BlendedSplineReference(MotionInterpolator(tensor),tensor,stride=6,start_frame=130,ramp_frames=20)
    data=[sampler.kinematics(float(f),240.) for f in fs]
    return tuple(np.array([x[j].numpy()[0] for x in data]) for j in range(3))
def wrench_rows(q,foot):
    state=model.evaluate(q,np.zeros(35));R=Rotation.from_quat(state['feet'][:,3:]).as_matrix()
    G,b=build_single_support_wrench_constraints(contact_mode=torch.tensor([foot+1]),foot_rotation_world=torch.tensor(R[None]),friction_coefficient=.6,coupled_yaw=True,toe_length=.085,heel_length=.025,half_width=.02,sole_depth=.035,minimum_normal_force=20. if foot==0 else 100.,maximum_normal_force=800.)
    G=G.numpy()[0,:,6*foot:6*foot+6];b=b.numpy()[0];active=np.any(abs(G)>0,axis=1)
    G,b=G[active],b[active];scale=np.linalg.norm(G*np.array([350.,350.,350.,35.,35.,35.]),axis=1)
    return G/scale[:,None],b/scale
def left_response(state):
    L,R=state['jacobian']
    B=np.linalg.solve(R[:,:6].T,L[:,:6].T)
    return np.vstack((-B,R[:,6:].T@B-L[:,6:].T))
def evaluate_node(q,v,a,node,solution):
    state=model.evaluate(q,v,a);value=inverse_output(state,1);left=np.zeros(6)
    if node in left_slices:
        left=solution[left_slices[node]];value+=left_response(state)@left
    G,b=wrench_rows(q,1);violation=float(np.maximum(G@value[:6]-b,0).max())
    if node in left_slices:
        L,c=wrench_rows(q,0);violation=max(violation,float(np.maximum(L@left-c,0).max()))
    reconstructed=state['jacobian'][1].T@value[:6]+state['jacobian'][0].T@left+np.r_[np.zeros(6),value[6:]]
    return value,violation,float(np.max(abs(reconstructed-state['generalized_force'])))
for iteration in range(args.iterations):
    qs,vs,acc=sample(out,frames);maps=effort_variations(model,qs,vs,acc,T,Td,times,indices,feet)
    H=2*(2000*BM.T@BM+2*VM.T@VM+.00001*JM.T@JM+.01*AM.T@AM)+sparse.eye(nc)*1e-6
    H=sparse.block_diag((H,sparse.eye(ns)*2e8,sparse.eye(nf)*1e-6),format='csc');linear=np.zeros(total)
    source_error=qs[:,19:]-B@coeff-source_upper
    linear[:nc]=4000*np.asarray(BM.T@source_error.ravel()).ravel()
    mats=[];low=[];high=[]
    def add(matrix,lo,hi):
        matrix=sparse.csc_matrix(matrix)
        if matrix.shape[1]<total:matrix=sparse.hstack((matrix,sparse.csc_matrix((matrix.shape[0],total-matrix.shape[1]))),format='csc')
        mats.append(matrix);low.extend(np.broadcast_to(lo,(matrix.shape[0],)));high.extend(np.broadcast_to(hi,(matrix.shape[0],)))
    endpoint=np.r_[np.arange(3*U),np.arange(nc-3*U,nc)];E=sparse.eye(nc,format='csc')[endpoint];add(E,0.,0.)
    add(BM,-args.offset_limit,args.offset_limit);add(sparse.eye(nc),coeff.ravel()-.04,coeff.ravel()+.04)
    add(BM,(limits[12:,0]+.01-qs[:,19:]+B@coeff).ravel(),(limits[12:,1]-.01-qs[:,19:]+B@coeff).ravel())
    add(VM,(-speed[12:]-vs[:,18:]+V@coeff).ravel(),(speed[12:]-vs[:,18:]+V@coeff).ravel())
    add(sparse.hstack((sparse.csc_matrix((ns,nc)),sparse.eye(ns))),0.,np.inf)
    for node,(k,foot) in enumerate(zip(indices,feet)):
        raw,P,W,Q=maps[k];mapping=np.kron(B[k],P)+np.kron(V[k],W)+np.kron(A[k],Q);constant=raw-mapping@coeff.ravel();G,b=wrench_rows(qs[k],foot)
        D=np.zeros((len(G)+58,35));D[:len(G),:6]=G;D[len(G):len(G)+29,6:]=np.diag(1/effort);D[len(G)+29:,6:]=-np.diag(1/effort)
        upper=np.r_[b,np.ones(58)]-D@constant;rows=np.c_[D@mapping,np.zeros((len(D),ns+nf))];rows[:,nc+node]=-1
        if node in left_slices:
            rows[:,left_slices[node]]=D@left_response(model.evaluate(qs[k],vs[k],acc[k]))
            L,c=wrench_rows(qs[k],0);left_rows=np.zeros((len(L),total));left_rows[:,left_slices[node]]=L;left_rows[:,nc+node]=-1
            add(left_rows,-np.inf,c)
        add(rows,-np.inf,upper)
    x,status=solve(H,linear,sparse.vstack(mats,format='csc'),np.array(low),np.array(high))
    if not status['accepted']:raise RuntimeError(status)
    coeff=x[:nc].reshape(K,U);offset=BSpline(knots,coeff,3)(np.arange(len(base)));offset[:start]=0;offset[end+1:]=0
    out=base.copy();out[:,19:]+=offset
    qs,vs,acc=sample(out,frames);violations=[]
    for node,(k,foot) in enumerate(zip(indices,feet)):
        actual,wrench_error,equality_error=evaluate_node(qs[k],vs[k],acc[k],node,x)
        if equality_error>1e-7:raise RuntimeError('Force allocation did not reconstruct rigid-body dynamics')
        violations.append(max(0.,wrench_error,float(np.max(abs(actual[6:])/effort-1))))
    row=dict(iteration=iteration,solver=status,peak_linear_slack=float(x[nc:nc+ns].max()),peak_exact_normalized_violation=max(violations),peak_upper_offset=float(abs(offset).max()))
    history.append(row);print(json.dumps(row),flush=True)
fs=np.arange(start-6,end+6.01,.5);qs,vs,acc=sample(out,fs);margin=float(np.minimum(qs[:,7:]-limits[:,0],limits[:,1]-qs[:,7:]).min());rmse=float(np.sqrt(np.mean((out[:,7:]-source[:,7:])**2,axis=1)).max())
dense_violations=[];dense_details=[];equality_errors=[]
for k,f in enumerate(fs):
    if not 960<=f<=args.check_end:continue
    node=frame_nodes[round(float(f),6)];value,wrench_error,equality_error=evaluate_node(qs[k],vs[k],acc[k],node,x);equality_errors.append(equality_error)
    dense_violations.append(max(0.,wrench_error,float(np.max(abs(value[6:])/effort-1))))
    ratio=abs(value[6:])/effort;j=int(np.argmax(ratio))
    dense_details.append(dict(frame=float(f),peak_wrench_normalized_violation=wrench_error,peak_effort_fraction=float(ratio[j]),joint=str(names[j]),torque=float(value[6+j]),effort_cap=float(effort[j])))
report=dict(parent=args.parent,pose_cost_target="raw source upper joints at each fitted frame",maximum_inverse_dynamics_equality_error=max(equality_errors),planned_double_support_nodes=len(double_nodes),search_parameters=vars(args),worst_effort=max(dense_details,key=lambda row:row['peak_effort_fraction']),worst_wrench=max(dense_details,key=lambda row:row['peak_wrench_normalized_violation']),peak_dense_normalized_violation=max(dense_violations),scope=f'Upper-body-only optimistic full-sole inverse-dynamics fit over source frames 960..{args.check_end:g}, using source-labeled right and double support with explicit left-wrench allocation. No change to parent root or legs; no claim about stance velocity or observed point/line geometry.',iterations=history,peak_raw_reference_joint_rmse=rmse,minimum_dense_joint_margin=margin,peak_dense_speed_fraction=float(np.max(abs(vs[:,6:])/speed)),physical_tracking_certified=False)
from reference_joint_extrema import joint_extrema
tensor=torch.tensor(out,dtype=torch.float64);sampler=BlendedSplineReference(MotionInterpolator(tensor),tensor,stride=6,start_frame=130,ramp_frames=20)
report.update(joint_extrema(sampler.smooth,start-6,end+6,240.,limits,speed))
report['preflight_passed']=bool(report['peak_dense_normalized_violation']<1e-4 and report['minimum_continuous_joint_margin']>0 and report['maximum_continuous_joint_velocity_fraction']<=1 and rmse<.19 and margin>0 and report['peak_dense_speed_fraction']<=1)
(p/(name+'.json')).write_text(json.dumps(report,indent=2)+'\n')
np.savez_compressed(p/(name+'.diagnostic.npz'),rejected_or_unexecuted_control_q=out,coefficients=coeff,frames=fs,left_wrenches=x[nc+ns:].reshape(-1,6),left_wrench_frames=[float(frames[indices[node]]) for node in double_nodes])
if report['preflight_passed']:np.savez_compressed(p/(name+'.npz'),control_q=out,fps=200.,baseline_sha256=parent['baseline_sha256'],source_sha256=parent['source_sha256'])
print(json.dumps(report,indent=2),flush=True)
