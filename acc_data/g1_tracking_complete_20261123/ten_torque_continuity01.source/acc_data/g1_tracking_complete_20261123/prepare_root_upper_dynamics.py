"""Offline root translation and upper-body fit with foot-preserving leg coupling.

Linearized RNEA includes contact wrench, motor effort and velocity constraints.
Double support uses equal force sharing as an explicit preparation assumption.
Bounded IK restores exact foot poses, followed by mandatory nonlinear audits.
This is not a runtime controller or a dynamic certificate.
"""
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import osqp
import torch
from scipy import sparse
from scipy.interpolate import BSpline
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[2]; OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'acc_data/amp_walk_20261103/feedback_followup'))
from numpy_centroidal_geometry import CentroidalGeometry
from blended_spline import BlendedSplineReference
from fabrics_sim.reference.interpolation import MotionInterpolator
from smooth_clock import smooth_clock
label=sys.argv[1];root_bound=float(sys.argv[2]);upper_bound=float(sys.argv[3]);bound=1.
parent=OUT/'prepared_combined_root_y50.npz';target=OUT/f'prepared_root_upper_dynamics_{label}.npz'
if target.exists():raise FileExistsError(target)
with np.load(parent) as data:base={k:data[k] for k in data.files}
source=np.load(ROOT/'acc_data/g1_tracking_contacts_20261123/eight_verify.control_reference.npz')['source_q']
assert hashlib.sha256(source.tobytes()).hexdigest()==base['source_sha256'].item()
names=np.load(OUT/'ten_verify.npz')['final_joint_order'].tolist()
model=CentroidalGeometry(str(ROOT/'src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf'),names)
ref=torch.tensor(base['control_q']);sampler=BlendedSplineReference(MotionInterpolator(ref),ref,stride=6,start_frame=130,ramp_frames=20)
cache=np.load(OUT/'reference_com_cache.npz');frames=np.arange(648,1033,2);times=cache['time'][frames];contacts=cache['contacts'][frames]
nodes=np.unique(np.r_[np.arange(0,len(source),6),len(source)-1]);knots=np.r_[np.full(4,nodes[0]),nodes[2:-2],np.full(4,nodes[-1])]
greville=np.array([np.mean(knots[i+1:i+4]) for i in range(len(nodes))]);select=np.flatnonzero((greville>=672)&(greville<=1008))
basis=BSpline(knots,np.eye(len(nodes))[:,select],3);B=basis(frames)
rates=np.array([smooth_clock(t,.9,.35,1.2,2.3,.25,.8,3.,.2,1.2)[1:] for t in times])*200
Bv=basis(frames,1)*rates[:,0,None]
Ba=basis(frames,2)*rates[:,0,None]**2+basis(frames,1)*rates[:,1,None]
n=len(frames);nv=len(select)*20
def advance(q,v,dt):
    q=q.copy();q[:3]+=dt*v[:3];q[3:7]=(Rotation.from_rotvec(dt*v[3:6])*Rotation.from_quat(q[3:7])).as_quat();q[7:]+=dt*v[6:];return q
state_path=OUT/'root_upper_dynamics_linearization.npz'
parent_hash=hashlib.sha256(parent.read_bytes()).hexdigest()
from g1_armature import unitree_armatures
armature=np.asarray(unitree_armatures(names)).ravel()
from numpy_leg_kinematics import LegKinematics
legs=LegKinematics(str(ROOT/'src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf'),names)
q_indices=np.r_[np.arange(3),np.arange(7,36)];v_indices=np.r_[np.arange(3),np.arange(6,35)]
node_maps=[];map_residual=0.
for k in greville[select].astype(int):
    N=np.zeros((32,20));N[:3,:3]=root_bound*np.eye(3);N[15:,3:]=upper_bound*np.eye(17)
    for side in (0,1):
        Jf=legs.evaluate(base['control_q'][k],side)[2];target_velocity=np.vstack((-root_bound*np.eye(3),np.zeros((3,3))))
        N[3+6*side:9+6*side,:3]=np.linalg.lstsq(Jf,target_velocity,rcond=1e-8)[0]
        map_residual=max(map_residual,float(np.max(abs(Jf@N[3+6*side:9+6*side,:3]-target_velocity))))
    node_maps.append(N)
node_maps=np.asarray(node_maps)
assert map_residual<1e-5,map_residual
def mapped_basis(row):
    return np.concatenate([weight*N for weight,N in zip(row,node_maps)],axis=1)
def outputs(q,v,a,contact):
    c=model.evaluate(q)[0];tau=model.inverse_dynamics(q,v,a)
    F=tau[:3].copy();M=tau[3:6].copy();H=M-np.cross(c-q[:3],F)
    count=max(1,int(np.sum(contact)))
    for side in range(2):
        if not contact[side]:continue
        foot,_,Jf=legs.evaluate(q,side)
        wrench=np.r_[F,M-np.cross(foot-q[:3],F)]/count
        tau[6+side*6:12+side*6]-=Jf.T@wrench
    return np.r_[F,H,tau[6:]+armature*a[6:]]
if state_path.exists():
    states=np.load(state_path);assert states['parent_sha256'].item()==parent_hash
    C=states['C'];F=states['F'];H=states['H'];J=states['J'];feet=states['feet']
    torque=states['torque'];Lq=states['Lq'];Lv=states['Lv'];La=states['La'];velocities=states['velocities']
else:
    Cs=[];Fs=[];Hs=[];Js=[];footposes=[];torques=[];q_derivatives=[];v_derivatives=[];a_derivatives=[];velocities=[]
    for i,k in enumerate(frames):
        q,v,a=[x.numpy()[0].astype(float) for x in sampler.kinematics(float(k),*rates[i])]
        c,_,poses=model.evaluate(q);jc,_=model.centroidal_matrices(q);eps=1e-4;value=outputs(q,v,a,contacts[i])
        lq=np.zeros((35,32));lv=lq.copy();la=lq.copy()
        for j,(qi,vi) in enumerate(zip(q_indices,v_indices)):
            plus=q.copy();minus=q.copy();plus[qi]+=eps;minus[qi]-=eps
            lq[:,j]=(outputs(plus,v,a,contacts[i])-outputs(minus,v,a,contacts[i]))/(2*eps)
            plus=v.copy();minus=v.copy();plus[vi]+=eps;minus[vi]-=eps
            lv[:,j]=(outputs(q,plus,a,contacts[i])-outputs(q,minus,a,contacts[i]))/(2*eps)
            plus=a.copy();plus[vi]+=1.
            la[:,j]=outputs(q,v,plus,contacts[i])-value
        Cs.append(c);Fs.append(value[:3]);Hs.append(value[3:6]);torques.append(value[6:]);Js.append(jc[:,v_indices])
        q_derivatives.append(lq);v_derivatives.append(lv);a_derivatives.append(la);velocities.append(v[6:])
        corners=np.array([[.12,.025,-.035],[.12,-.025,-.035],[-.05,.025,-.035],[-.05,-.025,-.035]])
        footposes.append(np.array([(poses[side+'_ankle_roll_link'][1]@corners.T).T+poses[side+'_ankle_roll_link'][0] for side in ('left','right')]))
        if i%20==0:print('exact dynamics linearization frame',k,flush=True)
    C,F,H,J,feet,torque,Lq,Lv,La,velocities=map(np.asarray,(Cs,Fs,Hs,Js,footposes,torques,q_derivatives,v_derivatives,a_derivatives,velocities))
    np.savez_compressed(state_path,C=C,F=F,H=H,J=J,feet=feet,torque=torque,Lq=Lq,Lv=Lv,La=La,velocities=velocities,parent_sha256=parent_hash)
B0map=[mapped_basis(row) for row in B];B1map=[mapped_basis(row) for row in Bv];B2map=[mapped_basis(row) for row in Ba]
JB=sparse.vstack([sparse.csr_matrix(J[i]@B0map[i]) for i in range(n)],format='csr')
operators=[sparse.csr_matrix(Lq[i]@B0map[i]+Lv[i]@B1map[i]+La[i]@B2map[i]) for i in range(n)]
DF=sparse.vstack([op[:3] for op in operators],format='csr');DH=sparse.vstack([op[3:6] for op in operators],format='csr')
DT=sparse.vstack([op[6:] for op in operators],format='csr')
DV=sparse.vstack([sparse.csr_matrix(op[3:]) for op in B1map],format='csr')
urdf_root=ET.parse(ROOT/'src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf').getroot()
physical_limits={joint.get('name'):[float(joint.find('limit').get(key)) for key in ('effort','velocity')] for joint in urdf_root.findall('joint') if joint.find('limit') is not None}
effort,velocity_limit=np.array([physical_limits[name] for name in names]).T
effort*=.8
rows=[];rhs=[];kind=[]
for i,k in enumerate(frames):
    if not 690<=k<=990 or not contacts[i].any():continue
    dc=JB[3*i:3*i+3];df=DF[3*i:3*i+3];dh=DH[3*i:3*i+3]
    points=np.concatenate([feet[i,s] for s in range(2) if contacts[i,s]])[:,:2]
    for nx,ny,b in ConvexHull(points).equations:
        numerator=nx*(C[i,0]*F[i,2]-C[i,2]*F[i,0]-H[i,1])+ny*(C[i,1]*F[i,2]-C[i,2]*F[i,1]+H[i,0])+b*F[i,2]
        row=nx*(F[i,2]*dc[0]-F[i,0]*dc[2]+C[i,0]*df[2]-C[i,2]*df[0]-dh[1])+ny*(F[i,2]*dc[1]-F[i,1]*dc[2]+C[i,1]*df[2]-C[i,2]*df[1]+dh[0])+b*df[2]
        rows.append(row/(model.mass*9.81));rhs.append(-numerator/(model.mass*9.81));kind.append('cop')
    for angle in np.arange(8)*np.pi/4:
        x,y=np.cos(angle),np.sin(angle);mu=.55*np.cos(np.pi/8)
        rows.append((x*df[0]+y*df[1]-mu*df[2])/(model.mass*9.81));rhs.append(-(x*F[i,0]+y*F[i,1]-mu*F[i,2])/(model.mass*9.81));kind.append('friction')
    rows.append(-df[2]/(model.mass*9.81));rhs.append((F[i,2]-50)/(model.mass*9.81));kind.append('normal')
    for j in range(29):
        for sign in (-1.,1.):
            rows.append(sign*DT[29*i+j]/effort[j]);rhs.append(1.-sign*torque[i,j]/effort[j]);kind.append('joint_torque_equal_DS')
            rows.append(sign*DV[29*i+j]/velocity_limit[j]);rhs.append(1.-sign*velocities[i,j]/velocity_limit[j]);kind.append('joint_velocity')
G=sparse.vstack(rows,format='csc');rhs=np.asarray(rhs);ns=len(rhs)
# Every physical inequality has an explicit reported slack; no hidden claim of feasibility.
smooth=sparse.vstack([sparse.csr_matrix(op[3:]) for op in B2map],format='csc')
P0=sparse.eye(nv)+1e-6*smooth.T@smooth
P=sparse.block_diag([P0,1e5*sparse.eye(ns)],format='csc')
mat=sparse.vstack([sparse.eye(nv+ns),sparse.hstack([G,-sparse.eye(ns)])],format='csc')
urdf_root=ET.parse(ROOT/'src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf').getroot()
limits={j.get('name'):[float(j.find('limit').get(s)) for s in ('lower','upper')] for j in urdf_root.findall('joint') if j.find('limit') is not None}
joint_lower,joint_upper=np.array([limits[name] for name in names]).T
lower=np.full(nv,-1.);upper=np.full(nv,1.)
upper[2::20]=0. # Lowering only avoids worsening near-straight leg reach.
node_position_operator=sparse.block_diag([sparse.csr_matrix(N[3:]) for N in node_maps],format='csc')
node_positions=base['control_q'][greville[select].astype(int),7:]
node_lower=np.maximum(joint_lower-node_positions,-.25).ravel()
node_upper=np.minimum(joint_upper-node_positions,.25).ravel()
mat=sparse.vstack([mat,sparse.hstack([node_position_operator,sparse.csc_matrix((len(node_lower),ns))])],format='csc')
solver=osqp.OSQP();solver.setup(P=P,q=np.zeros(nv+ns),A=mat,
    l=np.r_[lower,np.zeros(ns),np.full(ns,-np.inf),node_lower],u=np.r_[upper,np.full(ns,np.inf),rhs,node_upper],
    eps_abs=5e-5,eps_rel=5e-5,max_iter=100000,rho=20.,adaptive_rho_tolerance=2.,polish=True,verbose=True)
fit=solver.solve();print('solver',fit.info.status,flush=True)
np.savez_compressed(target.with_suffix('.solver_iterate.npz'),x=fit.x,
    constraint_value=mat@fit.x,primal_objective=fit.info.obj_val)
report=dict(method='Offline root translation/upper-body spline fit with foot-nullspace leg coupling and exact RNEA derivatives; equal force sharing assumed in double support',
    solver_status=fit.info.status,root_axis_bound_m=root_bound,upper_axis_bound_rad=upper_bound,
    node_joint_correction_bound_rad=.25,parent_sha256=parent_hash,source_sha256=base['source_sha256'].item(),
    offline_solver_primal_residual=float(getattr(fit.info,'prim_res',getattr(fit.info,'pri_res',np.nan))),
    offline_solver_dual_residual=float(getattr(fit.info,'dual_res',getattr(fit.info,'dua_res',np.nan))),
    offline_solver_iterations=int(fit.info.iter),offline_solver_tolerance=5e-5,
    preflight_passed=False,full_tracking_passed=False)
if fit.info.status_val in (1,2):
    coefficients=fit.x[:nv].reshape(len(select),20)
    joint_coefficients=np.einsum('nij,nj->ni',node_maps,coefficients)
    global_coeff=np.zeros((len(nodes),32));global_coeff[select]=joint_coefficients
    correction=np.stack([np.interp(np.arange(len(source)),greville,global_coeff[:,j]) for j in range(32)],axis=1)
    out=base['control_q'].copy();out[:,q_indices]+=correction
    # Restore exact original world foot poses using bounded leg IK. Root
    # translation, root orientation and upper-body output stay fixed here.
    from scipy.optimize import lsq_linear
    max_foot_pos=max_foot_rot=0.
    for k in np.flatnonzero(np.max(abs(correction),axis=1)>1e-10):
        q=out[k].astype(float)
        for side in (0,1):
            goal_p,goal_R,_=legs.evaluate(base['control_q'][k],side);sl=slice(7+6*side,13+6*side)
            for iteration in range(120):
                pos,rot,Jf=legs.evaluate(q,side);error=np.r_[goal_p-pos,Rotation.from_matrix(goal_R@rot.T).as_rotvec()]
                if np.max(abs(error))<5e-7:break
                lo=np.maximum(-.05,legs.lower[6*side:6*side+6]-q[sl]);hi=np.minimum(.05,legs.upper[6*side:6*side+6]-q[sl])
                if np.any(lo>=hi):q[sl]=np.clip(q[sl],legs.lower[6*side:6*side+6]+1e-8,legs.upper[6*side:6*side+6]-1e-8);continue
                step=lsq_linear(np.vstack((Jf,.0001*np.eye(6))),np.r_[error,np.zeros(6)],bounds=(lo,hi),tol=1e-10).x;q[sl]+=step
            out[k]=q;pos,rot,_=legs.evaluate(out[k],side)
            max_foot_pos=max(max_foot_pos,float(np.linalg.norm(pos-goal_p)));max_foot_rot=max(max_foot_rot,float(Rotation.from_matrix(goal_R@rot.T).magnitude()))
    np.testing.assert_array_equal(out[:,3:7],base['control_q'][:,3:7])
    residual=G@fit.x[:nv]-rhs
    report.update(peak_source_joint_rmse_rad=float(np.sqrt(np.mean((out[:,7:]-source[:,7:])**2,axis=1)).max()),
        maximum_upper_joint_limit_violation_rad=float(np.maximum(joint_lower-out[:,7:],out[:,7:]-joint_upper).max()),
        maximum_foot_position_residual_m=max_foot_pos,maximum_foot_rotation_residual_rad=max_foot_rot,
        maximum_root_correction_m=float(np.linalg.norm(correction[:,:3],axis=1).max()),
        linearized_max_violation={name:float(np.max(residual[np.array(kind)==name])) for name in set(kind)},
        baseline_max_violation={name:float(np.max(-rhs[np.array(kind)==name])) for name in set(kind)},
        maximum_actual_joint_correction_rad=float(np.max(abs(out[:,7:]-base['control_q'][:,7:]))),
        recomputed_centroidal_and_closed_loop_still_required=True)
    report['preflight_passed']=(max_foot_pos<2e-5 and max_foot_rot<2e-4 and report['peak_source_joint_rmse_rad']<.19
        and report['maximum_upper_joint_limit_violation_rad']<1e-5)
    if report['preflight_passed']:
        with target.open('xb') as stream:np.savez_compressed(stream,**dict(base,control_q=out))
target.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n');print(report,flush=True)
