"""Offline upper-body spline fit to centroidal support constraints.

Feet and floating root remain exactly unchanged. This linearized preparation
is neither a controller nor a dynamic certificate; recomputed kinematics and
closed-loop checks remain mandatory.
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
label=sys.argv[1];bound=float(sys.argv[2])
parent=OUT/'prepared_combined_root_y50.npz';target=OUT/f'prepared_upper_dynamics_{label}.npz'
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
n=len(frames);nv=len(select)*17
def advance(q,v,dt):
    q=q.copy();q[:3]+=dt*v[:3];q[3:7]=(Rotation.from_rotvec(dt*v[3:6])*Rotation.from_quat(q[3:7])).as_quat();q[7:]+=dt*v[6:];return q
state_path=OUT/'upper_dynamics_linearization.npz'
parent_hash=hashlib.sha256(parent.read_bytes()).hexdigest()
from g1_armature import unitree_armatures
armature=np.asarray(unitree_armatures(names)).ravel()[12:]
def outputs(q,v,a):
    c=model.evaluate(q)[0];tau=model.inverse_dynamics(q,v,a)
    return np.r_[tau[:3],tau[3:6]-np.cross(c-q[:3],tau[:3]),tau[18:]+armature*a[18:]]
if state_path.exists():
    states=np.load(state_path);assert states['parent_sha256'].item()==parent_hash
    C=states['C'];F=states['F'];H=states['H'];J=states['J'];feet=states['feet']
    torque=states['torque'];Lq=states['Lq'];Lv=states['Lv'];La=states['La'];velocities=states['velocities']
else:
    Cs=[];Fs=[];Hs=[];Js=[];footposes=[];torques=[];q_derivatives=[];v_derivatives=[];a_derivatives=[];velocities=[]
    for i,k in enumerate(frames):
        q,v,a=[x.numpy()[0].astype(float) for x in sampler.kinematics(float(k),*rates[i])]
        c,_,poses=model.evaluate(q);jc,_=model.centroidal_matrices(q);eps=1e-4;value=outputs(q,v,a)
        lq=np.zeros((23,17));lv=lq.copy();la=lq.copy()
        for j in range(17):
            plus=q.copy();minus=q.copy();plus[19+j]+=eps;minus[19+j]-=eps
            lq[:,j]=(outputs(plus,v,a)-outputs(minus,v,a))/(2*eps)
            plus=v.copy();minus=v.copy();plus[18+j]+=eps;minus[18+j]-=eps
            lv[:,j]=(outputs(q,plus,a)-outputs(q,minus,a))/(2*eps)
            plus=a.copy();plus[18+j]+=1.
            la[:,j]=outputs(q,v,plus)-value
        Cs.append(c);Fs.append(value[:3]);Hs.append(value[3:6]);torques.append(value[6:]);Js.append(jc[:,18:])
        q_derivatives.append(lq);v_derivatives.append(lv);a_derivatives.append(la);velocities.append(v[18:])
        corners=np.array([[.12,.025,-.035],[.12,-.025,-.035],[-.05,.025,-.035],[-.05,-.025,-.035]])
        footposes.append(np.array([(poses[side+'_ankle_roll_link'][1]@corners.T).T+poses[side+'_ankle_roll_link'][0] for side in ('left','right')]))
        if i%20==0:print('exact dynamics linearization frame',k,flush=True)
    C,F,H,J,feet,torque,Lq,Lv,La,velocities=map(np.asarray,(Cs,Fs,Hs,Js,footposes,torques,q_derivatives,v_derivatives,a_derivatives,velocities))
    np.savez_compressed(state_path,C=C,F=F,H=H,J=J,feet=feet,torque=torque,Lq=Lq,Lv=Lv,La=La,velocities=velocities,parent_sha256=parent_hash)
JB=sparse.vstack([sparse.csr_matrix(np.kron(B[i:i+1],J[i])) for i in range(n)],format='csr')
operators=[sparse.csr_matrix(np.kron(B[i:i+1],Lq[i])+np.kron(Bv[i:i+1],Lv[i])+np.kron(Ba[i:i+1],La[i])) for i in range(n)]
DF=sparse.vstack([op[:3] for op in operators],format='csr');DH=sparse.vstack([op[3:6] for op in operators],format='csr')
DT=sparse.vstack([op[6:] for op in operators],format='csr')
DV=sparse.kron(sparse.csr_matrix(Bv),sparse.eye(17),format='csr')
urdf_root=ET.parse(ROOT/'src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf').getroot()
physical_limits={joint.get('name'):[float(joint.find('limit').get(key)) for key in ('effort','velocity')] for joint in urdf_root.findall('joint') if joint.find('limit') is not None}
effort,velocity_limit=np.array([physical_limits[name] for name in names[12:]]).T
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
    for j in range(17):
        for sign in (-1.,1.):
            rows.append(sign*DT[17*i+j]/effort[j]);rhs.append(1.-sign*torque[i,j]/effort[j]);kind.append('upper_torque')
            rows.append(sign*DV[17*i+j]/velocity_limit[j]);rhs.append(1.-sign*velocities[i,j]/velocity_limit[j]);kind.append('upper_velocity')
G=sparse.vstack(rows,format='csc');rhs=np.asarray(rhs);ns=len(rhs)
# Every physical inequality has an explicit reported slack; no hidden claim of feasibility.
smooth=sparse.kron(sparse.csr_matrix(Ba),sparse.eye(17),format='csc')
P0=sparse.eye(nv)+1e-6*smooth.T@smooth
P=sparse.block_diag([P0,1e5*sparse.eye(ns)],format='csc')
mat=sparse.vstack([sparse.eye(nv+ns),sparse.hstack([G,-sparse.eye(ns)])],format='csc')
urdf_root=ET.parse(ROOT/'src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf').getroot()
limits={j.get('name'):[float(j.find('limit').get(s)) for s in ('lower','upper')] for j in urdf_root.findall('joint') if j.find('limit') is not None}
joint_lower,joint_upper=np.array([limits[name] for name in names[12:]]).T
node_positions=base['control_q'][greville[select].astype(int),19:]
lower=np.maximum(-bound,joint_lower-node_positions).ravel()
upper=np.minimum(bound,joint_upper-node_positions).ravel()
solver=osqp.OSQP();solver.setup(P=P,q=np.zeros(nv+ns),A=mat,
    l=np.r_[lower,np.zeros(ns),np.full(ns,-np.inf)],u=np.r_[upper,np.full(ns,np.inf),rhs],
    eps_abs=2e-6,eps_rel=2e-6,max_iter=100000,polish=True,verbose=False)
fit=solver.solve();print('solver',fit.info.status,flush=True)
report=dict(method='Offline upper-body spline fit with exact RNEA derivatives, contact, torque and velocity inequalities with explicit slack',
    solver_status=fit.info.status,maximum_coefficient_rad=bound,parent_sha256=parent_hash,source_sha256=base['source_sha256'].item(),
    preflight_passed=False,full_tracking_passed=False)
if fit.info.status_val in (1,2):
    coefficients=fit.x[:nv].reshape(len(select),17);global_coeff=np.zeros((len(nodes),17));global_coeff[select]=coefficients
    # Interior Greville times are integral multiples of six. Linear interpolation
    # writes exactly the optimized coefficients at the runtime sampler's nodes.
    correction=np.stack([np.interp(np.arange(len(source)),greville,global_coeff[:,j]) for j in range(17)],axis=1)
    out=base['control_q'].copy();out[:,19:]+=correction
    np.testing.assert_array_equal(out[:,:19],base['control_q'][:,:19])
    sampler_out=BlendedSplineReference(MotionInterpolator(torch.tensor(out)),torch.tensor(out),stride=6,start_frame=130,ramp_frames=20)
    expected=B@coefficients
    measured=np.array([sampler_out(float(k)).numpy()[0,19:]-sampler(float(k)).numpy()[0,19:] for k in frames])
    interpolation_error=float(np.max(abs(expected-measured)))
    residual=G@fit.x[:nv]-rhs
    report.update(peak_source_joint_rmse_rad=float(np.sqrt(np.mean((out[:,7:]-source[:,7:])**2,axis=1)).max()),
        maximum_upper_joint_limit_violation_rad=float(np.maximum(joint_lower-out[:,19:],out[:,19:]-joint_upper).max()),
        runtime_coefficient_error_rad=interpolation_error,linearized_max_violation={name:float(np.max(residual[np.array(kind)==name])) for name in set(kind)},
        baseline_max_violation={name:float(np.max(-rhs[np.array(kind)==name])) for name in set(kind)},
        maximum_actual_joint_correction_rad=float(np.max(abs(correction))),
        recomputed_centroidal_and_closed_loop_still_required=True)
    report['preflight_passed']=(interpolation_error<1e-6 and report['peak_source_joint_rmse_rad']<.19
        and report['maximum_upper_joint_limit_violation_rad']<1e-5)
    if report['preflight_passed']:
        with target.open('xb') as stream:np.savez_compressed(stream,**dict(base,control_q=out))
target.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n');print(report,flush=True)
