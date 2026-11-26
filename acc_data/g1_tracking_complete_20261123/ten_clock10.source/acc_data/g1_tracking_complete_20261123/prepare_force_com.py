"""Offline whole-clip COM force fit and bounded foot-preserving IK.

This reference preparation is not a runtime planner. The unchanged original
source remains the fidelity target. Force constraints are necessary only;
they do not certify full-body or contact-moment feasibility.
"""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import osqp
from scipy import sparse
from scipy.optimize import lsq_linear
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[2]; OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'acc_data/amp_walk_20261103/feedback_followup'))
from numpy_leg_kinematics import LegKinematics
from numpy_centroidal_geometry import CentroidalGeometry

label=sys.argv[1]; bound=float(sys.argv[2]); mu=float(sys.argv[3])
fixed_root='--fixed-root' in sys.argv[4:]
rotation_weight=float(sys.argv[sys.argv.index('--rotation-weight')+1]) if '--rotation-weight' in sys.argv else .05
target=OUT/f'prepared_force_com_{label}.npz'
if target.exists(): raise FileExistsError(target)
parent=OUT/'prepared_combined_root_y50.npz'
with np.load(parent) as a: base={k:a[k] for k in a.files}
with np.load(OUT/'reference_com_cache.npz') as a: cache={k:a[k] for k in a.files}
assert cache['parent_sha256'].item()==hashlib.sha256(parent.read_bytes()).hexdigest()
source=np.load(ROOT/'acc_data/g1_tracking_contacts_20261123/eight_verify.control_reference.npz')['source_q']
assert hashlib.sha256(source.tobytes()).hexdigest()==base['source_sha256'].item()
names=np.load(OUT/'ten_verify.npz')['final_joint_order'].tolist()
urdf=str(ROOT/'src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf')
legs=LegKinematics(urdf,names); model=CentroidalGeometry(urdf,names)
D2=sparse.load_npz(OUT/'reference_com_D2.npz'); n=len(source); times=cache['time']; C=cache['com']
D3=sparse.diags(1/np.diff(times[1:-1]))@sparse.diags([-np.ones(n-3),np.ones(n-3)],[0,1],shape=(n-3,n-2))@D2
S2,S3=D2/1000.,D3/1000.
P0=sparse.eye(n)+4*S2.T@S2+.0004*S3.T@S3
P=sparse.block_diag([P0]*3,format='csc')
linear=np.concatenate([4*S2.T@(D2@C[:,j])+.0004*S3.T@(D3@C[:,j]) for j in range(3)])
low=np.full((3,n),-bound); high=np.full((3,n),bound)
low[2]=-25.; high[2]=0.
low[:,:641]=high[:,:641]=0.; low[:,1980:]=high[:,1980:]=0.
rows=np.arange(659,1960); S=S2[rows]; az=(D2@C[:,2])[rows]
zero=sparse.csr_matrix(S.shape)
A=[sparse.eye(3*n),sparse.hstack([zero,zero,S])]
lower=[low.ravel(),-4.-az]; upper=[high.ravel(),8.-az]
# Eight tangent halfplanes approximate a circular friction cone from outside.
# Use cos(pi/8) margin so all feasible horizontal forces lie inside mu Fz.
for angle in np.arange(8)*np.pi/4:
    x,y=np.cos(angle),np.sin(angle); m=mu*np.cos(np.pi/8)
    A.append(sparse.hstack([x*S,y*S,-m*S]))
    upper.append(m*(9.81+az)-(D2@(x*C[:,0]+y*C[:,1]))[rows])
    lower.append(np.full(len(rows),-np.inf))
solver=osqp.OSQP(); solver.setup(P=P,q=linear,A=sparse.vstack(A,format='csc'),
    l=np.concatenate(lower),u=np.concatenate(upper),eps_abs=2e-6,eps_rel=2e-6,
    max_iter=150000,polish=True,verbose=False)
fit=solver.solve()
report=dict(method='Offline bounded COM force fit with unchanged foot poses',
    fixed_root_orientation=fixed_root,
    rotation_regularization=rotation_weight,
    offline_solver_status=fit.info.status,friction_coefficient=mu,xy_correction_bound_mm=bound,
    source_sha256=base['source_sha256'].item(),parent_sha256=cache['parent_sha256'].item(),
    full_tracking_passed=False,preflight_passed=False)
if fit.info.status_val not in (1,2):
    target.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n'); print(report,flush=True);sys.exit(1)
delta=fit.x.reshape(3,n).T/1000.; delta[:641]=0.;delta[1980:]=0.
acc=D2@(C+delta); friction=np.linalg.norm(acc[rows,:2],axis=1)-mu*(9.81+acc[rows,2])
if max(friction.max(),(-4.-acc[rows,2]).max(),(acc[rows,2]-8.).max())>1e-3:
    raise ValueError('Independent force fit certificate failed')
def skew(v):
    x,y,z=v; return np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])
out=base['control_q'].copy(); max_pos=max_rot=max_com=max_rootrot=0.;worst=None
for k in range(641,1980):
    if np.max(abs(delta[k]))<1e-9:continue
    original=base['control_q'][k].astype(float); desired=model.evaluate(original)[0]+delta[k]
    goals=[legs.evaluate(original,s)[:2] for s in (0,1)]
    q=original.copy(); q[:3]+=delta[k]; root0=Rotation.from_quat(original[3:7])
    for iteration in range(100):
        J=np.zeros((15,18)); error=np.zeros(15)
        for side in (0,1):
            pos,rot,legJ=legs.evaluate(q,side); p,R=goals[side]; sl=slice(6*side,6*(side+1))
            error[sl]=np.r_[p-pos,Rotation.from_matrix(R@rot.T).as_rotvec()]
            J[sl,:3]=np.vstack((np.eye(3),np.zeros((3,3))))
            J[sl,3:6]=np.vstack((-skew(pos-q[:3]),np.eye(3)))
            J[sl,6+6*side:12+6*side]=legJ
        com,Jc,_=model.evaluate(q);error[12:]=desired-com
        J[12:,:3]=np.eye(3);J[12:,3:6]=-skew(com-q[:3]);J[12:,6:]=Jc[:,:12]
        if np.max(abs(error))<5e-7:break
        dr=q[:3]-original[:3];rr=(Rotation.from_quat(q[3:7])*root0.inv()).as_rotvec()
        lo=np.r_[np.maximum(-.004,-.08-dr),np.maximum(-.01,-.05-rr),np.maximum(-.06,legs.lower-q[7:19])]
        hi=np.r_[np.minimum(.004,.08-dr),np.minimum(.01,.05-rr),np.minimum(.06,legs.upper-q[7:19])]
        regularizer=np.diag(np.r_[np.full(3,.005),np.full(3,rotation_weight),np.full(12,.001)])
        if fixed_root:
            keep=np.r_[np.arange(3),np.arange(6,18)];step=np.zeros(18)
            step[keep]=lsq_linear(np.vstack((J[:,keep],regularizer[:,keep])),np.r_[error,np.zeros(18)],bounds=(lo[keep],hi[keep]),tol=1e-10,max_iter=100).x
        else:
            step=lsq_linear(np.vstack((J,regularizer)),np.r_[error,np.zeros(18)],bounds=(lo,hi),tol=1e-10,max_iter=100).x
        q[:3]+=step[:3];q[3:7]=(Rotation.from_rotvec(step[3:6])*Rotation.from_quat(q[3:7])).as_quat();q[7:19]+=step[6:]
    out[k]=q
    for side in (0,1):
        p,R,_=legs.evaluate(out[k],side);pe=float(np.linalg.norm(p-goals[side][0]));re=float(Rotation.from_matrix(R@goals[side][1].T).magnitude())
        if max(pe,re)>max(max_pos,max_rot):worst=k
        max_pos=max(max_pos,pe);max_rot=max(max_rot,re)
    max_com=max(max_com,float(np.linalg.norm(model.evaluate(out[k])[0]-desired)))
    max_rootrot=max(max_rootrot,float((Rotation.from_quat(out[k,3:7])*root0.inv()).magnitude()))
    if k%200==0:print('IK frame',k,'maximum residual',max(max_pos,max_rot,max_com),flush=True)
np.testing.assert_array_equal(out[:641],base['control_q'][:641]);np.testing.assert_array_equal(out[1980:],base['control_q'][1980:])
report.update(maximum_com_correction_by_axis_m=np.max(abs(delta),axis=0).tolist(),
    maximum_foot_position_residual_m=max_pos,maximum_foot_rotation_residual_rad=max_rot,
    maximum_com_ik_residual_m=max_com,maximum_added_root_rotation_rad=max_rootrot,worst_frame=worst,
    peak_source_joint_rmse_rad=float(np.sqrt(np.mean((out[:,7:]-source[:,7:])**2,axis=1)).max()),
    peak_source_root_displacement_m=float(np.linalg.norm(out[:,:3]-source[:,:3],axis=1).max()),
    maximum_friction_acceleration_excess_m_s2=float(friction.max()),runtime_spline_acceleration_still_requires_audit=True)
report['preflight_passed']=(max_pos<2e-5 and max_rot<2e-4 and max_com<2e-5 and max_rootrot<.1
    and report['peak_source_joint_rmse_rad']<.19 and report['peak_source_root_displacement_m']<.15)
target.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
if report['preflight_passed']:
    with target.open('xb') as stream:np.savez_compressed(stream,**dict(base,control_q=out))
print(report,flush=True)
