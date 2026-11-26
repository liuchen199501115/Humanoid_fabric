"""Offline COM-height preparation; no runtime planner or additional WBC.

Fit a small height correction with unilateral-support acceleration bounds,
then solve bounded full-body IK preserving both original control foot poses.
The raw PKL remains untouched and independently audited during tracking.
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

ROOT = Path(__file__).resolve().parents[2]; OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'acc_data/amp_walk_20261103/feedback_followup'))
from numpy_leg_kinematics import LegKinematics
from numpy_centroidal_geometry import CentroidalGeometry

parent = OUT/'prepared_combined_root_y50.npz'
with np.load(parent) as a: base = {k:a[k] for k in a.files}
with np.load(OUT/'reference_com_cache.npz') as a: cache = {k:a[k] for k in a.files}
assert cache['parent_sha256'].item() == hashlib.sha256(parent.read_bytes()).hexdigest()
source = np.load(ROOT/'acc_data/g1_tracking_contacts_20261123/eight_verify.control_reference.npz')['source_q']
assert hashlib.sha256(source.tobytes()).hexdigest() == base['source_sha256'].item()
names = np.load(OUT/'ten_verify.npz')['final_joint_order'].tolist()
urdf = str(ROOT/'src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf')
legs = LegKinematics(urdf, names); model = CentroidalGeometry(urdf, names)
D2 = sparse.load_npz(OUT/'reference_com_D2.npz')
n = len(source); times = cache['time']; z = cache['com'][:,2]
D3 = sparse.diags(1/np.diff(times[1:-1])) @ sparse.diags([-np.ones(n-3),np.ones(n-3)],[0,1],shape=(n-3,n-2)) @ D2
S2, S3 = D2/1000., D3/1000.
P = sparse.eye(n)+4*(S2.T@S2)+.0004*(S3.T@S3)
linear = 4*S2.T@(D2@z)+.0004*S3.T@(D3@z)
rows = np.arange(659,1960)
A = sparse.vstack((sparse.eye(n), S2[rows])).tocsc()
low_shift = np.full(n,-25.); high_shift = np.full(n,25.)
low_shift[:641] = high_shift[:641] = 0.
low_shift[1980:] = high_shift[1980:] = 0.

def skew(v):
    x,y,z = v
    return np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])

for low_acc, high_acc, label in [(-6.,8.,'6_8'),(-4.,6.,'4_6')]:
    target = OUT/f'prepared_vertical_com_{label}.npz'
    if target.exists(): raise FileExistsError(target)
    solver = osqp.OSQP()
    solver.setup(P=P.tocsc(), q=linear, A=A,
        l=np.r_[low_shift,low_acc-(D2@z)[rows]],
        u=np.r_[high_shift,high_acc-(D2@z)[rows]],
        eps_abs=1e-7,eps_rel=1e-7,max_iter=100000,polish=True,verbose=False)
    fit = solver.solve()
    if fit.info.status_val not in (1,2):
        raise RuntimeError('Offline COM fit: '+fit.info.status)
    delta = fit.x/1000.
    delta[:641] = 0.; delta[1980:] = 0.
    acc = D2@(z+delta)
    if max(float((low_acc-acc[rows]).max()),float((acc[rows]-high_acc).max())) > 1e-4:
        raise ValueError('Offline acceleration bound failed independent check')
    out = base['control_q'].copy(); max_pos=max_rot=max_com=max_rootrot=0.; worst=None
    for k in range(641,1980):
        if abs(delta[k]) < 1e-9: continue
        original = base['control_q'][k].astype(float)
        desired_com_z = model.evaluate(original)[0][2]+delta[k]
        goals = [legs.evaluate(original, side)[:2] for side in (0,1)]
        q = original.copy(); q[2] += delta[k]
        root0 = Rotation.from_quat(original[3:7])
        for iteration in range(80):
            J = np.zeros((13,16)); error = np.zeros(13)
            for side in (0,1):
                position,rotation,leg_J = legs.evaluate(q,side)
                p,R = goals[side]; sl=slice(6*side,6*(side+1))
                error[sl] = np.r_[p-position,Rotation.from_matrix(R@rotation.T).as_rotvec()]
                J[sl,0] = [0,0,1,0,0,0]
                J[sl,1:4] = np.vstack((-skew(position-q[:3]),np.eye(3)))
                J[sl,4+6*side:10+6*side] = leg_J
            com,Jc,_ = model.evaluate(q)
            error[12] = desired_com_z-com[2]
            J[12,0] = 1.; J[12,1:4] = -skew(com-q[:3])[2]; J[12,4:] = Jc[2,:12]
            if np.max(abs(error)) < 5e-7: break
            root_error = (Rotation.from_quat(q[3:7])*root0.inv()).as_rotvec()
            lo = np.r_[max(-.004,-.05-(q[2]-original[2])),np.maximum(-.01,-.05-root_error),np.maximum(-.06,legs.lower-q[7:19])]
            hi = np.r_[min(.004,.05-(q[2]-original[2])),np.minimum(.01,.05-root_error),np.minimum(.06,legs.upper-q[7:19])]
            regularizer = np.diag(np.r_[.005,np.full(3,.05),np.full(12,.001)])
            step = lsq_linear(np.vstack((J,regularizer)),np.r_[error,np.zeros(16)],bounds=(lo,hi),tol=1e-10,max_iter=100).x
            q[2] += step[0]; q[3:7]=(Rotation.from_rotvec(step[1:4])*Rotation.from_quat(q[3:7])).as_quat();q[7:19]+=step[4:]
        out[k] = q
        for side in (0,1):
            p,R,_=legs.evaluate(out[k],side)
            pe=float(np.linalg.norm(p-goals[side][0])); re=float(Rotation.from_matrix(R@goals[side][1].T).magnitude())
            if max(pe,re) > max(max_pos,max_rot):worst=k
            max_pos=max(max_pos,pe);max_rot=max(max_rot,re)
        max_com=max(max_com,abs(model.evaluate(out[k])[0][2]-desired_com_z))
        max_rootrot=max(max_rootrot,float((Rotation.from_quat(out[k,3:7])*root0.inv()).magnitude()))
    np.testing.assert_array_equal(out[:641],base['control_q'][:641])
    np.testing.assert_array_equal(out[1980:],base['control_q'][1980:])
    report=dict(method='Offline bounded COM height fit and foot-preserving IK; no runtime MPC',
        fitted_vertical_acceleration_limits=[low_acc,high_acc],offline_solver_status=fit.info.status,
        maximum_com_height_correction_m=float(abs(delta).max()),maximum_foot_position_residual_m=max_pos,
        maximum_foot_rotation_residual_rad=max_rot,maximum_com_ik_residual_m=max_com,
        maximum_added_root_rotation_rad=max_rootrot,worst_frame=worst,
        peak_source_joint_rmse_rad=float(np.sqrt(np.mean((out[:,7:]-source[:,7:])**2,axis=1)).max()),
        peak_source_root_displacement_m=float(np.linalg.norm(out[:,:3]-source[:,:3],axis=1).max()),
        source_sha256=base['source_sha256'].item(),parent_sha256=cache['parent_sha256'].item(),
        runtime_spline_acceleration_still_requires_audit=True,full_tracking_passed=False)
    report['preflight_passed']=(max_pos<2e-5 and max_rot<2e-4 and max_com<2e-5 and max_rootrot<.1
        and report['peak_source_joint_rmse_rad']<.19 and report['peak_source_root_displacement_m']<.15)
    target.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
    if report['preflight_passed']:
        with target.open('xb') as stream:np.savez_compressed(stream,**dict(base,control_q=out))
    print(report,flush=True)
