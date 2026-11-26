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
from reference_com_ik import fit_com_reference
out, ik_report = fit_com_reference(base, source, delta, legs, model,
    rotation_weight=rotation_weight, fixed_root=fixed_root,
    max_iterations=int(sys.argv[sys.argv.index('--ik-iterations')+1]) if '--ik-iterations' in sys.argv else 100)
report.update(ik_report)
report['maximum_friction_acceleration_excess_m_s2'] = float(friction.max())
target.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
if report['preflight_passed']:
    with target.open('xb') as stream:np.savez_compressed(stream,**dict(base,control_q=out))
print(report,flush=True)
