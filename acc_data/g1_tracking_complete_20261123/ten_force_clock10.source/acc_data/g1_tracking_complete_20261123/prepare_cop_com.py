"""Offline COM fit with frozen full-body angular momentum and sole polygons.

The linearized fit is only a candidate generator. IK, runtime interpolation,
recomputed angular momentum, and closed-loop tracking need independent checks.
"""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import osqp
from scipy import sparse
from scipy.spatial import ConvexHull

ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'acc_data/amp_walk_20261103/feedback_followup'))
from numpy_leg_kinematics import LegKinematics
from numpy_centroidal_geometry import CentroidalGeometry
from reference_com_ik import fit_com_reference

label=sys.argv[1];bound=float(sys.argv[2]);parent_name=sys.argv[3] if len(sys.argv)>3 else 'prepared_force_com_xy30_mu55'
parent=OUT/(parent_name+'.npz');target=OUT/('prepared_cop_'+label+'.npz')
if target.exists():raise FileExistsError(target)
with np.load(parent) as a:base={k:a[k] for k in a.files}
with np.load(OUT/(parent_name+'.centroidal_cache.npz')) as a:cache={k:a[k] for k in a.files}
assert cache['parent_sha256'].item()==hashlib.sha256(parent.read_bytes()).hexdigest()
with np.load(OUT/'reference_com_cache.npz') as a:contacts=a['contacts'];times=a['time']
source=np.load(ROOT/'acc_data/g1_tracking_contacts_20261123/eight_verify.control_reference.npz')['source_q']
assert hashlib.sha256(source.tobytes()).hexdigest()==base['source_sha256'].item()
names=np.load(OUT/'ten_verify.npz')['final_joint_order'].tolist()
urdf=str(ROOT/'src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf')
model=CentroidalGeometry(urdf,names);legs=LegKinematics(urdf,names)
n=len(source);D2=sparse.load_npz(OUT/'reference_com_D2.npz')
D3=sparse.diags(1/np.diff(times[1:-1]))@sparse.diags([-np.ones(n-3),np.ones(n-3)],[0,1],shape=(n-3,n-2))@D2
S2,S3=D2/1000.,D3/1000.;P0=sparse.eye(n)+4*S2.T@S2+.0004*S3.T@S3
P=sparse.block_diag([P0]*2,format='csc');linear=np.zeros(2*n)
low=np.full(2*n,-bound);high=np.full(2*n,bound)
for j in (0,1):low[j*n:j*n+641]=high[j*n:j*n+641]=0.;low[j*n+1980:(j+1)*n]=high[j*n+1980:(j+1)*n]=0.
A=[sparse.eye(2*n)];lower=[low];upper=[high]
corners=np.array([[.12,.025,-.035],[.12,-.025,-.035],[-.05,.025,-.035],[-.05,-.025,-.035]])
C=cache['com'];acc=cache['acceleration'];H=cache['momentum_rate'];fz=model.mass*(9.81+acc[:,2])
coeff=[];rr=[];cc=[];rhs=[];hull_frame=[]
for k in range(660,1961):
    if not contacts[k].any():continue
    if fz[k]<=0:raise ValueError(('Parent requires negative ground normal force',k))
    points=np.concatenate([(cache['feet_rotation'][k,s]@corners.T).T+cache['feet_position'][k,s] for s in range(2) if contacts[k,s]])[:,:2]
    hull=ConvexHull(points)
    cop=C[k,:2]-C[k,2]*model.mass*acc[k,:2]/fz[k]+np.array([-H[k,1],H[k,0]])/fz[k]
    operator=(-C[k,2]*model.mass/fz[k]*D2.getrow(k-1)).tolil();operator[0,k]+=1.;operator=operator.tocsr()/1000.
    for nx,ny,b in hull.equations:
        row=len(rhs)
        for axis,v in enumerate((nx,ny)):
            rr.extend([row]*len(operator.indices));cc.extend((operator.indices+axis*n).tolist());coeff.extend((v*operator.data).tolist())
        rhs.append(-float(np.dot([nx,ny],cop)+b));hull_frame.append(k)
copA=sparse.csr_matrix((coeff,(rr,cc)),shape=(len(rhs),2*n));A.append(copA);lower.append(np.full(len(rhs),-np.inf));upper.append(np.asarray(rhs))
rows=np.arange(659,1960);S=S2[rows]
for angle in np.arange(8)*np.pi/4:
    x,y=np.cos(angle),np.sin(angle);m=.55*np.cos(np.pi/8)
    A.append(sparse.hstack([x*S,y*S]));lower.append(np.full(len(rows),-np.inf));upper.append(m*(9.81+acc[rows+1,2])-x*acc[rows+1,0]-y*acc[rows+1,1])
solver=osqp.OSQP();solver.setup(P=P,q=linear,A=sparse.vstack(A,format='csc'),l=np.concatenate(lower),u=np.concatenate(upper),
    eps_abs=2e-6,eps_rel=2e-6,max_iter=200000,polish=True,verbose=False)
fit=solver.solve();report=dict(method='Offline linearized CoP fit with frozen reference angular momentum',xy_bound_mm=bound,
    parent_sha256=cache['parent_sha256'].item(),source_sha256=base['source_sha256'].item(),offline_solver_status=fit.info.status,
    full_tracking_passed=False,preflight_passed=False)
print('CoP fit',fit.info.status,flush=True)
if fit.info.status_val not in (1,2):
    target.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n');sys.exit(1)
delta=np.zeros((n,3));delta[:,:2]=fit.x.reshape(2,n).T/1000.;delta[:641]=0.;delta[1980:]=0.
report['linearized_cop_constraint_max_residual_m']=float(np.max(copA@fit.x-np.asarray(rhs)))
if report['linearized_cop_constraint_max_residual_m']>1e-4:raise ValueError('Independent CoP fit check failed')
out,ik=fit_com_reference(base,source,delta,legs,model,max_iterations=200)
report.update(ik);report['recomputed_momentum_and_runtime_spline_still_require_audit']=True
target.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
if report['preflight_passed']:
    with target.open('xb') as stream:np.savez_compressed(stream,**dict(base,control_q=out))
print(report,flush=True)
