"""Refine a saved offline fit with exact leg IK and an optional smooth root dip."""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
from scipy.optimize import lsq_linear
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'acc_data/amp_walk_20261103/feedback_followup'))
from numpy_leg_kinematics import LegKinematics
name=sys.argv[1];dip=float(sys.argv[2]);label=sys.argv[3]
report0=json.loads((OUT/(name+'.json')).read_text())
assert report0['solver_status'] in ('solved','solved inaccurate')
parent=OUT/'prepared_combined_root_y50.npz'
assert hashlib.sha256(parent.read_bytes()).hexdigest()==report0['parent_sha256']
target=OUT/('prepared_root_upper_refined_'+label+'.npz')
if target.exists():raise FileExistsError(target)
with np.load(parent) as data:base={k:data[k] for k in data.files}
source=np.load(ROOT/'acc_data/g1_tracking_contacts_20261123/eight_verify.control_reference.npz')['source_q']
assert hashlib.sha256(source.tobytes()).hexdigest()==base['source_sha256'].item()
names=np.load(OUT/'ten_verify.npz')['final_joint_order'].tolist()
legs=LegKinematics(str(ROOT/'src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf'),names)
nodes=np.unique(np.r_[np.arange(0,len(source),6),len(source)-1]);knots=np.r_[np.full(4,nodes[0]),nodes[2:-2],np.full(4,nodes[-1])]
greville=np.array([np.mean(knots[i+1:i+4]) for i in range(len(nodes))]);select=np.flatnonzero((greville>=672)&(greville<=1008))
node_maps=[]
for k in greville[select].astype(int):
    N=np.zeros((32,20));N[:3,:3]=report0['root_axis_bound_m']*np.eye(3);N[15:,3:]=report0['upper_axis_bound_rad']*np.eye(17)
    for side in (0,1):
        Jf=legs.evaluate(base['control_q'][k],side)[2]
        N[3+6*side:9+6*side,:3]=np.linalg.lstsq(Jf,np.vstack((-report0['root_axis_bound_m']*np.eye(3),np.zeros((3,3)))),rcond=1e-8)[0]
    node_maps.append(N)
coeff=np.load(OUT/(name+'.solver_iterate.npz'))['x'][:len(select)*20].reshape(len(select),20)
global_coeff=np.zeros((len(nodes),32));global_coeff[select]=np.einsum('nij,nj->ni',node_maps,coeff)
correction=np.stack([np.interp(np.arange(len(source)),greville,global_coeff[:,j]) for j in range(32)],axis=1)
def smooth(u):
    u=np.clip(u,0,1);return u**3*(10-15*u+6*u*u)
frames=np.arange(len(source));weight=smooth((frames-660)/24)*(1-smooth((frames-984)/36))
correction[:,2]-=dip*weight
out=base['control_q'].copy();out[:,np.r_[np.arange(3),np.arange(7,36)]]+=correction
residuals=[]
for k in np.flatnonzero(np.max(abs(correction),axis=1)>1e-10):
    q=out[k].astype(float)
    for side in (0,1):
        p,R,_=legs.evaluate(base['control_q'][k],side);sl=slice(7+6*side,13+6*side)
        lower=legs.lower[6*side:6*side+6];upper=legs.upper[6*side:6*side+6]
        q[sl]=np.clip(q[sl],lower+1e-8,upper-1e-8)
        for iteration in range(600):
            pos,rot,J=legs.evaluate(q,side);error=np.r_[p-pos,Rotation.from_matrix(R@rot.T).as_rotvec()]
            if np.max(abs(error))<3e-7:break
            lo=np.maximum(-.04,lower-q[sl]);hi=np.minimum(.04,upper-q[sl])
            step=lsq_linear(np.vstack((J,1e-5*np.eye(6))),np.r_[error,np.zeros(6)],bounds=(lo,hi),tol=1e-10).x
            q[sl]+=step
            if np.linalg.norm(step)<1e-10:break
        out[k]=q;pos,rot,_=legs.evaluate(out[k],side)
        residuals.append([k,side,np.linalg.norm(pos-p),Rotation.from_matrix(R@rot.T).magnitude()])
residuals=np.asarray(residuals);max_pos=float(residuals[:,2].max());max_rot=float(residuals[:,3].max());worst=residuals[np.argmax(residuals[:,2])]
report=dict(method='Exact bounded IK refinement of a saved offline dynamics fit',fit=name,
    fit_report_sha256=hashlib.sha256((OUT/(name+'.json')).read_bytes()).hexdigest(),additional_root_dip_m=dip,
    maximum_foot_position_residual_m=max_pos,maximum_foot_rotation_residual_rad=max_rot,worst_frame=int(worst[0]),worst_foot=int(worst[1]),
    peak_source_joint_rmse_rad=float(np.sqrt(np.mean((out[:,7:]-source[:,7:])**2,axis=1)).max()),
    peak_source_root_displacement_m=float(np.linalg.norm(out[:,:3]-source[:,:3],axis=1).max()),
    source_sha256=base['source_sha256'].item(),parent_sha256=report0['parent_sha256'],full_tracking_passed=False)
report['preflight_passed']=max_pos<2e-5 and max_rot<2e-4 and report['peak_source_joint_rmse_rad']<.19 and report['peak_source_root_displacement_m']<.15
target.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
if report['preflight_passed']:
    with target.open('xb') as stream:np.savez_compressed(stream,**dict(base,control_q=out))
else:np.savez_compressed(target.with_suffix('.ik_diagnostic.npz'),candidate_q=out,residuals=residuals)
print(report,flush=True)
