"""Offline tenth-swing shaping; original source and all acceptance stay fixed."""
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

parent=OUT/'prepared_combined_root_y50.npz'
with np.load(parent) as a:base={k:a[k] for k in a.files}
source=np.load(ROOT/'acc_data/g1_tracking_contacts_20261123/eight_verify.control_reference.npz')['source_q']
assert hashlib.sha256(source.tobytes()).hexdigest()==base['source_sha256'].item()
names=np.load(OUT/'ten_verify.npz')['final_joint_order'].tolist()
legs=LegKinematics(str(ROOT/'src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf'),names)
def smooth(u):
    u=np.clip(u,0,1);return u**3*(10-15*u+6*u*u)
for end in [802,812]:
    start=762;target=OUT/f'prepared_swing10_smooth_{end}.npz'
    if target.exists():raise FileExistsError(target)
    p0,R0,_=legs.evaluate(base['control_q'][start],1);p1,R1,_=legs.evaluate(base['control_q'][end],1)
    dr=Rotation.from_matrix(R1@R0.T).as_rotvec();out=base['control_q'].copy();max_pos=max_rot=max_change=0.;records=[]
    for k in range(start+1,end):
        original=base['control_q'][k].astype(float);oldp,oldR,_=legs.evaluate(original,1)
        u=(k-start)/(end-start);h=smooth(u);blend=min(smooth((k-start)/8),smooth((end-k)/8))
        p=(1-h)*p0+h*p1;p[2]+=.04*64*u**3*(1-u)**3
        R=Rotation.from_rotvec(h*dr).as_matrix()@R0
        p=oldp+blend*(p-oldp);R=Rotation.from_rotvec(blend*Rotation.from_matrix(R@oldR.T).as_rotvec()).as_matrix()@oldR
        q=original.copy()
        for iteration in range(100):
            pos,rot,J=legs.evaluate(q,1);error=np.r_[p-pos,Rotation.from_matrix(R@rot.T).as_rotvec()]
            if np.max(abs(error))<5e-7:break
            lo=np.maximum(-.06,legs.lower[6:]-q[13:19]);hi=np.minimum(.06,legs.upper[6:]-q[13:19])
            step=lsq_linear(np.vstack((J,.001*np.eye(6))),np.r_[error,np.zeros(6)],bounds=(lo,hi),tol=1e-10,max_iter=100).x;q[13:19]+=step
        out[k]=q;pos,rot,_=legs.evaluate(out[k],1)
        max_pos=max(max_pos,float(np.linalg.norm(p-pos)));max_rot=max(max_rot,float(Rotation.from_matrix(R@rot.T).magnitude()))
        max_change=max(max_change,float(np.linalg.norm(p-oldp)));records.append(dict(frame=k,target_position=p.tolist()))
    np.testing.assert_array_equal(out[:start+1],base['control_q'][:start+1]);np.testing.assert_array_equal(out[end:],base['control_q'][end:])
    np.testing.assert_array_equal(out[:,:13],base['control_q'][:,:13]);np.testing.assert_array_equal(out[:,19:],base['control_q'][:,19:])
    report=dict(method='Offline minimum-jerk swing shape blended into original endpoints',start_frame=start,end_frame=end,
        peak_foot_position_residual_m=max_pos,peak_foot_rotation_residual_rad=max_rot,peak_foot_target_change_m=max_change,
        peak_source_joint_rmse_rad=float(np.sqrt(np.mean((out[:,7:]-source[:,7:])**2,axis=1)).max()),
        parent_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),source_sha256=base['source_sha256'].item(),frames=records,full_tracking_passed=False)
    report['preflight_passed']=max_pos<2e-5 and max_rot<2e-4 and report['peak_source_joint_rmse_rad']<.19
    target.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
    if report['preflight_passed']:
        with target.open('xb') as stream:np.savez_compressed(stream,**dict(base,control_q=out))
    print({k:v for k,v in report.items() if k!='frames'},flush=True)
