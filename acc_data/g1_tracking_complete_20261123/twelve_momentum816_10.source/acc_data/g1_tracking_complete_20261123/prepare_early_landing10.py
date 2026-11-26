"""Advance the actual tenth touchdown endpoint to improve left-foot release."""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
from scipy.optimize import lsq_linear
from scipy.spatial.transform import Rotation,Slerp
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'acc_data/amp_walk_20261103/feedback_followup'))
from numpy_leg_kinematics import LegKinematics
parent=OUT/(sys.argv[1]+'.npz') if len(sys.argv)>1 else OUT/'prepared_combined_root_y50.npz'
with np.load(parent) as data:base={k:data[k] for k in data.files}
source=np.load(ROOT/'acc_data/g1_tracking_contacts_20261123/eight_verify.control_reference.npz')['source_q']
assert hashlib.sha256(source.tobytes()).hexdigest()==base['source_sha256'].item()
names=np.load(OUT/'ten_verify.npz')['final_joint_order'].tolist();legs=LegKinematics(str(ROOT/'src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf'),names)
start,land,end=768,802,822
poses=[legs.evaluate(base['control_q'][k],1)[:2] for k in range(start,end+1)]
positions=np.array([p for p,R in poses]);rotations=Slerp(np.arange(start,end+1),Rotation.from_matrix([R for p,R in poses]))
def smooth(u):
    u=np.clip(u,0,1);return u**3*(10-15*u+6*u*u)
for lead in (4,6):
    target=OUT/(f'{parent.stem}_early10_f{lead}.npz' if len(sys.argv)>1 else f'prepared_early_landing10_f{lead}.npz')
    if target.exists():raise FileExistsError(target)
    stop=land-lead;join=stop-8;out=base['control_q'].copy();max_pos=max_rot=max_shift=0.;phases=[]
    for k in range(start+1,end):
        if k<join:f=k+lead*smooth((k-start)/8)
        elif k<stop:
            u=(k-join)/8;f=land-8+8*(u+4*u**3-7*u**4+3*u**5)
        else:f=float(land)
        phases.append(f);j=min(len(positions)-2,int(f)-start);alpha=f-(start+j)
        p=(1-alpha)*positions[j]+alpha*positions[j+1];R=rotations(f).as_matrix()
        original=base['control_q'][k].astype(float);oldp,oldR,_=legs.evaluate(original,1)
        if k>land:
            blend=smooth((k-land)/(end-land));p+=blend*(oldp-p)
            R=Rotation.from_rotvec(blend*Rotation.from_matrix(oldR@R.T).as_rotvec()).as_matrix()@R
        q=original.copy()
        for iteration in range(300):
            pos,rot,J=legs.evaluate(q,1);error=np.r_[p-pos,Rotation.from_matrix(R@rot.T).as_rotvec()]
            if np.max(abs(error))<3e-7:break
            lo=np.maximum(-.06,legs.lower[6:]-q[13:19]);hi=np.minimum(.06,legs.upper[6:]-q[13:19])
            step=lsq_linear(np.vstack((J,1e-4*np.eye(6))),np.r_[error,np.zeros(6)],bounds=(lo,hi),tol=1e-10).x;q[13:19]+=step
        out[k]=q;pos,rot,_=legs.evaluate(out[k],1)
        max_pos=max(max_pos,float(np.linalg.norm(p-pos)));max_rot=max(max_rot,float(Rotation.from_matrix(R@rot.T).magnitude()));max_shift=max(max_shift,float(np.linalg.norm(p-oldp)))
    np.testing.assert_array_equal(out[:start+1],base['control_q'][:start+1]);np.testing.assert_array_equal(out[end:],base['control_q'][end:]);assert np.min(np.diff(phases))>=-1e-10
    report=dict(method='True tenth touchdown advance with phase stopping and pose rejoining',advance_frames=lead,target_touchdown_frame=stop,
        original_touchdown_frame=land,rejoin_frame=end,maximum_foot_position_residual_m=max_pos,maximum_foot_rotation_residual_rad=max_rot,
        maximum_foot_target_change_m=max_shift,peak_source_joint_rmse_rad=float(np.sqrt(np.mean((out[:,7:]-source[:,7:])**2,axis=1)).max()),
        parent_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),source_sha256=base['source_sha256'].item(),full_tracking_passed=False)
    report['preflight_passed']=max_pos<2e-5 and max_rot<2e-4 and report['peak_source_joint_rmse_rad']<.19
    target.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
    if report['preflight_passed']:
        with target.open('xb') as stream:np.savez_compressed(stream,**dict(base,control_q=out))
    print(target.name,report,flush=True)
