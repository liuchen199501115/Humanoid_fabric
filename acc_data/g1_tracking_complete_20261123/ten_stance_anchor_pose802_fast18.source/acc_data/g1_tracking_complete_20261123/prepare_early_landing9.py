"""Truly advance ninth touchdown, with C2 phase stop and pose rejoining.

Previous bell-shaped phase offsets returned to zero before the original landing.
Here the foot reaches the original landing pose early and holds it. Raw-source
flight, landing, hold, and full-body fidelity gates remain unchanged at runtime.
"""
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
parent=OUT/'prepared_combined_root_y50.npz'
with np.load(parent) as a:base={k:a[k] for k in a.files}
source=np.load(ROOT/'acc_data/g1_tracking_contacts_20261123/eight_verify.control_reference.npz')['source_q']
assert hashlib.sha256(source.tobytes()).hexdigest()==base['source_sha256'].item()
names=np.load(OUT/'ten_verify.npz')['final_joint_order'].tolist();legs=LegKinematics(str(ROOT/'src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf'),names)
start,land,end=730,762,782
poses=[legs.evaluate(base['control_q'][k],0)[:2] for k in range(start,end+1)]
positions=np.array([p for p,R in poses]);rotations=Slerp(np.arange(start,end+1),Rotation.from_matrix([R for p,R in poses]))
def smooth(u):
    u=np.clip(u,0,1);return u**3*(10-15*u+6*u*u)
for lead in [8,12]:
    target=OUT/f'prepared_early_landing9_f{lead}.npz'
    if target.exists():raise FileExistsError(target)
    stop=land-lead;join=stop-12;out=base['control_q'].copy();max_pos=max_rot=max_shift=0.;phases=[]
    for k in range(start+1,end):
        if k<join:f=k+lead*smooth((k-start)/8)
        elif k<stop:
            u=(k-join)/12;f=land-12+12*(u+4*u**3-7*u**4+3*u**5)
        else:f=float(land)
        phases.append(f)
        j=min(len(positions)-2,int(f)-start);alpha=f-(start+j)
        p=(1-alpha)*positions[j]+alpha*positions[j+1];R=rotations(f).as_matrix()
        original=base['control_q'][k].astype(float);oldp,oldR,_=legs.evaluate(original,0)
        if k>land:
            blend=smooth((k-land)/(end-land));p=p+blend*(oldp-p)
            R=Rotation.from_rotvec(blend*Rotation.from_matrix(oldR@R.T).as_rotvec()).as_matrix()@R
        q=original.copy()
        for iteration in range(160):
            pos,rot,J=legs.evaluate(q,0);error=np.r_[p-pos,Rotation.from_matrix(R@rot.T).as_rotvec()]
            if np.max(abs(error))<5e-7:break
            lo=np.maximum(-.06,legs.lower[:6]-q[7:13]);hi=np.minimum(.06,legs.upper[:6]-q[7:13])
            step=lsq_linear(np.vstack((J,.001*np.eye(6))),np.r_[error,np.zeros(6)],bounds=(lo,hi),tol=1e-10,max_iter=100).x;q[7:13]+=step
        out[k]=q;pos,rot,_=legs.evaluate(out[k],0)
        max_pos=max(max_pos,float(np.linalg.norm(p-pos)));max_rot=max(max_rot,float(Rotation.from_matrix(R@rot.T).magnitude()))
        max_shift=max(max_shift,float(np.linalg.norm(p-oldp)))
    np.testing.assert_array_equal(out[:start+1],base['control_q'][:start+1]);np.testing.assert_array_equal(out[end:],base['control_q'][end:])
    assert np.min(np.diff(phases))>=-1e-10
    report=dict(method='Early touchdown endpoint with C2 phase stopping and original pose rejoining',advance_frames=lead,
        target_touchdown_frame=stop,original_touchdown_frame=land,rejoin_frame=end,
        maximum_foot_position_residual_m=max_pos,maximum_foot_rotation_residual_rad=max_rot,maximum_foot_target_change_m=max_shift,
        peak_source_joint_rmse_rad=float(np.sqrt(np.mean((out[:,7:]-source[:,7:])**2,axis=1)).max()),
        parent_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),source_sha256=base['source_sha256'].item(),full_tracking_passed=False)
    report['preflight_passed']=max_pos<2e-5 and max_rot<2e-4 and report['peak_source_joint_rmse_rad']<.19
    target.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
    if report['preflight_passed']:
        with target.open('xb') as stream:np.savez_compressed(stream,**dict(base,control_q=out))
        for swingend in [802,812]:
            right_parent=OUT/f'prepared_swing10_smooth_{swingend}.npz';combined=out.copy();combined[:,13:19]=np.load(right_parent)['control_q'][:,13:19]
            joint_rmse=float(np.sqrt(np.mean((combined[:,7:]-source[:,7:])**2,axis=1)).max())
            name=OUT/f'prepared_early9_f{lead}_swing10_{swingend}.npz'
            if name.exists():raise FileExistsError(name)
            combined_report=dict(report,combined_right_swing_end=swingend,right_parent_sha256=hashlib.sha256(right_parent.read_bytes()).hexdigest(),peak_source_joint_rmse_rad=joint_rmse,preflight_passed=joint_rmse<.19)
            name.with_suffix('.json').write_text(json.dumps(combined_report,indent=2)+'\n')
            if combined_report['preflight_passed']:
                with name.open('xb') as stream:np.savez_compressed(stream,**dict(base,control_q=combined))
            print(name.name,combined_report,flush=True)
    print(target.name,report,flush=True)
