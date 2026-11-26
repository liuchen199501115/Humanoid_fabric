"""Offline quaternion smoothing with foot-preserving bounded IK."""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import lsq_linear
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'acc_data/amp_walk_20261103/feedback_followup'))
from numpy_leg_kinematics import LegKinematics
from numpy_centroidal_geometry import CentroidalGeometry

label=sys.argv[1];sigma=float(sys.argv[2]);parent_name=sys.argv[3]
end_frame=int(sys.argv[4]) if len(sys.argv)>4 else 1980
start_frame=(int(sys.argv[sys.argv.index('--start-frame')+1]) if '--start-frame' in sys.argv else 640)
ramp_frames=(int(sys.argv[sys.argv.index('--ramp-frames')+1]) if '--ramp-frames' in sys.argv else 60)
assert 640<=start_frame<end_frame<=1980 and 0<ramp_frames<=(end_frame-start_frame)/2
com_weight=(float(sys.argv[sys.argv.index('--com-ik-weight')+1]) if '--com-ik-weight' in sys.argv else .01)
assert np.isfinite(com_weight) and 0<com_weight<=.01
parent=OUT/(parent_name+'.npz');target=OUT/('prepared_root_smooth_'+label+'.npz')
if target.exists():raise FileExistsError(target)
with np.load(parent) as a:base={k:a[k] for k in a.files}
source=np.load(ROOT/'acc_data/g1_tracking_contacts_20261123/eight_verify.control_reference.npz')['source_q']
assert hashlib.sha256(source.tobytes()).hexdigest()==base['source_sha256'].item()
names=np.load(OUT/'ten_verify.npz')['final_joint_order'].tolist()
urdf=str(ROOT/'src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf');legs=LegKinematics(urdf,names);model=CentroidalGeometry(urdf,names)
original_q=base['control_q'];quats=original_q[:,3:7].astype(float).copy()
for k in range(1,len(quats)):
    if np.dot(quats[k],quats[k-1])<0:quats[k]*=-1
smooth=gaussian_filter1d(quats,sigma,axis=0,mode='nearest');smooth/=np.linalg.norm(smooth,axis=1)[:,None]
change=(Rotation.from_quat(smooth)*Rotation.from_quat(quats).inv()).as_rotvec()
phase=np.minimum(np.clip((np.arange(len(quats))-start_frame)/ramp_frames,0,1),np.clip((end_frame-np.arange(len(quats)))/ramp_frames,0,1))
blend=phase**3*(10-15*phase+6*phase**2)
desired_rot=(Rotation.from_rotvec(change*blend[:,None])*Rotation.from_quat(quats)).as_quat()
out=original_q.copy();max_pos=max_rot=max_com=0.;worst=None
for k in range(start_frame+1,end_frame):
    original=original_q[k].astype(float);q=original.copy();q[3:7]=desired_rot[k]
    desired_com=model.evaluate(original)[0];goals=[legs.evaluate(original,s)[:2] for s in (0,1)]
    # Feet are primary in this offline IK; COM displacement is minimized,
    # not falsely certified as exactly preserved when leg limits bind.
    for iteration in range(180):
        J=np.zeros((15,15));error=np.zeros(15)
        for side in (0,1):
            pos,rot,legJ=legs.evaluate(q,side);p,R=goals[side];sl=slice(6*side,6*(side+1))
            error[sl]=np.r_[p-pos,Rotation.from_matrix(R@rot.T).as_rotvec()]
            J[sl,:3]=np.vstack((np.eye(3),np.zeros((3,3))));J[sl,3+6*side:9+6*side]=legJ
        com,Jc,_=model.evaluate(q);error[12:]=desired_com-com;J[12:,:3]=np.eye(3);J[12:,3:]=Jc[:,:12]
        if np.max(abs(error[:12]))<5e-7 and np.max(abs(error[12:]))<5e-7:break
        weights=np.r_[np.ones(12),np.full(3,com_weight)]
        dr=q[:3]-original[:3]
        lo=np.r_[np.maximum(-.004,-.04-dr),np.maximum(-.06,legs.lower-q[7:19])]
        hi=np.r_[np.minimum(.004,.04-dr),np.minimum(.06,legs.upper-q[7:19])]
        regularizer=np.diag(np.r_[np.full(3,.0001),np.full(12,.00001)])
        step=lsq_linear(np.vstack((weights[:,None]*J,regularizer)),np.r_[weights*error,np.zeros(15)],bounds=(lo,hi),tol=1e-10,max_iter=100).x
        q[:3]+=step[:3];q[7:19]+=step[3:]
        if np.linalg.norm(step)<1e-9:break
    out[k]=q
    for side in (0,1):
        p,R,_=legs.evaluate(out[k],side);pe=float(np.linalg.norm(p-goals[side][0]));re=float(Rotation.from_matrix(R@goals[side][1].T).magnitude())
        if max(pe,re)>max(max_pos,max_rot):worst=k
        max_pos=max(max_pos,pe);max_rot=max(max_rot,re)
    max_com=max(max_com,float(np.linalg.norm(model.evaluate(out[k])[0]-desired_com)))
    if k%200==0:print('IK frame',k,'max feet',max_pos,max_rot,'max com',max_com,flush=True)
np.testing.assert_array_equal(out[:start_frame+1],original_q[:start_frame+1]);np.testing.assert_array_equal(out[end_frame:],original_q[end_frame:])
report=dict(method='Offline quaternion smoothing and bounded foot-preserving IK',sigma_frames=sigma,
    correction_end_frame=end_frame,
    correction_start_frame=start_frame,correction_ramp_frames=ramp_frames,
    com_ik_weight=com_weight,
    maximum_foot_position_residual_m=max_pos,maximum_foot_rotation_residual_rad=max_rot,maximum_com_change_m=max_com,
    maximum_added_root_rotation_rad=float(np.linalg.norm(change*blend[:,None],axis=1).max()),worst_frame=worst,
    peak_source_joint_rmse_rad=float(np.sqrt(np.mean((out[:,7:]-source[:,7:])**2,axis=1)).max()),
    peak_source_root_displacement_m=float(np.linalg.norm(out[:,:3]-source[:,:3],axis=1).max()),
    parent_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),source_sha256=base['source_sha256'].item(),full_tracking_passed=False)
report['preflight_passed']=max_pos<2e-5 and max_rot<2e-4 and max_com<.01 and report['maximum_added_root_rotation_rad']<.15 and report['peak_source_joint_rmse_rad']<.19 and report['peak_source_root_displacement_m']<.15
target.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
if report['preflight_passed']:
    with target.open('xb') as stream:np.savez_compressed(stream,**dict(base,control_q=out))
print(report,flush=True)
