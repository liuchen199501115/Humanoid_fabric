"""Bounded offline IK for a whole-clip COM correction, preserving both feet."""
import numpy as np
from scipy.optimize import lsq_linear
from scipy.spatial.transform import Rotation


def fit_com_reference(base, source, delta, legs, model, *, rotation_weight=.05, fixed_root=False, max_iterations=100, foot_offsets=None, foot_rotations=None):
    report = {}
    def skew(v):
        x,y,z=v; return np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])
    out=base['control_q'].copy(); max_pos=max_rot=max_com=max_rootrot=0.;worst=None
    for k in range(641,1980):
        if np.max(abs(delta[k]))<1e-9 and (foot_offsets is None or np.max(abs(foot_offsets[k]))<1e-9):continue
        original=base['control_q'][k].astype(float); desired=model.evaluate(original)[0]+delta[k]
        goals=[legs.evaluate(original,s)[:2] for s in (0,1)]
        if foot_offsets is not None:
            goals=[(p+foot_offsets[k,s],R) for s,(p,R) in enumerate(goals)]
        if foot_rotations is not None:
            goals=[(p,foot_rotations[k,s]) for s,(p,R) in enumerate(goals)]
        q=original.copy(); q[:3]+=delta[k]; root0=Rotation.from_quat(original[3:7])
        for iteration in range(max_iterations):
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
        runtime_spline_acceleration_still_requires_audit=True)
    report['preflight_passed']=(max_pos<2e-5 and max_rot<2e-4 and max_com<2e-5 and max_rootrot<.1
        and report['peak_source_joint_rmse_rad']<.19 and report['peak_source_root_displacement_m']<.15)
    return out, report
