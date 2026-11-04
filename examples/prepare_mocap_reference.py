"""Build and audit an explicit C2 candidate; does not execute physics."""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from fabrics_sim.reference.interpolation import MotionInterpolator
from inspect_mocap_reference import urdf_fk
from test_g1_two_tasks import G1_JOINT_NAMES,load_motion_trajectory,load_motion_contact_sequence


from fabrics_sim.reference.preprocessing import scalar_derivative_peaks


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('motion',type=Path)
    p.add_argument('--contacts-pkl',type=Path)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--joint-acceleration-cap',type=float,required=True,
                   help='Explicit planning cap in rad/s^2; NOT a certified actuator limit')
    args=p.parse_args()
    if not np.isfinite(args.joint_acceleration_cap) or args.joint_acceleration_cap<=0:
        raise ValueError('Expected positive finite planning acceleration cap')
    q,fps=load_motion_trajectory(args.motion)
    if len(q)<4:raise ValueError('At least four frames required')
    contact_path=args.contacts_pkl or args.motion
    if args.contacts_pkl:
        contact_q,contact_fps=load_motion_trajectory(contact_path)
        if contact_fps!=fps or contact_q.shape!=q.shape or not np.allclose(contact_q.numpy(),q.numpy(),rtol=0,atol=1e-6):
            raise ValueError('Contact companion does not match original poses/fps')
    contacts=load_motion_contact_sequence(contact_path)
    sampler=MotionInterpolator(q,method='bspline')
    c=sampler.coefficients.double().numpy()
    vpeak,apeak=scalar_derivative_peaks(c[:,:,7:]);vpeak*=fps;apeak*=fps**2
    urdf=Path('src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf')
    limits={x.attrib['name']:x.find('limit') for x in ET.parse(urdf).getroot().findall('joint')}
    vmax=np.array([float(limits[n].attrib['velocity']) for n in G1_JOINT_NAMES])
    lower=np.array([float(limits[n].attrib['lower']) for n in G1_JOINT_NAMES])
    upper=np.array([float(limits[n].attrib['upper']) for n in G1_JOINT_NAMES])
    if np.any(q.numpy()[:,7:]<lower-1e-6) or np.any(q.numpy()[:,7:]>upper+1e-6):
        raise ValueError('Source control points violate joint limits')
    speed=min(1.,float(np.min(vmax/np.maximum(vpeak,1e-12))),
              float(np.sqrt(args.joint_acceleration_cap/max(float(apeak.max()),1e-12))))
    rows=[sampler.kinematics(i,fps*speed) for i in range(len(q))]
    smooth=np.concatenate([r[0].numpy() for r in rows]);v=np.concatenate([r[1].numpy() for r in rows]);a=np.concatenate([r[2].numpy() for r in rows])
    if not all(np.isfinite(x).all() for x in (smooth,v,a)):raise ValueError('Nonfinite candidate')
    original,original_com,_=urdf_fk(q.numpy(),G1_JOINT_NAMES,urdf)
    filtered,filtered_com,_=urdf_fk(smooth,G1_JOINT_NAMES,urdf)
    feet_error=np.stack([np.linalg.norm(original[s+'_ankle_roll_link'][0]-filtered[s+'_ankle_roll_link'][0],axis=1) for s in ('left','right')],axis=1)
    errors=np.abs(smooth[:,7:]-q.numpy()[:,7:])
    report=dict(scope='Offline C2 candidate; not WBC feasibility or tracking certification',
        full_tracking_passed=False,physical_simulation_run=False,motion=str(args.motion),
        method='clamped cubic B-spline using source control points; normalized quaternion',
        contact_source=str(contact_path),contact_labels_present=contacts is not None,
        contact_policy='Labels retain original frame indices; times scale uniformly; geometry NOT revalidated',
        original_fps=fps,playback_speed=speed,sample_span_s=(len(q)-1)/(fps*speed),
        joint_order_status='Uses existing loader; absent source names remain an unresolved assumption',
        planning_acceleration_cap_rad_s2=args.joint_acceleration_cap,
        original_speed_smoothed_peak_joint_velocity_rad_s=float(vpeak.max()),
        original_speed_smoothed_peak_joint_acceleration_rad_s2=float(apeak.max()),
        retimed_peak_joint_velocity_rad_s=float(vpeak.max()*speed),
        retimed_peak_joint_acceleration_rad_s2=float(apeak.max()*speed**2),
        joint_position_bounds_preserved='Convex hull of in-limit control points; no clipping',
        peak_joint_change_rad=float(errors.max()),peak_frame_joint_rmse_rad=float(np.sqrt(np.mean(errors**2,axis=1)).max()),
        peak_foot_position_change_m=feet_error.max(0).tolist(),
        peak_com_position_change_m=float(np.linalg.norm(filtered_com-original_com,axis=1).max()),
        warnings=['Foot trajectories change; contact geometry and collision need re-audit',
                  'Velocity and acceleration scalar caps do not certify braking/torque/contact feasibility',
                  'Endpoints may have nonzero velocity; start/stop and time-ramp boundary must be planned',
                  'NPZ contains samples plus original spline coefficients; resampling samples is not the same spline'])
    args.output_dir.mkdir(parents=True,exist_ok=False)
    (args.output_dir/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    np.savez_compressed(args.output_dir/'candidate.npz',q=smooth,v=v,a=a,source_q=q.numpy(),
        source_frame=np.arange(len(q)),time=np.arange(len(q))/(fps*speed),fps=fps,
        speed=speed,coefficients=c,joint_names=np.array(G1_JOINT_NAMES),foot_error=feet_error,
        **({} if contacts is None else dict(contact_sequence=contacts.numpy(),contact_names=np.array(['left_foot','right_foot']))))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(3,1,figsize=(12,8),layout='constrained');t=np.arange(len(q))/fps
    ax[0].plot(t,feet_error);ax[0].set_ylabel('Foot deviation [m]');ax[0].legend(['left','right'])
    ax[1].plot(t,np.max(np.abs(v[:,6:]),axis=1));ax[1].set_ylabel('Joint speed [rad/s]')
    ax[2].plot(t,np.max(np.abs(a[:,6:]),axis=1));ax[2].set_ylabel('Joint acc. [rad/s²]');ax[2].set_xlabel('Original source time [s]')
    fig.suptitle('C2 + uniform retiming candidate; offline only');fig.savefig(args.output_dir/'comparison.png',dpi=130)
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
