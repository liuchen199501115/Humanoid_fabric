"""Adapt a source step to the measured support foot on level ground."""
import numpy as np
from scipy.spatial.transform import Rotation


def source_toe_pivot_pose(stance_pose, source_pose):
    """Plan source pitch about the stance's fixed material toe midpoint.

    This only supplies reference geometry; measured contact must independently
    authorize a toe constraint. It does not move a live foot or extend a sole.
    """
    stance,source=(np.asarray(p,dtype=float) for p in (stance_pose,source_pose))
    if any(p.shape!=(7,) or not np.isfinite(p).all() for p in (stance,source)):
        raise ValueError('Expected finite stance/source poses [7]')
    old=Rotation.from_quat(stance[3:]);matrix=old.as_matrix()
    src=Rotation.from_quat(source[3:]).as_matrix()
    yaw=np.arctan2(matrix[1,0],matrix[0,0])
    pitch=max(0.,float(np.arctan2(-src[2,0],np.hypot(src[0,0],src[1,0]))))
    target=Rotation.from_euler('z',yaw)*Rotation.from_euler('y',pitch)
    toe=np.array([.12,0.,-.035])
    anchor=stance[:3]+old.apply(toe)
    return np.r_[anchor-target.apply(toe),target.as_quat()]


def source_com_landing_target(source_com, source_feet, stance_pose, landing_pose):
    """Retain source fore-aft COM placement relative to actual landing feet.

    Lateral COM remains centered for transfer. This is a geometric endpoint;
    the COM preview must still verify support and terminal capture constraints.
    """
    com,feet,stance,landing=(np.asarray(v,dtype=float) for v in
                             (source_com,source_feet,stance_pose,landing_pose))
    if (com.shape!=(3,) or feet.shape!=(2,7) or stance.shape!=(7,) or landing.shape!=(7,)
            or not all(np.isfinite(v).all() for v in (com,feet,stance,landing))):
        raise ValueError('Invalid COM landing geometry')
    forward=Rotation.from_quat(stance[3:]).as_matrix()[:2,0]
    norm=np.linalg.norm(forward)
    if norm<1e-6:
        raise ValueError('Stance heading has no horizontal component')
    forward/=norm
    midpoint=.5*(stance[:2]+landing[:2])
    return midpoint+forward*np.dot(com[:2]-feet[:,:2].mean(0),forward)


def source_grounded_step(source_swing):
    """Keep the source landing's world XY/yaw; flatten its sole on z=0.

    No stride clipping or alignment to measured support is applied. Full-pose
    fidelity still needs separate validation because sole roll/pitch is leveled.
    """
    pose=np.asarray(source_swing,dtype=float)
    if pose.shape!=(7,) or not np.isfinite(pose).all():
        raise ValueError('Expected a finite source pose [7]')
    rotation=Rotation.from_quat(pose[3:]).as_matrix()
    yaw=np.arctan2(rotation[1,0],rotation[0,0])
    return np.r_[pose[:2],.035,Rotation.from_euler('z',yaw).as_quat()]


def source_contact_step(source_swing):
    """Use the source swing endpoint's orientation and bounded sole grounding.

    The source swing tracker retains roll/pitch. Its contact/COM planner must
    therefore use this same pose, rather than silently planning a flat sole.
    """
    pose=np.asarray(source_swing,dtype=float)
    if pose.shape!=(7,) or not np.isfinite(pose).all():
        raise ValueError('Expected a finite source pose [7]')
    corners=np.array([[-.05,-.025,-.035],[-.05,.025,-.035],
                      [.12,-.025,-.035],[.12,.025,-.035]])
    clearance=float((Rotation.from_quat(pose[3:]).apply(corners)+pose[:3])[:,2].min())
    if abs(clearance)>.04+1e-7:
        raise ValueError('Source contact endpoint exceeds the 4 cm grounding correction')
    result=pose.copy();result[2]-=clearance
    return result


def grounded_step(source_support, source_swing, measured_support,
                  max_stride=.16, max_width=.18):
    """Preserve source direction/yaw, cap stride in support-heading coordinates.

    This geometric cap is conservative, not a full reachability certificate.
    Relative placement avoids accumulating the source's absolute-position error.
    Poses use world XYZ and XYZW quaternion.
    """
    source_support, source_swing, measured_support = (
        np.asarray(p, dtype=float) for p in (source_support, source_swing, measured_support))
    if any(p.shape != (7,) or not np.isfinite(p).all()
           for p in (source_support, source_swing, measured_support)):
        raise ValueError('Expected three finite poses [7]')
    if not np.isfinite([max_stride,max_width]).all() or min(max_stride,max_width)<=0:
        raise ValueError('Step limits must be finite and positive')
    def heading(p):
        r = Rotation.from_quat(p[3:]).as_matrix()
        return np.arctan2(r[1, 0], r[0, 0])
    source_yaw, actual_yaw = heading(source_support), heading(measured_support)
    source_rotation = Rotation.from_euler('z', source_yaw)
    actual_rotation = Rotation.from_euler('z', actual_yaw)
    displacement = source_rotation.inv().apply(source_swing[:3]-source_support[:3])
    displacement[0] = np.clip(displacement[0], -max_stride, max_stride)
    displacement[1] = np.clip(displacement[1], -max_width, max_width)
    position = measured_support[:3]+actual_rotation.apply(displacement)
    position[2] = .035
    orientation = Rotation.from_euler('z', actual_yaw+heading(source_swing)-source_yaw).as_quat()
    return np.r_[position, orientation]
