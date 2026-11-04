"""Flat-ground balance check before deliberately releasing a support foot."""
import numpy as np
from scipy.spatial.transform import Rotation


def source_transfer_reference(com,velocity,acceleration,stance_pose):
    """Retain source forward motion while transferring lateral load to stance."""
    com,velocity,acceleration,stance=(np.asarray(x,dtype=float) for x in
                                      (com,velocity,acceleration,stance_pose))
    if (any(x.shape!=(3,) for x in (com,velocity,acceleration)) or stance.shape!=(7,)
            or not all(np.isfinite(x).all() for x in (com,velocity,acceleration,stance))):
        raise ValueError('Expected finite COM kinematics and a stance pose')
    forward=Rotation.from_quat(stance[3:]).as_matrix()[:2,0]
    norm=np.linalg.norm(forward)
    if norm<1e-6:raise ValueError('Stance heading must have a horizontal component')
    forward/=norm
    return (stance[:2]+forward*np.dot(com[:2]-stance[:2],forward),
            forward*np.dot(velocity[:2],forward),forward*np.dot(acceleration[:2],forward))


def support_balance_ready(com, velocity, feet, max_speed=.06):
    com,velocity,feet=(np.asarray(a,dtype=float) for a in (com,velocity,feet))
    if com.shape!=(3,) or velocity.shape!=(3,) or feet.shape!=(2,7):
        raise ValueError('Expected COM/velocity [3] and feet [2,7]')
    if not all(np.isfinite(a).all() for a in (com,velocity,feet)):
        raise ValueError('Balance observations must be finite')
    capture=com[:2]+velocity[:2]/np.sqrt(9.81/max(com[2],.1))
    delta=np.c_[capture[None]-feet[:,:2],np.zeros(2)]
    local=Rotation.from_quat(feet[:,3:]).inv().apply(delta)
    return ((local[:,0]>=-.04)&(local[:,0]<=.11)&(np.abs(local[:,1])<=.02)
            &(np.linalg.norm(velocity[:2])<=max_speed))
