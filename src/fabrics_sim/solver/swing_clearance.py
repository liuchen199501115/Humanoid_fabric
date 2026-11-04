"""Measured sole-corner acceleration bounds during the airborne swing phase."""

import math
import torch


def clearance_active(mode, lifted, frame, landing_frame, fps, release_seconds=.1):
    """Release before planned landing; never constrain the initial toe-off."""
    return (mode in (1, 2) and lifted and landing_frame is not None
            and (landing_frame - frame) / fps > release_seconds)


def lift_clearance(elapsed, height=.02, duration=.2):
    """C2 clearance floor for an already released foot, on wall-clock time.

    Starts at the ground with zero derivatives, then raises the requested
    minimum clearance. The caller must release this bound before touchdown.
    This is a lower envelope, not a replacement swing-foot trajectory.
    """
    if not all(math.isfinite(v) for v in (elapsed, height, duration)) or height < 0 or duration <= 0:
        raise ValueError('Invalid lift clearance profile')
    s = min(1., max(0., elapsed / duration))
    return (height*s**3*(10-15*s+6*s*s),
            height*30*s*s*(1-s)**2/duration,
            height*60*s*(1-s)*(1-2*s)/duration**2)


def sole_clearance_constraints(position, rotation, jacobian, bias, velocity,
                               corners, clearance=.005, bandwidth=20.,
                               clearance_velocity=0., clearance_acceleration=0.):
    """Return C, d for C @ qdd <= d and measured corner heights/velocities.

    Jacobian and bias are world-frame origin linear then angular components.
    Each corner satisfies hdd + 2*w*hd + w**2*h >= 0 for
    h = z - clearance, including a moving clearance floor's derivatives.
    Centripetal acceleration is included. This is a local acceleration barrier,
    not a certificate against unmodelled contact impulses or infeasible states.
    """
    if not math.isfinite(clearance) or clearance < 0:
        raise ValueError('Clearance must be finite and nonnegative')
    if not math.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError('Bandwidth must be finite and positive')
    if not all(math.isfinite(v) for v in (clearance_velocity, clearance_acceleration)):
        raise ValueError('Clearance derivatives must be finite')
    offsets = (rotation[:, None] @ corners[None, :, :, None]).squeeze(-1)
    angular_columns = jacobian[:, 3:].transpose(-1, -2)[:, None]
    point_jac = jacobian[:, None, :3] + torch.linalg.cross(
        angular_columns, offsets[:, :, None], dim=-1).transpose(-1, -2)
    omega = (jacobian[:, 3:] @ velocity[..., None]).squeeze(-1)[:, None]
    point_bias = (bias[:, None, :3]
                  + torch.linalg.cross(bias[:, None, 3:], offsets, dim=-1)
                  + torch.linalg.cross(omega, torch.linalg.cross(omega, offsets, dim=-1), dim=-1))
    height = position[:, None, 2] + offsets[..., 2]
    vertical_jac = point_jac[:, :, 2]
    speed = (vertical_jac @ velocity[..., None]).squeeze(-1)
    bound = (point_bias[..., 2] - clearance_acceleration
             + 2*bandwidth*(speed-clearance_velocity) + bandwidth**2*(height-clearance))
    return -vertical_jac, bound, height, speed
