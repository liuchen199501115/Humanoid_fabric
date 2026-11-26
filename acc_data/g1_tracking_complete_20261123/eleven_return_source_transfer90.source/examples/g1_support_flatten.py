"""Soft flattening acceleration in the measured contact frame.

The level sole heading and the point/edge constraint frame are distinct.
Only the existing free angular task changes; contact rows remain unchanged.
"""
import math
from fabrics_sim.fabrics.dynamics_aware import rotation_error


def flat_support_target(rotation,sole_frame,contact_frame,angular_jac,velocity,
                        *,gain_scale=1.,feedback_limit=25.):
    if (not math.isfinite(gain_scale) or gain_scale<=0
            or not math.isfinite(feedback_limit) or feedback_limit<=0):
        raise ValueError('Flattening gain and feedback limit must be finite and positive')
    error=(contact_frame.transpose(-1,-2)@rotation_error(rotation,sole_frame)[...,None]).squeeze(-1)
    rate=(angular_jac@velocity[...,None]).squeeze(-1)
    return (-50*gain_scale*error-14*math.sqrt(gain_scale)*rate).clamp(-feedback_limit,feedback_limit)
