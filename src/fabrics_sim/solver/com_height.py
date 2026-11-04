"""A moving lower height envelope for the measured center of mass."""
import math

import torch


def com_height_constraint(position, jacobian, bias, velocity, reference_position,
                          reference_velocity, reference_acceleration, margin=.02,
                          bandwidth=10.):
    """Return C, d for C qdd <= d, bounding downward COM tracking error.

    h = z - z_ref + margin obeys hdd + 2*w*hd + w*w*h >= 0.
    This uses physical acceleration and the moving reference derivatives.
    Forward invariance requires an admissible initial state and exact dynamics;
    impacts or an infeasible QP can still invalidate this local bound.
    """
    batch = position.shape[0]
    expected = ((position, (batch,3)), (jacobian, (batch,3,35)),
                (bias, (batch,3)), (velocity, (batch,35)),
                (reference_position, (batch,3)), (reference_velocity, (batch,3)),
                (reference_acceleration, (batch,3)))
    if (not all(math.isfinite(v) and v > 0 for v in (margin, bandwidth))
            or any(x.shape != shape or x.device != position.device or x.dtype != position.dtype
                   or not torch.isfinite(x).all() for x,shape in expected)):
        raise ValueError('COM height bound needs finite matching physical states and positive margin/bandwidth')
    speed = (jacobian @ velocity[...,None]).squeeze(-1)
    gap = position[:,2] - reference_position[:,2] + margin
    bound = (bias[:,2] - reference_acceleration[:,2]
             + 2*bandwidth*(speed[:,2]-reference_velocity[:,2]) + bandwidth**2*gap)
    return -jacobian[:,2:3], bound[:,None]
