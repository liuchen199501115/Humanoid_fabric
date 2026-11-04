"""Acceleration inequalities matching a centroidal planner's momentum budget."""
import math

import torch


def momentum_acceleration_bounds(matrix, bias, momentum, dt, limit, rate_limit):
    """Bound Hdot = A*qdd + bias and H + dt*Hdot on selected axes.

    This is a local, first-order prediction. It cannot certify impact response
    or future joint/torque feasibility. An out-of-budget measured state is not
    clipped: contradictory bounds remain contradictory for the QP to reject.
    """
    if (matrix.ndim!=3 or momentum.shape!=matrix.shape[:2] or bias.shape!=momentum.shape
            or not all(torch.isfinite(v).all() for v in (matrix,bias,momentum))
            or any(v.dtype!=matrix.dtype or v.device!=matrix.device for v in (bias,momentum))
            or not all(math.isfinite(v) and v>0 for v in (dt,limit,rate_limit))):
        raise ValueError('Momentum bounds need matching finite tensors and positive finite budgets')
    lower=torch.maximum(torch.full_like(momentum,-rate_limit),(-limit-momentum)/dt)
    upper=torch.minimum(torch.full_like(momentum,rate_limit),(limit-momentum)/dt)
    return torch.cat((matrix,-matrix),dim=1),torch.cat((upper-bias,bias-lower),dim=1)
