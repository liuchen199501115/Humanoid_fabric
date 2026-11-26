"""Acceleration metrics without task requests or extra QP solves."""
import math
import torch


def linear_acceleration_weight(base, target, time, *, start=0., ramp=0.):
    """Blend only the base translation cost, on the simulation clock."""
    if not math.isfinite(base) or base <= 0:
        raise ValueError('Base acceleration weight must be finite and positive')
    if target is not None and (not math.isfinite(target) or target <= 0):
        raise ValueError('Linear acceleration weight must be finite and positive')
    if any(not math.isfinite(x) or x < 0 for x in (time, start, ramp)):
        raise ValueError('Acceleration weight times must be finite and nonnegative')
    if target is None or time < start:
        return base
    phase = min(1., (time-start)/ramp) if ramp else 1.
    blend = phase**3 * (10 - 15*phase + 6*phase**2)
    return base + blend*(target-base)


def preserve_com_acceleration(metric, com_jacobian, weight):
    """Add weight*||J_com*(a-a_reference)||^2 to an acceleration metric."""
    if not math.isfinite(weight) or weight < 0:
        raise ValueError('COM acceleration preservation weight must be finite and nonnegative')
    if weight == 0:
        return metric
    if metric.ndim == 1:
        metric = torch.diag(metric).unsqueeze(0)
    return metric + weight * (com_jacobian.transpose(-1, -2) @ com_jacobian)


def track_com_feedback(metric, reference, snapshot, scale, *, axes='xyz'):
    """Complete the square for acceleration tracking plus ORIGINAL COM PD.

    0.5*||a-reference||_metric^2 +
    0.5*scale*||J_com*a+bias-original_desired||_com_metric^2.
    This is an algebraic change of objective, not another constrained solve.
    Neither the physical mass matrix nor the contact projection is changed.
    """
    if not math.isfinite(scale) or scale < 0:
        raise ValueError('COM feedback metric scale must be finite and nonnegative')
    if axes not in ('xy', 'xyz'):
        raise ValueError('COM feedback axes must be xy or xyz')
    if scale == 0:
        return metric, reference
    # Accumulate in double precision before passing the objective to DAQP.
    W = metric.double()
    if W.ndim == 1:
        W = torch.diag(W)[None]
    J = snapshot['com_jac'].double()
    com_metric = snapshot['com_acc_metric'].double()
    if axes == 'xy':
        mask = com_metric.new_tensor([1., 1., 0.])
        com_metric = com_metric * mask[:, None] * mask[None, :]
    pullback = J.transpose(-1, -2) @ (scale * com_metric)
    combined = W + pullback @ J
    target = snapshot['com_acc_desired'].double() - snapshot['com_jac_bias'].double()
    rhs = W @ reference.double()[..., None] + pullback @ target[..., None]
    center = torch.linalg.solve(combined, rhs).squeeze(-1)
    # The solver's state tensors share the physical model's dtype.
    return combined.to(reference.dtype), center.to(reference.dtype)


def com_feedback_scale(scale, time, *, start=0., ramp=0.):
    """Smoothly enable feedback on the simulation clock; retain the prefix."""
    if any(not math.isfinite(x) or x < 0 for x in (scale, time, start, ramp)):
        raise ValueError('COM feedback scale and times must be finite and nonnegative')
    if time < start:
        return 0.
    phase = min(1., (time-start)/ramp) if ramp else 1.
    return scale * phase**3 * (10 - 15*phase + 6*phase**2)
