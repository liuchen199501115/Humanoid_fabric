"""Optional support-knee flexion barrier for the instantaneous acceleration QP."""
import math
import torch


def support_knee_constraints(position, velocity, active_contacts, *,
                             minimum_angle, bandwidth=20., knee_indices=(3, 9)):
    """Add h'' + 2*w*h' + w*w*h >= 0, with h = knee - minimum.

    This is an extra conditioning constraint, not a changed physical joint
    limit. Ideal forward invariance also requires h >= 0 and h' + w*h >= 0
    initially; measured impacts can violate those conditions. The caller must
    continue checking physical feasibility and original-source tracking.
    """
    if (position.ndim != 2 or position.shape[-1] != 29 or velocity.shape != position.shape
            or active_contacts.shape != (len(position), 2) or active_contacts.dtype != torch.bool):
        raise ValueError('Expected joint states [B,29] and boolean contacts [B,2]')
    if not bool(torch.isfinite(position).all() and torch.isfinite(velocity).all()):
        raise ValueError('Joint states must be finite')
    if (not math.isfinite(minimum_angle) or not 0 < minimum_angle < 1
            or not math.isfinite(bandwidth) or bandwidth <= 0):
        raise ValueError('Invalid support-knee barrier parameters')
    if len(knee_indices) != 2 or len(set(knee_indices)) != 2 or any(not 0 <= i < 29 for i in knee_indices):
        raise ValueError('Expected two distinct knee indices')
    indices = list(knee_indices)
    h = position[:, indices] - minimum_angle
    rate = velocity[:, indices]
    minimum_acceleration = -2 * bandwidth * rate - bandwidth**2 * h
    C = position.new_zeros((len(position), 2, 35))
    for foot, joint in enumerate(indices):
        C[:, foot, 6+joint] = -active_contacts[:, foot].to(position.dtype)
    d = torch.where(active_contacts, -minimum_acceleration, torch.zeros_like(h))
    diagnostics = dict(support_knee_active=active_contacts,
                       support_knee_margin=h,
                       support_knee_invariant=rate+bandwidth*h,
                       support_knee_minimum_acceleration=minimum_acceleration)
    return C, d, diagnostics
