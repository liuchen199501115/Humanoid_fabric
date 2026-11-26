"""Unilateral reactions at measured stationary mechanical joint stops.

The caller determines the observed mode. An active stop fixes its joint
acceleration at zero and can supply only an inward reaction. This models a
passive constraint already present in the simulator; it applies no effort.
"""
import math
import torch
import torch.nn.functional as F

from .admm_qp import QPData


def observed_stop_directions(position, velocity, lower, upper, *,
                             position_tolerance=1e-4, velocity_tolerance=.05):
    if (position.shape != velocity.shape or position.shape != lower.shape
            or position.shape != upper.shape or position.ndim != 2
            or position.shape[1] != 29 or not (upper > lower).all()
            or not all(torch.isfinite(x).all() for x in (position, velocity, lower, upper))
            or position_tolerance <= 0 or velocity_tolerance <= 0):
        raise ValueError('Expected finite G1 joint states and positive stop tolerances')
    stationary = velocity.abs() <= velocity_tolerance
    high = (position-upper).abs() <= position_tolerance
    low = (position-lower).abs() <= position_tolerance
    return torch.where(stationary & high, torch.ones_like(position),
        torch.where(stationary & low, -torch.ones_like(position), torch.zeros_like(position)))


def augment_joint_stops(qp, directions, reaction_limits, *, reaction_weight=1e-4):
    """Append passive reaction variables and locked-joint equalities to one QP.

    Positive direction is an upper stop: M a - J' f - S' tau + S' rho = -b.
    Lower stops reverse the reaction sign. The dynamics rows are the first 35
    equalities of the existing 76-variable G1 inverse-dynamics problem.
    """
    if (qp.h.shape[1] != 76 or directions.shape != (len(qp.h), 29)
            or reaction_limits.shape != directions.shape
            or not torch.isfinite(directions).all()
            or not ((directions == 0) | (directions == 1) | (directions == -1)).all()
            or not torch.isfinite(reaction_limits).all() or (reaction_limits <= 0).any()
            or not math.isfinite(reaction_weight) or reaction_weight <= 0
            or qp.A_eq is None or qp.A_eq.shape[1] < 35):
        raise ValueError('Invalid observed joint-stop QP inputs')
    indices = torch.nonzero(directions.ne(0).any(0), as_tuple=False).flatten()
    if not len(indices):
        return qp, indices
    n, batch = len(indices), len(qp.h)
    signs = directions[:, indices].to(qp.H)
    active = signs.ne(0)
    H = F.pad(qp.H, (0, n, 0, n))
    H[:, 76:, 76:] = reaction_weight * torch.eye(n, device=H.device, dtype=H.dtype)
    A = F.pad(qp.A_eq.to(H), (0, n))
    lock = H.new_zeros(batch, n, 76+n)
    for column, joint in enumerate(indices.tolist()):
        A[:, 6+joint, 76+column] = signs[:, column]
        lock[:, column, 6+joint] = active[:, column].to(H)
    upper = torch.cat((qp.ub.to(H), reaction_limits[:, indices].to(H)*active), dim=1)
    return QPData(H, F.pad(qp.h, (0, n)), F.pad(qp.G.to(H), (0, n)), qp.b.to(H),
        F.pad(qp.lb.to(H), (0, n)), upper,
        torch.cat((A, lock), dim=1), F.pad(qp.b_eq.to(H), (0, n))), indices
