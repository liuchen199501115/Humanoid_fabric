"""Compose local tracking leaves before the sole inverse-dynamics QP.

Contact-aware priority uses the affine acceleration manifold a=P*a+c. This is
an instantaneous algebraic policy, with no horizon or constrained solve.
"""
import math
from dataclasses import replace

import torch

from fabrics_sim.fabrics.priority import dynamic_nullspace, prioritize_requests
from fabrics_sim.fabrics.task_request import TaskRequest


def scale_secondary_tasks(requests, primary_names, primary_rows, scale):
    """Soft weighting of the historical primary group; never another QP."""
    if not math.isfinite(scale) or not 0 <= scale <= 1:
        raise ValueError('Secondary task scale must lie in [0,1]')
    if scale == 1:
        return requests
    result = []
    for request in requests:
        factors = torch.full_like(request.desired, math.sqrt(scale))
        if request.name in primary_names:
            rows = primary_rows.get(request.name)
            if rows is None:
                factors.fill_(1.)
            else:
                factors[:, list(rows)] = 1.
        result.append(replace(request, metric=request.metric*factors[..., None]*factors[:, None, :]))
    return result


def compose_tracking_policy(requests, mass, contact_jacobian, contact_bias,
                            contact_mask, mode, regularization=1e-4):
    if mode not in ('none', 'paper-tracking', 'contact-tracking'):
        raise ValueError('Physical tracking composition requires a supported mode')
    if not requests or regularization <= 0:
        raise ValueError('Tasks and positive regularization are required')
    Jc = contact_jacobian.double() * contact_mask[..., None]
    P, inverse = dynamic_nullspace(Jc, mass, return_inverse=True)
    offset = -(inverse @ (contact_bias.double()*contact_mask)[..., None]).squeeze(-1)
    leaves = requests
    if mode == 'contact-tracking':
        leaves = [TaskRequest(r.name, r.A.double() @ P,
            r.bias.double() + (r.A.double() @ offset[..., None]).squeeze(-1),
            r.desired.double(), r.metric.double(), r.active, r.priority) for r in requests]
    diagnostics = {}
    if mode != 'none':
        jacobians = {r.name: r.A for r in leaves}
        leaves, diagnostics = prioritize_requests(leaves, (mass,)*3, (jacobians,)*3,
            torch.zeros_like(offset), .001, physical_targets=True)
    H = regularization*torch.eye(35, device=mass.device, dtype=torch.float64)[None].expand_as(mass).clone()
    force = torch.zeros_like(offset)
    if mode == 'contact-tracking':
        # Regularize around the affine contact offset, not impossible zero
        # contact-normal acceleration; this preserves the physical task cost.
        force -= regularization*offset
    for r in leaves:
        h, g = r.natural_form(dtype=torch.float64)
        H += h
        force += g
    acceleration = torch.linalg.solve(H, -force[..., None]).squeeze(-1)
    if not torch.isfinite(acceleration).all():
        raise FloatingPointError('Composed tracking acceleration is not finite')
    projected = (P @ acceleration[..., None]).squeeze(-1) + offset
    diagnostics['fabric_priority_contact_error'] = (Jc @ projected[..., None]
        + (contact_bias.double()*contact_mask)[..., None]).abs().amax(dim=(-2, -1))
    return acceleration.to(mass.dtype), H, projected.to(mass.dtype), diagnostics
