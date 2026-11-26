"""Two-level Fabric pullback following arXiv:2303.04279, equations (8)-(9).

Priority belongs to the motion policy, not to another QP stage. For G1 we map
COM and angular-momentum regulation to stability; imitation tasks are level 2.
The actuated-joint selector excludes six unactuated base coordinates. Physical
contact/effort constraints remain the downstream inverse-dynamics QP's job.
"""
import math

import torch

from .task_request import TaskRequest


STABILITY_TASKS = frozenset(('center_of_mass', 'centroidal_momentum'))


def dynamic_nullspace(jacobian, mass):
    """I - D^-1 J^T (J D^-1 J^T)^dagger J, including redundant/zero rows."""
    J, D = jacobian.double(), mass.double()
    if (D.ndim != 3 or J.ndim != 3 or D.shape[0] != J.shape[0]
            or D.shape[1:] != (J.shape[-1], J.shape[-1])
            or not torch.isfinite(J).all() or not torch.isfinite(D).all()
            or not torch.allclose(D, D.transpose(-1, -2), atol=1e-8, rtol=1e-6)):
        raise ValueError('Expected finite symmetric batched inertia and matching Jacobian')
    identity = torch.eye(D.shape[-1], device=D.device, dtype=D.dtype)[None].expand_as(D)
    # Row scaling leaves the exact nullspace unchanged and avoids mixing
    # COM metres and angular momentum units in the rank threshold.
    J = J / J.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    factor = torch.linalg.cholesky(.5 * (D + D.transpose(-1, -2)))
    inverse_jt = torch.cholesky_solve(J.transpose(-1, -2), factor)
    gram = J @ inverse_jt
    gram_inverse = torch.linalg.pinv(.5*(gram + gram.transpose(-1, -2)), hermitian=True, rtol=1e-10)
    return identity - inverse_jt @ gram_inverse @ J


def prioritize_requests(requests, masses, jacobian_samples, velocity, difference_step):
    """Project level-2 Jacobians, differentiating the ENTIRE J*S*N product.

    Samples are (current, q+eps*v, q-eps*v). Existing physical tracking targets
    and task metrics are retained; only the pullback and its curvature change.
    No task list is cached here. Returns local leaves and matrix diagnostics.
    """
    if not math.isfinite(difference_step) or difference_step <= 0:
        raise ValueError('Positive central-difference step required')
    if len(masses) != 3 or len(jacobian_samples) != 3:
        raise ValueError('Expected current, plus and minus samples')
    if velocity.ndim != 2 or velocity.shape[-1] != 35 or not torch.isfinite(velocity).all():
        raise ValueError('Expected finite [B,35] velocity')
    high = [r for r in requests if r.name in STABILITY_TASKS]
    if not high:
        raise ValueError('Paper priority needs at least one stability task')
    selector = torch.eye(35, device=velocity.device, dtype=torch.float64)
    selector[:6, :6] = 0.
    high_jacobians = []
    nullspaces = []
    for mass, samples in zip(masses, jacobian_samples):
        J = torch.cat([samples[r.name].double() @ selector *
            (r.active & (r.metric.diagonal(dim1=-2, dim2=-1) > 0))[..., None]
            for r in high], dim=1)
        high_jacobians.append(J)
        nullspaces.append(dynamic_nullspace(J, mass))
    projected = []
    for request in requests:
        stable = request.name in STABILITY_TASKS
        matrices = [samples[request.name].double() @ selector @
                    (torch.eye(35, device=velocity.device, dtype=torch.float64) if stable else N)
                    for samples, N in zip(jacobian_samples, nullspaces)]
        curvature = ((matrices[1]-matrices[2])/(2*difference_step) @ velocity.double()[..., None]).squeeze(-1)
        projected.append(TaskRequest(request.name, matrices[0], curvature,
            request.desired.double(), request.metric.double(), request.active,
            'high' if stable else 'low'))
    J, N = high_jacobians[0], nullspaces[0]
    return projected, dict(fabric_priority_high_jacobian=J,
        fabric_priority_nullspace=N,
        fabric_priority_nullspace_error=(J @ N).abs().amax(dim=(-2, -1)),
        fabric_priority_idempotence_error=(N @ N-N).abs().amax(dim=(-2, -1)))
