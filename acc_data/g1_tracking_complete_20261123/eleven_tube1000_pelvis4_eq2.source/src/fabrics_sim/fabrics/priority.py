"""Two-level Fabric pullback following arXiv:2303.04279, equations (8)-(9).

Priority belongs to the motion policy, not to another QP stage. For G1 we map
COM and angular-momentum regulation to stability; imitation tasks are level 2.
Unlike the paper's actuated-joint output, this controller's root is the full
35-D generalized acceleration, so its selector is identity. Removing base
columns would change the meaning of existing physical COM tracking targets.
Actuation, contact and effort constraints remain the inverse-dynamics QP's job.
"""
import math

import torch

from .task_request import TaskRequest


STABILITY_TASKS = frozenset(('center_of_mass', 'centroidal_momentum'))


def dynamic_nullspace(jacobian, mass, *, return_inverse=False):
    """I - D^-1 J^T (J D^-1 J^T)^dagger J, including redundant/zero rows."""
    J, D = jacobian.double(), mass.double()
    if (D.ndim != 3 or J.ndim != 3 or D.shape[0] != J.shape[0]
            or D.shape[1:] != (J.shape[-1], J.shape[-1])
            or not torch.isfinite(J).all() or not torch.isfinite(D).all()):
        raise ValueError('Expected finite batched inertia and matching Jacobian')
    # URDF inertia is assembled in float32. Near-zero off-diagonal entries
    # need an absolute roundoff allowance based on the matrix scale, not a
    # double-precision elementwise tolerance after casting.
    symmetry_error = (D-D.transpose(-1, -2)).abs().amax(dim=(-2, -1))
    tolerance = 16*torch.finfo(mass.dtype).eps*D.abs().amax(dim=(-2, -1)).clamp_min(1.)
    if (symmetry_error > tolerance).any():
        raise ValueError(f'Inertia is not symmetric: error={symmetry_error.tolist()}')
    identity = torch.eye(D.shape[-1], device=D.device, dtype=D.dtype)[None].expand_as(D)
    # Row scaling leaves the exact nullspace unchanged and avoids mixing
    # COM metres and angular momentum units in the rank threshold.
    row_scale = J.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    J = J / row_scale
    factor = torch.linalg.cholesky(.5 * (D + D.transpose(-1, -2)))
    inverse_jt = torch.cholesky_solve(J.transpose(-1, -2), factor)
    gram = J @ inverse_jt
    gram_inverse = torch.linalg.pinv(.5*(gram + gram.transpose(-1, -2)), hermitian=True, rtol=1e-10)
    inverse = inverse_jt @ gram_inverse
    nullspace = identity - inverse @ J
    if return_inverse:
        return nullspace, inverse / row_scale.transpose(-1, -2)
    return nullspace


def prioritize_requests(requests, masses, jacobian_samples, velocity, difference_step,
                        *, physical_targets=False):
    """Project level-2 Jacobians, differentiating the ENTIRE J*S*N product.

    Samples are (current, q+eps*v, q-eps*v). Existing physical tracking targets
    and task metrics are retained; only the pullback and its curvature change.
    No task list is cached here. Returns local leaves and matrix diagnostics.
    With physical_targets, use an affine acceleration decomposition instead:
    low residual = J*N*a + J*a_high + original_bias - desired. This preserves
    physical tracking residuals whenever the high tasks are achieved. It uses
    the physical Jdot*v, not a differentiated virtual task map J*N.
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
    high_jacobians = []
    nullspaces = []
    for mass, samples in zip(masses, jacobian_samples):
        J = torch.cat([samples[r.name].double() *
            (r.active & (r.metric.diagonal(dim1=-2, dim2=-1) > 0))[..., None]
            for r in high], dim=1)
        high_jacobians.append(J)
        if physical_targets and not nullspaces:
            nullspace, inverse = dynamic_nullspace(J, mass, return_inverse=True)
            rhs = torch.cat([(r.desired-r.bias).double() *
                (r.active & (r.metric.diagonal(dim1=-2, dim2=-1) > 0)) for r in high], dim=1)
            high_acceleration = inverse @ rhs[..., None]
        else:
            nullspace = dynamic_nullspace(J, mass)
        nullspaces.append(nullspace)
    projected = []
    for request in requests:
        stable = request.name in STABILITY_TASKS
        matrices = [(samples[request.name].double() if stable else samples[request.name].double() @ N)
                    for samples, N in zip(jacobian_samples, nullspaces)]
        curvature = ((matrices[1]-matrices[2])/(2*difference_step) @ velocity.double()[..., None]).squeeze(-1)
        if physical_targets:
            curvature = request.bias.double()
            if not stable:
                curvature = curvature + (request.A.double() @ high_acceleration).squeeze(-1)
        projected.append(TaskRequest(request.name, matrices[0], curvature,
            request.desired.double(), request.metric.double(), request.active,
            'high' if stable else 'low'))
    J, N = high_jacobians[0], nullspaces[0]
    return projected, dict(fabric_priority_high_jacobian=J,
        fabric_priority_nullspace=N,
        fabric_priority_nullspace_error=(J @ N).abs().amax(dim=(-2, -1)),
        fabric_priority_idempotence_error=(N @ N-N).abs().amax(dim=(-2, -1)))
