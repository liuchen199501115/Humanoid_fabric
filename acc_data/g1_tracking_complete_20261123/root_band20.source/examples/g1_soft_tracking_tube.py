"""Optional tracking-tube slack in the same, independently certified DAQP solve.

Only for direct acceleration control: physical acceleration is z[:35].
The caller excludes tube rows from its hard constraints and supplies them here.
Physical constraints are padded with zeros; none acquire a slack coefficient.
"""
import math

import torch
import torch.nn.functional as F

from fabrics_sim.solver.admm_qp import QPData, QPSolution
from fabrics_sim.solver.daqp_backend import DAQPBackend


class SoftTrackingTubeBackend(DAQPBackend):
    def __init__(self, weight, soft_rows=(True, True, True)):
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError('Tube slack weight must be finite and positive')
        super().__init__(single_solve=True)
        self.weight = weight
        if len(soft_rows) != 3 or any(type(x) is not bool for x in soft_rows):
            raise ValueError('Expected three boolean soft-row selectors')
        self.soft_rows = soft_rows
        self.tube = None
        self.last_augmented_problem = None
        self.last_augmented_solution = None
        self.last_slack = None

    @torch.no_grad()
    def solve(self, qp, warm_z=None, warm_lam_eq=None, workspace='tracking'):
        self.last_augmented_problem = self.last_augmented_solution = self.last_slack = None
        if self.tube is None:
            return super().solve(qp, warm_z, warm_lam_eq, workspace)
        C, d, scale = (x.to(qp.H) for x in self.tube)
        batch, n = qp.h.shape
        if C.shape != (batch, 3, 35) or d.shape != (batch, 3) or scale.shape != (3,):
            raise ValueError('Expected exactly three G1 tracking-tube rows')
        if not all(torch.isfinite(x).all() for x in (C, d, scale)) or (scale <= 0).any():
            raise ValueError('Invalid tracking-tube coefficients')
        # Normalize by the tube's squared radius (quaternion barrier uses sin²).
        # Slack measures normalized barrier-acceleration violation, in s^-2.
        C, d = C / scale[None, :, None], d / scale
        H = F.pad(qp.H, (0, 3, 0, 3))
        H[:, n:, n:] = self.weight * torch.eye(3, device=H.device, dtype=H.dtype)
        soft_G = H.new_zeros(batch, 3, n + 3)
        soft_G[:, :, :35] = C
        soft_G[:, :, n:] = -torch.eye(3, device=H.device, dtype=H.dtype)
        upper = F.pad(qp.ub.to(H), (0, 3))
        upper[:, n:] = torch.where(torch.tensor(self.soft_rows, device=H.device),
            H.new_tensor(float('inf')), H.new_tensor(0.))
        augmented = QPData(
            H, F.pad(qp.h, (0, 3)),
            torch.cat((F.pad(qp.G.to(H), (0, 3)), soft_G), dim=1),
            torch.cat((qp.b.to(H), d), dim=1),
            F.pad(qp.lb.to(H), (0, 3)),
            upper,
            None if qp.A_eq is None else F.pad(qp.A_eq.to(H), (0, 3)),
            None if qp.b_eq is None else qp.b_eq.to(H))
        if warm_z is not None:
            slack = (C @ warm_z[:, :35, None].to(H)).squeeze(-1) - d
            warm_z = torch.cat((warm_z.to(H), slack.clamp_min(0)), dim=1)
        result = super().solve(augmented, warm_z, warm_lam_eq, workspace + '_soft_tube')
        self.last_augmented_problem = augmented
        self.last_augmented_solution = result.z.clone()
        self.last_slack = result.z[:, n:].clone()
        return QPSolution(result.z[:, :n], result.converged, result.iters,
                          result.obj, result.lam_eq, result.status_ok)
