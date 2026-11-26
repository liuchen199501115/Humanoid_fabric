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
        self.braking = None
        self.last_braking_slack = None
        self.last_augmented_problem = None
        self.last_augmented_solution = None
        self.last_slack = None

    @torch.no_grad()
    def solve(self, qp, warm_z=None, warm_lam_eq=None, workspace='tracking'):
        self.last_augmented_problem = self.last_augmented_solution = self.last_slack = None
        self.last_braking_slack = None
        if self.tube is None:
            if self.braking is not None:
                raise ValueError('Optional braking requires an active tracking-tube problem')
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
        weights = qp.h.new_full((3,), self.weight)
        soft_rows = self.soft_rows
        if self.braking is not None:
            brake_C, brake_d, brake_weight = self.braking
            brake_C, brake_d = brake_C.to(qp.H), brake_d.to(qp.H)
            if (brake_C.ndim != 3 or brake_C.shape[0] != batch or brake_C.shape[2] != 35
                    or brake_C.shape[1] == 0 or brake_d.shape != brake_C.shape[:2]
                    or not torch.isfinite(brake_C).all() or not torch.isfinite(brake_d).all()
                    or not math.isfinite(brake_weight) or brake_weight <= 0):
                raise ValueError('Invalid optional soft braking rows')
            count = brake_C.shape[1]
            C, d = torch.cat((C, brake_C), dim=1), torch.cat((d, brake_d), dim=1)
            weights = torch.cat((weights, weights.new_full((count,), brake_weight)))
            soft_rows = (*soft_rows, *((True,) * count))
        rows = C.shape[1]
        H = F.pad(qp.H, (0, rows, 0, rows))
        H[:, n:, n:] = torch.diag(weights)
        soft_G = H.new_zeros(batch, rows, n + rows)
        soft_G[:, :, :35] = C
        soft_G[:, :, n:] = -torch.eye(rows, device=H.device, dtype=H.dtype)
        upper = F.pad(qp.ub.to(H), (0, rows))
        upper[:, n:] = torch.where(torch.tensor(soft_rows, device=H.device),
            H.new_tensor(float('inf')), H.new_tensor(0.))
        augmented = QPData(
            H, F.pad(qp.h, (0, rows)),
            torch.cat((F.pad(qp.G.to(H), (0, rows)), soft_G), dim=1),
            torch.cat((qp.b.to(H), d), dim=1),
            F.pad(qp.lb.to(H), (0, rows)),
            upper,
            None if qp.A_eq is None else F.pad(qp.A_eq.to(H), (0, rows)),
            None if qp.b_eq is None else qp.b_eq.to(H))
        if warm_z is not None:
            slack = (C @ warm_z[:, :35, None].to(H)).squeeze(-1) - d
            warm_z = torch.cat((warm_z.to(H), slack.clamp_min(0)), dim=1)
        result = super().solve(augmented, warm_z, warm_lam_eq, workspace + '_soft_tube')
        self.last_augmented_problem = augmented
        self.last_augmented_solution = result.z.clone()
        self.last_slack = result.z[:, n:n+3].clone()
        if self.braking is not None:
            self.last_braking_slack = result.z[:, n+3:].clone()
        return QPSolution(result.z[:, :n], result.converged, result.iters,
                          result.obj, result.lam_eq, result.status_ok)
