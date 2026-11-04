from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

@dataclass
class QPData:
    """Batched QP data. All tensors have leading batch dim B."""

    H:     Tensor
    h:     Tensor
    G:     Tensor
    b:     Tensor
    lb:    Tensor
    ub:    Tensor

    A_eq:  Tensor | None = None
    b_eq:  Tensor | None = None

@dataclass
class QPSolution:
    """Solution returned by the ADMM solver."""

    z: Tensor
    converged: Tensor
    iters: int
    obj: Tensor
    lam_eq: Tensor | None = None
    status_ok: Tensor | None = None
    """Optional native solver status; an explicit failure cannot pass validation."""

class ADMMSolver:
    """GPU-batched ADMM QP solver."""

    def __init__(
        self,
        max_iter: int = 200,
        rho: float = 1.0,
        rho_eq: float = 100.0,
        eps_abs: float = 1e-4,
        eps_rel: float = 1e-4,
        adaptive_rho: bool = True,
        adaptive_rho_interval: int = 25,
    ):
        self.max_iter = max_iter
        self.rho = rho
        self.rho_eq = rho_eq
        self.eps_abs = eps_abs
        self.eps_rel = eps_rel
        self.adaptive_rho = adaptive_rho
        self.adaptive_rho_interval = adaptive_rho_interval

    @torch.no_grad()
    def solve(
        self,
        qp: QPData,
        warm_z: Tensor | None = None,
        warm_lam_eq: Tensor | None = None,
    ) -> QPSolution:
        """Solve a batch of QPs."""
        B, n = qp.h.shape
        m    = qp.b.shape[1]
        device = qp.h.device
        dtype  = qp.h.dtype
        has_eq = qp.A_eq is not None

        rho = self.rho

        z = (warm_z.clone() if warm_z is not None
             else torch.zeros(B, n, device=device, dtype=dtype))
        Gz = torch.bmm(qp.G, z.unsqueeze(-1)).squeeze(-1)
        y  = torch.minimum(Gz, qp.b)
        u  = torch.zeros(B, m, device=device, dtype=dtype)

        z_box = z.clone()
        u_box = torch.zeros(B, n, device=device, dtype=dtype)

        if has_eq:
            n_eq   = qp.b_eq.shape[1]
            lam_eq = (warm_lam_eq.clone() if warm_lam_eq is not None
                      else torch.zeros(B, n_eq, device=device, dtype=dtype))

        GtG = torch.bmm(qp.G.transpose(1, 2), qp.G)
        I_n = torch.eye(n, device=device, dtype=dtype).unsqueeze(0)
        rho_t = torch.full((B, 1, 1), rho, device=device, dtype=dtype)

        def _factorize(rho_val: Tensor) -> tuple:
            M = qp.H + rho_val * (GtG + I_n)
            if not has_eq:
                return torch.linalg.lu_factor(M)

            # Enforce equalities through the KKT system instead of an
            # augmented-Lagrangian penalty. Contact/dynamics rows have very
            # different scales; a fixed equality penalty can leave large
            # residuals or prevent box constraints from converging.
            kkt = torch.zeros(
                B, n + n_eq, n + n_eq, device=device, dtype=dtype
            )
            kkt[:, :n, :n] = M
            kkt[:, :n, n:] = qp.A_eq.transpose(1, 2)
            kkt[:, n:, :n] = qp.A_eq
            # A caller may retain structurally masked or redundant equality
            # rows. Tiny dual regularization keeps the KKT factorization
            # nonsingular without materially relaxing physical equalities.
            dual_regularization = max(
                1.0e-9, float(torch.finfo(dtype).eps)
            )
            kkt[:, n:, n:] = -dual_regularization * torch.eye(
                n_eq, device=device, dtype=dtype
            ).unsqueeze(0)
            return torch.linalg.lu_factor(kkt)

        LU, pivots = _factorize(rho_t)

        iters_used = 0
        r_ineq = torch.zeros(B, m, device=device, dtype=dtype)
        r_box  = torch.zeros(B, n,  device=device, dtype=dtype)
        dual_residual = torch.full(
            (B,), float("inf"), device=device, dtype=dtype
        )

        for it in range(self.max_iter):
            y_previous = y.clone()
            z_box_previous = z_box.clone()

            rhs = -qp.h + rho_t.squeeze(-1) * (
                torch.bmm(qp.G.transpose(1, 2), (y - u).unsqueeze(-1)).squeeze(-1)
                + (z_box - u_box)
            )
            if has_eq:
                kkt_rhs = torch.cat((rhs, qp.b_eq), dim=1)
                kkt_solution = torch.linalg.lu_solve(
                    LU, pivots, kkt_rhs.unsqueeze(-1)
                ).squeeze(-1)
                z = kkt_solution[:, :n]
                lam_eq = kkt_solution[:, n:]
            else:
                z = torch.linalg.lu_solve(
                    LU, pivots, rhs.unsqueeze(-1)
                ).squeeze(-1)

            Gz = torch.bmm(qp.G, z.unsqueeze(-1)).squeeze(-1)
            y  = torch.minimum(Gz + u, qp.b)

            z_box = torch.clamp(z + u_box, min=qp.lb, max=qp.ub)

            r_ineq = Gz - y
            r_box  = z - z_box
            u     = u     + r_ineq
            u_box = u_box + r_box

            if has_eq:
                eq_res = (
                    torch.bmm(qp.A_eq, z.unsqueeze(-1)).squeeze(-1)
                    - qp.b_eq
                )

            iters_used = it + 1

            delta_y = y - y_previous
            delta_z_box = z_box - z_box_previous
            dual_vector = torch.bmm(
                qp.G.transpose(1, 2),
                delta_y.unsqueeze(-1),
            ).squeeze(-1) + delta_z_box
            dual_residual = rho * torch.linalg.vector_norm(
                dual_vector, dim=-1
            )

            if (it + 1) % 10 == 0:
                primal_res = torch.sqrt(
                    (r_ineq * r_ineq).sum(-1) + (r_box * r_box).sum(-1)
                )
                if has_eq:
                    primal_res = torch.sqrt(
                        primal_res.square()
                        + (eq_res * eq_res).sum(-1)
                    )

                # ADMM dual residual is driven by the change in the
                # consensus variables, not by the magnitude of the scaled
                # dual variables themselves.
                eps_pri = (
                    self.eps_abs * (m + n) ** 0.5
                    + self.eps_rel * torch.maximum(
                        torch.sqrt((Gz * Gz).sum(-1) + (z * z).sum(-1)),
                        torch.sqrt((y * y).sum(-1) + (z_box * z_box).sum(-1)),
                    )
                )
                stationarity_scale = torch.linalg.vector_norm(
                    rho
                    * (
                        torch.bmm(
                            qp.G.transpose(1, 2),
                            u.unsqueeze(-1),
                        ).squeeze(-1)
                        + u_box
                    ),
                    dim=-1,
                )
                eps_dual = (
                    self.eps_abs * n**0.5
                    + self.eps_rel * stationarity_scale
                )

                converged = (
                    (primal_res <= eps_pri)
                    & (dual_residual <= eps_dual)
                )
                if converged.all():
                    break

            if self.adaptive_rho and (it + 1) % self.adaptive_rho_interval == 0:
                primal_norm = torch.sqrt(
                    (r_ineq * r_ineq).sum(-1) + (r_box * r_box).sum(-1)
                ).mean()
                dual_norm = dual_residual.mean()
                rho_changed = False
                if primal_norm > 10.0 * dual_norm and rho < 1e3:
                    rho = min(rho * 2.0, 1e3)
                    rho_changed = True
                    u = u / 2.0
                    u_box = u_box / 2.0
                elif dual_norm > 10.0 * primal_norm and rho > 1e-3:
                    rho = max(rho / 2.0, 1e-3)
                    rho_changed = True
                    u = u * 2.0
                    u_box = u_box * 2.0
                if rho_changed:
                    rho_t = torch.full((B, 1, 1), rho, device=device, dtype=dtype)
                    LU, pivots = _factorize(rho_t)

        # Do not clamp the primal variable after solving: an independent
        # projection onto the box can invalidate equality and general
        # inequality constraints. Feasibility is checked explicitly below.
        z_final = z

        obj = (
            0.5 * (z_final * torch.bmm(qp.H, z_final.unsqueeze(-1)).squeeze(-1)).sum(-1)
            + (qp.h * z_final).sum(-1)
        )

        Gz_final = torch.bmm(qp.G, z_final.unsqueeze(-1)).squeeze(-1)
        ineq_ok  = (Gz_final <= qp.b + self.eps_abs).all(dim=-1)
        box_ok   = (
            (z_final >= qp.lb - self.eps_abs) & (z_final <= qp.ub + self.eps_abs)
        ).all(dim=-1)
        if has_eq:
            eq_res_final = torch.bmm(qp.A_eq, z_final.unsqueeze(-1)).squeeze(-1) - qp.b_eq
            eq_ok = (eq_res_final.abs() <= self.eps_abs * 100).all(dim=-1)
        else:
            eq_ok = torch.ones(B, device=device, dtype=torch.bool)

        return QPSolution(
            z=z_final,
            converged=ineq_ok & box_ok & eq_ok,
            iters=iters_used,
            obj=obj,
            lam_eq=lam_eq if has_eq else None,
        )
