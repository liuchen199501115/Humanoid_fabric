from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .admm_qp import ADMMSolver, QPData


@dataclass
class DynamicsAccelerationSolution:
    """Result of the dynamics-feasible acceleration QP."""

    acc: Tensor
    """Final generalized acceleration, shape [B, 35]."""

    contact_wrench: Tensor
    """Internal left/right contact wrenches, shape [B, 12]."""

    converged: Tensor
    """Solver feasibility flag, shape [B]."""

    validated: Tensor
    """Flag after explicitly checking dynamics/contact/friction residuals."""

    iters: int
    """Number of ADMM iterations used."""

    objective: Tensor
    """QP objective value, shape [B]."""

    base_dynamics_error: Tensor
    """Infinity norm of the six floating-base dynamics residuals, shape [B]."""

    contact_acceleration_error: Tensor
    """Infinity norm of the active contact acceleration residual, shape [B]."""

    friction_violation: Tensor
    """Maximum positive wrench-inequality violation, shape [B]."""


def _batched_vector(
    value: Tensor | float | None,
    *,
    batch_size: int,
    dimension: int,
    device: torch.device,
    dtype: torch.dtype,
    default: float,
    name: str,
) -> Tensor:
    """Convert a scalar/[N]/[B,N] value to a [B,N] tensor."""

    if value is None:
        return torch.full(
            (batch_size, dimension),
            default,
            device=device,
            dtype=dtype,
        )

    result = torch.as_tensor(value, device=device, dtype=dtype)

    if result.ndim == 0:
        return result.expand(batch_size, dimension)

    if result.ndim == 1:
        if result.shape != (dimension,):
            raise ValueError(
                f"{name} must have shape [{dimension}] or "
                f"[B,{dimension}], got {tuple(result.shape)}."
            )
        return result.unsqueeze(0).expand(batch_size, -1)

    if result.ndim == 2 and result.shape == (batch_size, dimension):
        return result

    raise ValueError(
        f"{name} must be scalar, [{dimension}], or "
        f"[B,{dimension}], got {tuple(result.shape)}."
    )


def _batched_acceleration_weight(
    value: Tensor | float,
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Convert a scalar/vector/matrix acceleration weight to [B,35,35]."""

    weight = torch.as_tensor(value, device=device, dtype=dtype)
    identity = torch.eye(35, device=device, dtype=dtype)

    if weight.ndim == 0:
        return weight * identity.unsqueeze(0).expand(batch_size, -1, -1)

    if weight.ndim == 1 and weight.shape == (35,):
        return torch.diag(weight).unsqueeze(0).expand(batch_size, -1, -1)

    if weight.ndim == 2 and weight.shape == (batch_size, 35):
        return torch.diag_embed(weight)

    if weight.ndim == 2 and weight.shape == (35, 35):
        return weight.unsqueeze(0).expand(batch_size, -1, -1)

    if weight.ndim == 3 and weight.shape == (batch_size, 35, 35):
        return weight

    raise ValueError(
        "acc_weight must be scalar, [35], [B,35], [35,35], or "
        f"[B,35,35], got {tuple(weight.shape)}."
    )


def build_two_foot_wrench_pyramid(
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    friction_coefficient: float = 0.6,
    yaw_friction_coefficient: float = 0.1,
    toe_length: float = 0.12,
    heel_length: float = 0.06,
    half_width: float = 0.04,
    maximum_normal_force: float = 800.0,
) -> tuple[Tensor, Tensor]:
    """Build linearized wrench inequalities for two rectangular feet.

    The per-foot wrench is ordered as [fx, fy, fz, mx, my, mz].

    Returns
    -------
    G_lambda:
        Shape [B, 24, 12].
    b_lambda:
        Shape [B, 24].
    """

    mu = float(friction_coefficient)
    mu_yaw = float(yaw_friction_coefficient)
    toe = float(toe_length)
    heel = float(heel_length)
    width = float(half_width)
    fz_max = float(maximum_normal_force)

    if mu < 0.0 or mu_yaw < 0.0:
        raise ValueError("Friction coefficients must be non-negative.")
    if toe < 0.0 or heel < 0.0 or width < 0.0:
        raise ValueError("Foot dimensions must be non-negative.")
    if fz_max <= 0.0:
        raise ValueError("maximum_normal_force must be positive.")

    foot_G = torch.tensor(
        [
            # |fx| <= mu fz
            [1.0, 0.0, -mu, 0.0, 0.0, 0.0],
            [-1.0, 0.0, -mu, 0.0, 0.0, 0.0],
            # |fy| <= mu fz
            [0.0, 1.0, -mu, 0.0, 0.0, 0.0],
            [0.0, -1.0, -mu, 0.0, 0.0, 0.0],
            # 0 <= fz <= fz_max
            [0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            # |mx| <= half_width fz
            [0.0, 0.0, -width, 1.0, 0.0, 0.0],
            [0.0, 0.0, -width, -1.0, 0.0, 0.0],
            # -toe fz <= my <= heel fz
            [0.0, 0.0, -heel, 0.0, 1.0, 0.0],
            [0.0, 0.0, -toe, 0.0, -1.0, 0.0],
            # |mz| <= mu_yaw fz
            [0.0, 0.0, -mu_yaw, 0.0, 0.0, 1.0],
            [0.0, 0.0, -mu_yaw, 0.0, 0.0, -1.0],
        ],
        device=device,
        dtype=dtype,
    )

    foot_b = torch.zeros(12, device=device, dtype=dtype)
    foot_b[5] = fz_max

    G_lambda = torch.zeros(
        batch_size,
        24,
        12,
        device=device,
        dtype=dtype,
    )
    b_lambda = torch.zeros(
        batch_size,
        24,
        device=device,
        dtype=dtype,
    )

    G_lambda[:, 0:12, 0:6] = foot_G
    G_lambda[:, 12:24, 6:12] = foot_G
    b_lambda[:, 0:12] = foot_b
    b_lambda[:, 12:24] = foot_b

    return G_lambda, b_lambda


class G1DynamicsAccelerationQP:
    """Project a 35-D acceleration through floating-base dynamics constraints."""

    NUM_VELOCITIES = 35
    NUM_BASE_VELOCITIES = 6
    NUM_CONTACT_WRENCHES = 12
    NUM_VARIABLES = 47
    NUM_EQUALITIES = 18

    def __init__(
        self,
        *,
        max_iter: int = 200,
        rho: float = 1.0,
        rho_eq: float = 1000.0,
        eps_abs: float = 1e-4,
        eps_rel: float = 1e-4,
        adaptive_rho: bool = True,
        acc_weight: Tensor | float = 1.0,
        wrench_weight: float = 1e-6,
        hessian_regularization: float = 1e-6,
        validation_tolerance: float = 1e-3,
        friction_validation_tolerance: float = 0.25,
        friction_coefficient: float = 0.6,
        yaw_friction_coefficient: float = 0.1,
        toe_length: float = 0.12,
        heel_length: float = 0.06,
        half_width: float = 0.04,
        maximum_normal_force: float = 800.0,
    ) -> None:
        self.solver = ADMMSolver(
            max_iter=max_iter,
            rho=rho,
            rho_eq=rho_eq,
            eps_abs=eps_abs,
            eps_rel=eps_rel,
            adaptive_rho=adaptive_rho,
        )

        if wrench_weight < 0.0:
            raise ValueError("wrench_weight must be non-negative.")
        if hessian_regularization <= 0.0:
            raise ValueError("hessian_regularization must be positive.")
        if validation_tolerance <= 0.0:
            raise ValueError("validation_tolerance must be positive.")
        if friction_validation_tolerance <= 0.0:
            raise ValueError(
                "friction_validation_tolerance must be positive."
            )

        self.acc_weight = acc_weight
        self.wrench_weight = float(wrench_weight)
        self.hessian_regularization = float(hessian_regularization)
        self.validation_tolerance = float(validation_tolerance)
        self.friction_validation_tolerance = float(
            friction_validation_tolerance
        )

        self.friction_coefficient = float(friction_coefficient)
        self.yaw_friction_coefficient = float(yaw_friction_coefficient)
        self.toe_length = float(toe_length)
        self.heel_length = float(heel_length)
        self.half_width = float(half_width)
        self.maximum_normal_force = float(maximum_normal_force)

        self._warm_z: Tensor | None = None
        self._warm_lam_eq: Tensor | None = None
        self._previous_contact_mode: Tensor | None = None

    def reset(self, env_ids: Tensor | None = None) -> None:
        """Reset all warm starts or only the selected batch environments."""

        if env_ids is None:
            self._warm_z = None
            self._warm_lam_eq = None
            self._previous_contact_mode = None
            return

        if self._warm_z is not None:
            self._warm_z[env_ids] = 0.0
        if self._warm_lam_eq is not None:
            self._warm_lam_eq[env_ids] = 0.0
        if self._previous_contact_mode is not None:
            self._previous_contact_mode[env_ids] = -1

    @staticmethod
    def _validate_inputs(
        acc_reference: Tensor,
        mass_matrix: Tensor,
        bias_force: Tensor,
        contact_jacobian: Tensor,
        jdot_qdot: Tensor,
        contact_mode: Tensor,
    ) -> None:
        """Validate fixed G1 dimensions and common device/dtype requirements."""

        if acc_reference.ndim != 2 or acc_reference.shape[1] != 35:
            raise ValueError(
                f"acc_reference must have shape [B,35], got "
                f"{tuple(acc_reference.shape)}."
            )

        batch_size = acc_reference.shape[0]
        expected_shapes = {
            "mass_matrix": (batch_size, 35, 35),
            "bias_force": (batch_size, 35),
            "contact_jacobian": (batch_size, 12, 35),
            "jdot_qdot": (batch_size, 12),
            "contact_mode": (batch_size,),
        }
        tensors = {
            "mass_matrix": mass_matrix,
            "bias_force": bias_force,
            "contact_jacobian": contact_jacobian,
            "jdot_qdot": jdot_qdot,
            "contact_mode": contact_mode,
        }

        for name, expected_shape in expected_shapes.items():
            if tensors[name].shape != expected_shape:
                raise ValueError(
                    f"{name} must have shape {expected_shape}, got "
                    f"{tuple(tensors[name].shape)}."
                )
            if tensors[name].device != acc_reference.device:
                raise ValueError(
                    f"{name} is on {tensors[name].device}, but acc_reference "
                    f"is on {acc_reference.device}."
                )

        for name in (
            "mass_matrix",
            "bias_force",
            "contact_jacobian",
            "jdot_qdot",
        ):
            if tensors[name].dtype != acc_reference.dtype:
                raise ValueError(
                    f"{name} has dtype {tensors[name].dtype}, but "
                    f"acc_reference has dtype {acc_reference.dtype}."
                )

        valid_modes = (
            (contact_mode == 0)
            | (contact_mode == 1)
            | (contact_mode == 2)
            | (contact_mode == 3)
        )
        if not bool(valid_modes.all()):
            raise ValueError("contact_mode values must be 0, 1, 2, or 3.")

    @staticmethod
    def _contact_mask(contact_mode: Tensor, dtype: torch.dtype) -> Tensor:
        """Return a fixed [B,12] left/right contact-row mask."""

        left_active = (contact_mode == 1) | (contact_mode == 3)
        right_active = (contact_mode == 2) | (contact_mode == 3)

        return torch.cat(
            (
                left_active[:, None].expand(-1, 6),
                right_active[:, None].expand(-1, 6),
            ),
            dim=1,
        ).to(dtype=dtype)

    @staticmethod
    def _prepare_friction_constraints(
        friction_G: Tensor,
        friction_b: Tensor,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor]:
        """Convert [m,12]/[m] or batched friction data to batched tensors."""

        G = torch.as_tensor(friction_G, device=device, dtype=dtype)
        b = torch.as_tensor(friction_b, device=device, dtype=dtype)

        if G.ndim == 2:
            if G.shape[1] != 12:
                raise ValueError(
                    f"friction_G must have 12 columns, got {G.shape[1]}."
                )
            G = G.unsqueeze(0).expand(batch_size, -1, -1)
        elif G.ndim != 3 or G.shape[0] != batch_size or G.shape[2] != 12:
            raise ValueError(
                "friction_G must have shape [m,12] or [B,m,12], got "
                f"{tuple(G.shape)}."
            )

        num_inequalities = G.shape[1]

        if b.ndim == 1:
            if b.shape != (num_inequalities,):
                raise ValueError(
                    f"friction_b must have shape [{num_inequalities}], got "
                    f"{tuple(b.shape)}."
                )
            b = b.unsqueeze(0).expand(batch_size, -1)
        elif b.shape != (batch_size, num_inequalities):
            raise ValueError(
                "friction_b must have shape [m] or [B,m], got "
                f"{tuple(b.shape)}."
            )

        return G, b

    @torch.no_grad()
    def solve(
        self,
        *,
        acc_reference: Tensor,
        mass_matrix: Tensor,
        bias_force: Tensor,
        contact_jacobian: Tensor,
        jdot_qdot: Tensor,
        contact_mode: Tensor,
        acc_lower: Tensor | float | None = None,
        acc_upper: Tensor | float | None = None,
        friction_G: Tensor | None = None,
        friction_b: Tensor | None = None,
    ) -> DynamicsAccelerationSolution:
        """Solve for the closest dynamics-feasible 35-D acceleration.

        Parameters
        ----------
        acc_reference:
            Regularized/dynamically consistent FABRICS acceleration [B,35].
        mass_matrix:
            Generalized mass matrix M(q), shape [B,35,35].
        bias_force:
            Generalized Coriolis/centrifugal/gravity vector h(q,nu), [B,35].
        contact_jacobian:
            Fixed left/right contact stack [B,12,35].
        jdot_qdot:
            Convective contact acceleration Jdot_c @ nu, shape [B,12].
        contact_mode:
            [B] integer tensor: 0 flight, 1 left, 2 right, 3 double support.
        acc_lower, acc_upper:
            Optional scalar/[35]/[B,35] acceleration bounds.
        friction_G, friction_b:
            Optional custom wrench inequalities
            G_lambda lambda <= b_lambda. Both must be provided together.

        Returns
        -------
        DynamicsAccelerationSolution
            Use ``result.acc`` as the final [B,35] output.
        """

        self._validate_inputs(
            acc_reference,
            mass_matrix,
            bias_force,
            contact_jacobian,
            jdot_qdot,
            contact_mode,
        )

        batch_size = acc_reference.shape[0]
        device = acc_reference.device
        dtype = acc_reference.dtype

        if (friction_G is None) != (friction_b is None):
            raise ValueError(
                "friction_G and friction_b must either both be provided or "
                "both be None."
            )

        contact_mode = contact_mode.to(device=device, dtype=torch.long)
        contact_mask = self._contact_mask(contact_mode, dtype)

        Jc = contact_jacobian * contact_mask.unsqueeze(-1)
        Jdot_nu = jdot_qdot * contact_mask

        # Only the first six rows are unactuated floating-base dynamics.
        M_base = mass_matrix[:, 0:6, :]
        h_base = bias_force[:, 0:6]
        J_base_transpose = Jc[:, :, 0:6].transpose(1, 2)

        # -------------------------------------------------------------
        # Objective:
        #   0.5 ||acc - acc_reference||_W^2
        #   + 0.5*w_lambda ||lambda||^2
        # -------------------------------------------------------------

        W_acc = _batched_acceleration_weight(
            self.acc_weight,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )

        H = torch.zeros(
            batch_size,
            self.NUM_VARIABLES,
            self.NUM_VARIABLES,
            device=device,
            dtype=dtype,
        )
        linear_term = torch.zeros(
            batch_size,
            self.NUM_VARIABLES,
            device=device,
            dtype=dtype,
        )

        H[:, 0:35, 0:35] = W_acc
        linear_term[:, 0:35] = -torch.bmm(
            W_acc,
            acc_reference.unsqueeze(-1),
        ).squeeze(-1)

        wrench_identity = torch.eye(
            12,
            device=device,
            dtype=dtype,
        ).unsqueeze(0)
        H[:, 35:47, 35:47] = self.wrench_weight * wrench_identity

        full_identity = torch.eye(
            self.NUM_VARIABLES,
            device=device,
            dtype=dtype,
        ).unsqueeze(0)
        H = H + self.hessian_regularization * full_identity
        H = 0.5 * (H + H.transpose(1, 2))

        # -------------------------------------------------------------
        # Equality constraints:
        #
        #   M_base acc - J_base^T lambda = -h_base
        #   Jc acc = -Jdot_c nu
        # -------------------------------------------------------------

        A_eq = torch.zeros(
            batch_size,
            self.NUM_EQUALITIES,
            self.NUM_VARIABLES,
            device=device,
            dtype=dtype,
        )
        b_eq = torch.zeros(
            batch_size,
            self.NUM_EQUALITIES,
            device=device,
            dtype=dtype,
        )

        A_eq[:, 0:6, 0:35] = M_base
        A_eq[:, 0:6, 35:47] = -J_base_transpose
        b_eq[:, 0:6] = -h_base

        A_eq[:, 6:18, 0:35] = Jc
        b_eq[:, 6:18] = -Jdot_nu

        # A masked contact row would otherwise be the degenerate equality
        # 0 = 0. Use that row to constrain the corresponding inactive wrench
        # component to zero. This keeps the fixed-size batched equality
        # system well-conditioned in flight and single-support modes.
        inactive_contact = 1.0 - contact_mask
        A_eq[:, 6:18, 35:47] = torch.diag_embed(inactive_contact)

        # -------------------------------------------------------------
        # Wrench inequalities:
        #
        #   G_lambda lambda <= b_lambda
        # -------------------------------------------------------------

        if friction_G is None:
            G_lambda, b_lambda = build_two_foot_wrench_pyramid(
                batch_size=batch_size,
                device=device,
                dtype=dtype,
                friction_coefficient=self.friction_coefficient,
                yaw_friction_coefficient=self.yaw_friction_coefficient,
                toe_length=self.toe_length,
                heel_length=self.heel_length,
                half_width=self.half_width,
                maximum_normal_force=self.maximum_normal_force,
            )
        else:
            G_lambda, b_lambda = self._prepare_friction_constraints(
                friction_G,
                friction_b,
                batch_size=batch_size,
                device=device,
                dtype=dtype,
            )

        num_inequalities = G_lambda.shape[1]
        G = torch.zeros(
            batch_size,
            num_inequalities,
            self.NUM_VARIABLES,
            device=device,
            dtype=dtype,
        )
        G[:, :, 35:47] = G_lambda

        # -------------------------------------------------------------
        # Box bounds:
        #
        #   acc_lower <= acc <= acc_upper
        #   lambda_inactive = 0
        # -------------------------------------------------------------

        lower = torch.full(
            (batch_size, self.NUM_VARIABLES),
            -1e6,
            device=device,
            dtype=dtype,
        )
        upper = torch.full(
            (batch_size, self.NUM_VARIABLES),
            1e6,
            device=device,
            dtype=dtype,
        )

        lower[:, 0:35] = _batched_vector(
            acc_lower,
            batch_size=batch_size,
            dimension=35,
            device=device,
            dtype=dtype,
            default=-1e6,
            name="acc_lower",
        )
        upper[:, 0:35] = _batched_vector(
            acc_upper,
            batch_size=batch_size,
            dimension=35,
            device=device,
            dtype=dtype,
            default=1e6,
            name="acc_upper",
        )

        if bool((lower[:, 0:35] > upper[:, 0:35]).any()):
            raise ValueError("acc_lower cannot exceed acc_upper.")

        wrench_active = contact_mask.bool()
        zero_wrench = torch.zeros(
            batch_size,
            12,
            device=device,
            dtype=dtype,
        )
        lower[:, 35:47] = torch.where(
            wrench_active,
            lower[:, 35:47],
            zero_wrench,
        )
        upper[:, 35:47] = torch.where(
            wrench_active,
            upper[:, 35:47],
            zero_wrench,
        )

        qp = QPData(
            H=H,
            h=linear_term,
            G=G,
            b=b_lambda,
            lb=lower,
            ub=upper,
            A_eq=A_eq,
            b_eq=b_eq,
        )

        # -------------------------------------------------------------
        # Warm start. Reset changed contact modes to prevent a stale
        # contact wrench/equality multiplier from entering the next solve.
        # -------------------------------------------------------------

        warm_z = None
        warm_lam_eq = None

        if self._warm_z is not None and self._warm_z.shape == (
            batch_size,
            self.NUM_VARIABLES,
        ):
            warm_z = self._warm_z.clone()
            warm_z[:, 35:47] *= contact_mask

        if (
            self._warm_lam_eq is not None
            and self._warm_lam_eq.shape
            == (batch_size, self.NUM_EQUALITIES)
        ):
            warm_lam_eq = self._warm_lam_eq.clone()

        if (
            self._previous_contact_mode is not None
            and self._previous_contact_mode.shape == contact_mode.shape
        ):
            changed = self._previous_contact_mode != contact_mode
            if warm_z is not None:
                warm_z[changed, 35:47] = 0.0
            if warm_lam_eq is not None:
                warm_lam_eq[changed] = 0.0

        solution = self.solver.solve(
            qp,
            warm_z=warm_z,
            warm_lam_eq=warm_lam_eq,
        )

        final_acc = solution.z[:, 0:35]
        final_wrench = solution.z[:, 35:47]

        # Explicit residual checks are important because the referenced
        # admm_qp.py reports primal feasibility, not a full KKT certificate.
        base_residual = (
            torch.bmm(M_base, final_acc.unsqueeze(-1)).squeeze(-1)
            + h_base
            - torch.bmm(
                J_base_transpose,
                final_wrench.unsqueeze(-1),
            ).squeeze(-1)
        )
        contact_residual = (
            torch.bmm(Jc, final_acc.unsqueeze(-1)).squeeze(-1)
            + Jdot_nu
        )
        friction_residual = (
            torch.bmm(
                G_lambda,
                final_wrench.unsqueeze(-1),
            ).squeeze(-1)
            - b_lambda
        )

        base_error = base_residual.abs().amax(dim=1)
        contact_error = contact_residual.abs().amax(dim=1)
        friction_violation = torch.clamp_min(
            friction_residual,
            0.0,
        ).amax(dim=1)

        tolerance = self.validation_tolerance
        box_violation = torch.maximum(
            torch.clamp_min(lower - solution.z, 0.0).amax(dim=1),
            torch.clamp_min(solution.z - upper, 0.0).amax(dim=1),
        )
        validated = (
            torch.isfinite(solution.z).all(dim=1)
            & (base_error <= tolerance)
            & (contact_error <= tolerance)
            & (
                friction_violation
                <= self.friction_validation_tolerance
            )
            & (box_violation <= tolerance)
        )

        self._warm_z = solution.z.clone()
        self._warm_lam_eq = (
            solution.lam_eq.clone()
            if solution.lam_eq is not None
            else None
        )
        self._previous_contact_mode = contact_mode.clone()

        return DynamicsAccelerationSolution(
            acc=final_acc,
            contact_wrench=final_wrench,
            converged=solution.converged,
            validated=validated,
            iters=solution.iters,
            objective=solution.obj,
            base_dynamics_error=base_error,
            contact_acceleration_error=contact_error,
            friction_violation=friction_violation,
        )

    @torch.no_grad()
    def compute_acceleration(self, **kwargs) -> Tensor:
        """Return only the final [B,35] acceleration."""

        return self.solve(**kwargs).acc


def example_usage() -> None:
    """Minimal shape example; replace all tensors with simulator values."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    batch_size = 128

    qp = G1DynamicsAccelerationQP(
        max_iter=200,
        rho=1.0,
        rho_eq=1000.0,
        acc_weight=1.0,
        wrench_weight=1e-6,
    )

    # These are placeholders only. Use the actual G1 rigid-body quantities.
    acc_dc = torch.zeros(batch_size, 35, device=device, dtype=dtype)
    mass_matrix = torch.eye(
        35,
        device=device,
        dtype=dtype,
    ).unsqueeze(0).expand(batch_size, -1, -1).clone()
    bias_force = torch.zeros(batch_size, 35, device=device, dtype=dtype)
    contact_jacobian = torch.zeros(
        batch_size,
        12,
        35,
        device=device,
        dtype=dtype,
    )
    jdot_qdot = torch.zeros(batch_size, 12, device=device, dtype=dtype)

    # 0 flight, 1 left support, 2 right support, 3 double support.
    contact_mode = torch.full(
        (batch_size,),
        3,
        device=device,
        dtype=torch.long,
    )

    final_acc = qp.compute_acceleration(
        acc_reference=acc_dc,
        mass_matrix=mass_matrix,
        bias_force=bias_force,
        contact_jacobian=contact_jacobian,
        jdot_qdot=jdot_qdot,
        contact_mode=contact_mode,
        acc_lower=-100.0,
        acc_upper=100.0,
    )

    print("final_acc:", final_acc.shape)


if __name__ == "__main__":
    example_usage()
