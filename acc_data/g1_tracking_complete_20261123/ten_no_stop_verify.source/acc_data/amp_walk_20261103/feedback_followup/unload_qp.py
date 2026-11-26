from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor

from fabrics_sim.solver.admm_qp import ADMMSolver, QPData
from fabrics_sim.fabrics.task_request import TaskRequest, task_objective


@dataclass
class DynamicsAccelerationSolution:
    """Result of the dynamics-feasible acceleration QP."""

    acc: Tensor
    """Command acceleration, shape [B,35]; integrate this output."""

    contact_wrench: Tensor
    """Predicted total left/right wrenches including servo feedback, [B,12]."""

    joint_torque: Tensor
    """Additional actuator torque optimized by the QP, shape [B, 29]."""

    total_joint_torque: Tensor
    """Known feedback plus optimized actuator torque, shape [B, 29]."""

    predicted_acc: Tensor
    """Acceleration including the explicit servo response at the solve snapshot."""

    nominal_acc: Tensor
    """Physical inverse-dynamics acceleration before PD; excludes command synchronization."""

    nominal_contact_wrench: Tensor
    """Contact wrench before the explicit servo response."""

    converged: Tensor
    """Solver feasibility flag, shape [B]."""

    validated: Tensor
    """Flag after explicitly checking physical constraints and optional task locks."""

    iters: int
    """Number of ADMM iterations used."""

    objective: Tensor
    """QP objective value, shape [B]."""

    base_dynamics_error: Tensor
    """Infinity norm of the six floating-base dynamics residuals, shape [B]."""

    joint_dynamics_error: Tensor
    """Infinity norm of the 29 actuated dynamics residuals, shape [B]."""

    contact_acceleration_error: Tensor
    """Infinity norm of the active contact acceleration residual, shape [B]."""

    friction_violation: Tensor
    """Maximum positive wrench-inequality violation, shape [B]."""

    effort_violation: Tensor
    """Maximum positive actuated-joint effort violation, shape [B]."""

    bound_violation: Tensor
    """Maximum violation of command boxes or predicted acceleration bounds."""

    task_residuals: dict[str, Tensor] | None = None
    """Active task residuals evaluated at predicted physical acceleration."""

    joint_stop_reaction: Tensor | None = None
    """Signed passive reaction at observed stationary joint stops, [B,29]."""


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


def build_single_support_wrench_constraints(
    *,
    contact_mode: Tensor,
    foot_rotation_world: Tensor,
    friction_coefficient: float = 0.6,
    yaw_friction_coefficient: float = 0.1,
    toe_length: float = 0.12,
    heel_length: float = 0.06,
    half_width: float = 0.04,
    minimum_normal_force: float = 20.0,
    maximum_normal_force: float = 800.0,
    sole_depth: float = 0.0,
    coupled_yaw: bool = False,
) -> tuple[Tensor, Tensor]:
    """Build world-frame wrench inequalities from each active foot's frame.

    Wrenches used by the dynamics QP are expressed in world coordinates at
    each ankle origin. CoP and friction limits, however, are meaningful in the
    foot frame. This function rotates each active single-support inequality
    into world coordinates. Double support constrains both feet with zero
    minimum load, allowing load transfer. Flight receives zero rows.
    sole_depth is the positive distance below the ankle to the contact plane.
    Moments are shifted to that plane before applying the CoP bounds.
    coupled_yaw adds the rectangular contact-wrench cone's coupled yaw bounds
    (Caron et al., https://arxiv.org/abs/1501.04719, Proposition 2), using an
    inscribed square approximation of circular Coulomb friction.
    """
    if contact_mode.ndim != 1:
        raise ValueError("contact_mode must have shape [B].")
    batch_size = contact_mode.shape[0]
    if foot_rotation_world.shape != (batch_size, 2, 3, 3):
        raise ValueError(
            "foot_rotation_world must have shape [B,2,3,3], got "
            f"{tuple(foot_rotation_world.shape)}."
        )
    if minimum_normal_force < 0.0:
        raise ValueError("minimum_normal_force must be nonnegative.")
    if maximum_normal_force <= minimum_normal_force:
        raise ValueError(
            "maximum_normal_force must exceed minimum_normal_force."
        )

    device = foot_rotation_world.device
    dtype = foot_rotation_world.dtype
    mu = float(friction_coefficient)
    mu_yaw = float(yaw_friction_coefficient)
    toe = float(toe_length)
    heel = float(heel_length)
    width = float(half_width)
    if min(mu, mu_yaw, toe, heel, width) < 0.0:
        raise ValueError("Friction and foot dimensions must be nonnegative.")
    if not math.isfinite(sole_depth) or sole_depth < 0.0:
        raise ValueError("sole_depth must be finite and nonnegative.")
    if coupled_yaw:
        mu /= math.sqrt(2.)

    local_G = torch.tensor(
        [
            [1.0, 0.0, -mu, 0.0, 0.0, 0.0],
            [-1.0, 0.0, -mu, 0.0, 0.0, 0.0],
            [0.0, 1.0, -mu, 0.0, 0.0, 0.0],
            [0.0, -1.0, -mu, 0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -width, 1.0, 0.0, 0.0],
            [0.0, 0.0, -width, -1.0, 0.0, 0.0],
            [0.0, 0.0, -heel, 0.0, 1.0, 0.0],
            [0.0, 0.0, -toe, 0.0, -1.0, 0.0],
            [0.0, 0.0, -mu_yaw, 0.0, 0.0, 1.0],
            [0.0, 0.0, -mu_yaw, 0.0, 0.0, -1.0],
        ],
        device=device,
        dtype=dtype,
    )
    if coupled_yaw:
        half_length=(toe+heel)/2.
        center=(toe-heel)/2.
        rows=[]
        for yaw_sign in (-1.,1.):
            for sx in (-1.,1.):
                for sy in (-1.,1.):
                    rows.append([sx*width,sy*half_length,-mu*(half_length+width),
                                 yaw_sign*sx*mu,yaw_sign*sy*mu,yaw_sign])
        coupled=torch.tensor(rows,device=device,dtype=dtype)
        # Move center moments to sole moments at ankle XY: My_c=My+c*Fz,
        # Mz_c=Mz-c*Fy. The shared depth shift below then moves to the ankle.
        coupled[:,2]+=center*coupled[:,4]
        coupled[:,1]-=center*coupled[:,5]
        local_G=torch.cat((local_G,coupled),dim=0)
    # m_sole = m_ankle - r_sole x f, r_sole = [0, 0, -depth].
    local_G[:, 0] += sole_depth * local_G[:, 4]
    local_G[:, 1] -= sole_depth * local_G[:, 3]
    rows_per_foot=local_G.shape[0]
    local_b = torch.zeros(rows_per_foot, device=device, dtype=dtype)
    local_b[4] = -float(minimum_normal_force)
    local_b[5] = float(maximum_normal_force)

    G_lambda = torch.zeros(batch_size, 2*rows_per_foot, 12, device=device, dtype=dtype)
    b_lambda = torch.zeros(batch_size, 2*rows_per_foot, device=device, dtype=dtype)
    for foot_index, mode_value in enumerate((1, 2)):
        rotation = foot_rotation_world[:, foot_index]
        force_rows_world = torch.matmul(
            local_G[None, :, 0:3], rotation.transpose(1, 2)
        )
        moment_rows_world = torch.matmul(
            local_G[None, :, 3:6], rotation.transpose(1, 2)
        )
        world_G = torch.cat((force_rows_world, moment_rows_world), dim=-1)
        active = ((contact_mode == mode_value) | (contact_mode == 3)).to(dtype=dtype)
        row_slice = slice(rows_per_foot * foot_index, rows_per_foot * (foot_index + 1))
        wrench_slice = slice(6 * foot_index, 6 * (foot_index + 1))
        G_lambda[:, row_slice, wrench_slice] = (
            active[:, None, None] * world_G
        )
        b_lambda[:, row_slice] = active[:, None] * local_b[None, :]
        # Only a sole support foot must carry the minimum normal load.
        b_lambda[:, rows_per_foot * foot_index + 4] = (
            -(contact_mode == mode_value).to(dtype=dtype)
            * float(minimum_normal_force)
        )

    return G_lambda, b_lambda


class G1DynamicsAccelerationQP:
    """Project a 35-D acceleration through floating-base dynamics constraints."""

    NUM_VELOCITIES = 35
    NUM_BASE_VELOCITIES = 6
    NUM_CONTACT_WRENCHES = 12
    NUM_JOINT_TORQUES = 29
    NUM_VARIABLES = 76
    NUM_EQUALITIES = 47

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
        torque_weight: float = 1e-6,
        hessian_regularization: float = 1e-6,
        validation_tolerance: float = 1e-3,
        friction_validation_tolerance: float = 0.25,
        effort_validation_tolerance: float = 0.25,
        box_validation_tolerance: float = 0.5,
        friction_coefficient: float = 0.6,
        yaw_friction_coefficient: float = 0.1,
        toe_length: float = 0.12,
        heel_length: float = 0.06,
        half_width: float = 0.04,
        maximum_normal_force: float = 800.0,
        solver_backend: str = "admm",
    ) -> None:
        self.solver = ADMMSolver(
            max_iter=max_iter,
            rho=rho,
            rho_eq=rho_eq,
            eps_abs=eps_abs,
            eps_rel=eps_rel,
            adaptive_rho=adaptive_rho,
        )
        if solver_backend == "osqp":
            from fabrics_sim.solver.osqp_backend import OSQPBackend
            # Near a straight support knee, saved physical QPs can remain
            # feasible but need more than 20k iterations to reach the physical
            # residual budget. This caps solver work, never relaxes validation.
            self.solver = OSQPBackend(max_iter=max(100000, max_iter),
                                      eps_abs=min(eps_abs, 1e-6), eps_rel=min(eps_rel, 1e-6))
        elif solver_backend == "daqp":
            from fabrics_sim.solver.daqp_backend import DAQPBackend
            self.solver = DAQPBackend()
        elif solver_backend != "admm":
            raise ValueError('Unknown QP solver backend')
        self.solver_backend = solver_backend

        if wrench_weight < 0.0:
            raise ValueError("wrench_weight must be non-negative.")
        if torque_weight < 0.0:
            raise ValueError("torque_weight must be non-negative.")
        if hessian_regularization <= 0.0:
            raise ValueError("hessian_regularization must be positive.")
        if validation_tolerance <= 0.0:
            raise ValueError("validation_tolerance must be positive.")
        if friction_validation_tolerance <= 0.0:
            raise ValueError(
                "friction_validation_tolerance must be positive."
            )
        if effort_validation_tolerance <= 0.0:
            raise ValueError(
                "effort_validation_tolerance must be positive."
            )
        if box_validation_tolerance <= 0.0:
            raise ValueError("box_validation_tolerance must be positive.")

        self.acc_weight = acc_weight
        self.wrench_weight = float(wrench_weight)
        self.torque_weight = float(torque_weight)
        self.hessian_regularization = float(hessian_regularization)
        self.validation_tolerance = float(validation_tolerance)
        self.friction_validation_tolerance = float(
            friction_validation_tolerance
        )
        self.effort_validation_tolerance = float(
            effort_validation_tolerance
        )
        self.box_validation_tolerance = float(box_validation_tolerance)

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
        if self.solver_backend in ('osqp','daqp'):
            self.solver.reset(env_ids)
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
        effort_limits: Tensor | float | None = None,
        known_joint_torque: Tensor | float | None = None,
        servo_bias: Tensor | None = None,
        servo_acc_gain: Tensor | None = None,
        command_acc_correction: Tensor | None = None,
        contact_wrench_reference: Tensor | None = None,
        contact_wrench_weights: Tensor | None = None,
        track_predicted_base: bool = False,
        predicted_acc_limit: Tensor | float | None = None,
        predicted_acc_lower: Tensor | None = None,
        predicted_acc_upper: Tensor | None = None,
        friction_G: Tensor | None = None,
        friction_b: Tensor | None = None,
        task_requests: list[TaskRequest] | None = None,
        task_tracks_predicted: bool = True,
        contact_row_mask: Tensor | None = None,
        priority_tasks: tuple[str, ...] = (),
        priority_requests: list[TaskRequest] | None = None,
        priority_levels: tuple[tuple[str, ...], ...] | None = None,
        priority_tolerance: float = .02,
        priority_rows: dict[str, tuple[int, ...]] | None = None,
        priority_projections: dict[str, Tensor] | None = None,
        acceleration_constraints: tuple[Tensor, Tensor] | None = None,
        joint_stop_directions: Tensor | None = None,
    ) -> DynamicsAccelerationSolution:
        """Solve for the closest dynamics-feasible 35-D acceleration.

        ``priority_tasks`` selects an optional first optimization stage. The
        second stage keeps its achieved task accelerations within
        ``priority_tolerance`` (in each task row's physical units). Hard
        dynamics, contact, friction, effort and acceleration limits apply to
        both stages. This preserves the numerically achieved first-stage
        result, not a certificate of the exact lexicographic optimum.

        ``priority_requests`` optionally supplies separate first-stage leaves;
        they are not added again to the final task objective.

        ``priority_levels`` optionally partitions ``priority_tasks`` into
        ordered stages. Every later stage retains ALL previous output locks.
        ``priority_rows`` optionally selects coordinates within a priority
        task (e.g. COM height only). Other rows retain their normal objective.
        ``priority_projections`` selects orthonormal task directions instead
        (e.g. heading-relative lateral COM). The full objective is unchanged.
        ``acceleration_constraints=(C,d)`` imposes C @ predicted_acc <= d
        in physical coordinates, including any affine servo response.

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
        contact_row_mask:
            Optional boolean [B,12]. False releases that contact coordinate
            and sets its dual wrench component to zero. Foot-level modes still
            apply. Jacobians, biases and wrench inequalities must all use the
            same origin and coordinate basis (e.g. a fixed frame at a toe line).
        acc_lower, acc_upper:
            Optional scalar/[35]/[B,35] acceleration bounds.
        effort_limits:
            Optional positive scalar/[29]/[B,29] symmetric joint-torque
            limits on the total actuator torque. These are imposed inside the
            QP, before command clipping.
        known_joint_torque:
            Optional scalar/[29]/[B,29] feedback torque that the simulator or
            robot drive adds to the optimized effort. The QP includes it in
            both dynamics and the total-effort bounds; ``joint_torque`` in the
            result is then the additional effort command.
        servo_bias, servo_acc_gain:
            Optional explicit servo model tau_PD = bias + gain * acc_j.
            Both have shape [B,29] and must be supplied together. Unlike
            known_joint_torque, this feedback is NOT canceled: ``acc`` is
            command acceleration; ``predicted_acc`` includes the
            contact-consistent physical response to PD. Force/effort limits
            constrain predicted TOTAL wrench/torque. This local model assumes
            unchanged contact and is not a full 20 ms rollout.
        command_acc_correction:
            Optional finite [B,35] command-integrator synchronization, with zero
            base rows. The decision ``acc`` is command acceleration; inverse
            dynamics uses ``nominal_acc = acc - command_acc_correction``.
            Pass physical Fabric reference + correction as acc_reference.
            This keeps synchronization inside command/effort constraints without
            projecting it out through physical foot-contact constraints.
        contact_wrench_reference:
            Optional finite [B,12] target for TOTAL world-frame contact wrench.
            When supplied, wrench regularization tracks this target including
            servo response, instead of minimizing nominal wrench toward zero.
        contact_wrench_weights:
            Optional nonnegative finite [B,12] diagonal weights multiplying
            wrench_weight. Requires a wrench reference. Zero rows are untracked.
        track_predicted_base:
            With a servo model, track predicted physical base acceleration
            instead of nominal base acceleration in the objective. Joint rows
            still track command acceleration for integration. This
            accounts for servo-induced base motion without globally canceling
            joint feedback. Requires both servo tensors.
        task_requests:
            Optional physical task intentions. Replaces acceleration-reference
            tracking with sum ||A*predicted_acc+bias-desired||_W^2 for all rows,
            including the full servo response and excluding command correction.
            Priorities are soft metric preferences. Hard constraints are unchanged.
        task_tracks_predicted:
            When false, task objectives track nominal physical acceleration,
            excluding servo response but still excluding command correction.
            Total-effort constraints and reported predicted task residuals
            continue to include the servo response.
        predicted_acc_limit:
            Optional positive scalar/[35]/[B,35] bound on acceleration AFTER
            servo response, separate from the nominal command bounds.
        predicted_acc_lower, predicted_acc_upper:
            Optional paired asymmetric bounds on physical acceleration after
            servo response and excluding command synchronization correction.
            Intersected with predicted_acc_limit when both are supplied.
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
        # Rank-deficient A.T W A can acquire negative eigenvalues if formed
        # in float32 before a CPU backend converts it to double. Accumulate
        # the objective in the backend's precision, retaining physical input
        # and output dtypes and the same regularization/constraints.
        objective_dtype = torch.float64 if self.solver_backend in ('daqp', 'osqp') else dtype

        if (friction_G is None) != (friction_b is None):
            raise ValueError(
                "friction_G and friction_b must either both be provided or "
                "both be None."
            )

        contact_mode = contact_mode.to(device=device, dtype=torch.long)
        contact_mask = self._contact_mask(contact_mode, dtype)
        if contact_row_mask is not None:
            if (contact_row_mask.shape != (batch_size, 12) or contact_row_mask.dtype != torch.bool
                    or contact_row_mask.device != device):
                raise ValueError("contact_row_mask must be boolean [B,12] on the input device.")
            # A released contact coordinate also releases its dual wrench
            # component; the equality below fixes that component to zero.
            contact_mask = contact_mask * contact_row_mask.to(dtype)
        known_torque = _batched_vector(
            known_joint_torque,
            batch_size=batch_size,
            dimension=self.NUM_JOINT_TORQUES,
            device=device,
            dtype=dtype,
            default=0.0,
            name="known_joint_torque",
        )
        if not bool(torch.isfinite(known_torque).all()):
            raise ValueError("known_joint_torque must be finite.")

        Jc = contact_jacobian * contact_mask.unsqueeze(-1)
        Jdot_nu = jdot_qdot * contact_mask
        correction = torch.zeros_like(acc_reference)
        if command_acc_correction is not None:
            if (command_acc_correction.shape != acc_reference.shape or
                    command_acc_correction.device != device or command_acc_correction.dtype != dtype or
                    not torch.isfinite(command_acc_correction).all() or
                    command_acc_correction[:, :6].count_nonzero()):
                raise ValueError("command_acc_correction must be finite [B,35], match input dtype/device, and have zero base rows.")
            correction = command_acc_correction

        has_servo = servo_bias is not None or servo_acc_gain is not None
        if track_predicted_base and not has_servo:
            raise ValueError("track_predicted_base requires a servo model.")
        if has_servo:
            if servo_bias is None or servo_acc_gain is None or known_joint_torque is not None:
                raise ValueError("Supply both servo tensors and omit known_joint_torque.")
            for value in (servo_bias, servo_acc_gain):
                if value.shape != (batch_size, 29) or value.device != device or value.dtype != dtype or not torch.isfinite(value).all():
                    raise ValueError("Servo tensors must be finite [B,29] with the input device/dtype.")
            if (servo_acc_gain < 0).any():
                raise ValueError("servo_acc_gain must be nonnegative.")
            # M da = S^T tau_PD + J^T dlambda, J da = 0.
            inv_mass = torch.linalg.inv(mass_matrix)
            mobility = Jc @ inv_mass @ Jc.transpose(1, 2)
            mobility = mobility + torch.diag_embed(1 - contact_mask)
            # Active foot rows are required to be independent. A rank-deficient
            # contact model must fail rather than invent a response.
            wrench_response = -torch.linalg.solve(mobility, (Jc @ inv_mass)[:, :, 6:])
            acceleration_response = inv_mass[:, :, 6:] + inv_mass @ Jc.transpose(1, 2) @ wrench_response

        # -------------------------------------------------------------
        # Objective:
        #   0.5 ||acc - acc_reference||_W^2
        #   + 0.5*w_lambda ||lambda||^2
        #   + 0.5*w_tau ||tau||^2
        # With track_predicted_base, replace the first six output rows by
        # predicted physical acceleration (affine in nominal acceleration).
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
            dtype=objective_dtype,
        )
        linear_term = torch.zeros(
            batch_size,
            self.NUM_VARIABLES,
            device=device,
            dtype=objective_dtype,
        )

        H[:, 0:35, 0:35] = W_acc
        linear_term[:, 0:35] = -torch.bmm(
            W_acc,
            acc_reference.unsqueeze(-1),
        ).squeeze(-1)

        if track_predicted_base:
            # Objective output y = T a_nom + offset. Only base rows include
            # the contact-consistent servo response; joint rows remain nominal.
            transform = torch.eye(35, device=device, dtype=dtype).expand(
                batch_size, -1, -1
            ).clone()
            transform[:, :6, 6:] += (
                acceleration_response[:, :6] * servo_acc_gain[:, None, :]
            )
            offset = torch.zeros_like(acc_reference)
            offset[:, :6] = (
                acceleration_response[:, :6] @ servo_bias.unsqueeze(-1)
            ).squeeze(-1)
            weighted_transform = transform.transpose(1, 2) @ W_acc
            H[:, :35, :35] = weighted_transform @ transform
            linear_term[:, :35] = (
                weighted_transform @ (offset - acc_reference).unsqueeze(-1)
            ).squeeze(-1)

        if task_requests is not None:
            # Replace joint acceleration tracking, rather than counting both
            # the collapsed Fabric acceleration and its leaf intentions.
            # Physical prediction = a_command - correction + response @ PD.
            transform = torch.eye(35, device=device, dtype=dtype).expand(batch_size, -1, -1).clone()
            offset = -correction
            if has_servo and task_tracks_predicted:
                transform[:, :, 6:] += acceleration_response * servo_acc_gain[:, None, :]
                offset = offset + (acceleration_response @ servo_bias.unsqueeze(-1)).squeeze(-1)
            H[:, :35, :35], linear_term[:, :35] = task_objective(
                task_requests, transform, offset, accumulation_dtype=objective_dtype)

        wrench_identity = torch.eye(
            12,
            device=device,
            dtype=dtype,
        ).unsqueeze(0)
        H[:, 35:47, 35:47] = self.wrench_weight * wrench_identity
        if contact_wrench_weights is not None:
            if (contact_wrench_reference is None or contact_wrench_weights.shape != (batch_size, 12)
                    or contact_wrench_weights.device != device or contact_wrench_weights.dtype != dtype
                    or not torch.isfinite(contact_wrench_weights).all()
                    or (contact_wrench_weights < 0).any()):
                raise ValueError("contact_wrench_weights requires a reference and nonnegative finite [B,12] weights.")
        if contact_wrench_reference is not None:
            if (contact_wrench_reference.shape != (batch_size, 12) or
                    contact_wrench_reference.device != device or contact_wrench_reference.dtype != dtype or
                    not torch.isfinite(contact_wrench_reference).all()):
                raise ValueError("contact_wrench_reference must be finite [B,12] matching input device/dtype.")
            wrench_map = torch.zeros(batch_size, 12, self.NUM_VARIABLES, device=device, dtype=dtype)
            wrench_map[:, :, 35:47] = wrench_identity
            wrench_offset = torch.zeros_like(contact_wrench_reference)
            if has_servo:
                wrench_map[:, :, 6:35] = wrench_response * servo_acc_gain[:, None, :]
                wrench_offset = (wrench_response @ servo_bias.unsqueeze(-1)).squeeze(-1)
            H[:, 35:47, 35:47] = 0.
            weighted_map = wrench_map if contact_wrench_weights is None else wrench_map * contact_wrench_weights[..., None]
            H += self.wrench_weight * (wrench_map.transpose(1, 2) @ weighted_map)
            linear_term += self.wrench_weight * (weighted_map.transpose(1, 2) @
                (wrench_offset - contact_wrench_reference).unsqueeze(-1)).squeeze(-1)
        torque_identity = torch.eye(
            self.NUM_JOINT_TORQUES,
            device=device,
            dtype=dtype,
        ).unsqueeze(0)
        H[:, 47:76, 47:76] = self.torque_weight * torque_identity

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
        #   M (acc-correction) - Jc^T lambda - S^T tau_qp = -h + S^T tau_known
        #   Jc (acc-correction) = -Jdot_c nu
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

        A_eq[:, 0:35, 0:35] = mass_matrix
        A_eq[:, 0:35, 35:47] = -Jc.transpose(1, 2)
        A_eq[:, 6:35, 47:76] = -torch.eye(
            self.NUM_JOINT_TORQUES,
            device=device,
            dtype=dtype,
        ).unsqueeze(0)
        b_eq[:, 0:35] = -bias_force
        b_eq[:, 0:35] += (mass_matrix @ correction.unsqueeze(-1)).squeeze(-1)
        b_eq[:, 6:35] += known_torque

        A_eq[:, 35:47, 0:35] = Jc
        b_eq[:, 35:47] = -Jdot_nu
        b_eq[:, 35:47] += (Jc @ correction.unsqueeze(-1)).squeeze(-1)

        # A masked contact row would otherwise be the degenerate equality
        # 0 = 0. Use that row to constrain the corresponding inactive wrench
        # component to zero. This keeps the fixed-size batched equality
        # system well-conditioned in flight and single-support modes.
        inactive_contact = 1.0 - contact_mask
        A_eq[:, 35:47, 35:47] = torch.diag_embed(inactive_contact)

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

        num_friction_inequalities = G_lambda.shape[1]
        G = torch.zeros(
            batch_size,
            num_friction_inequalities,
            self.NUM_VARIABLES,
            device=device,
            dtype=dtype,
        )
        G[:, :, 35:47] = G_lambda
        inequality_b = b_lambda
        if has_servo:
            force_feedback = G_lambda @ wrench_response
            G[:, :, 6:35] = force_feedback * servo_acc_gain[:, None, :]
            inequality_b = b_lambda - (force_feedback @ servo_bias.unsqueeze(-1)).squeeze(-1)
        effort_limit = None
        if effort_limits is not None:
            effort_limit = _batched_vector(
                effort_limits,
                batch_size=batch_size,
                dimension=29,
                device=device,
                dtype=dtype,
                default=0.0,
                name="effort_limits",
            )
            if not bool(torch.isfinite(effort_limit).all()):
                raise ValueError("effort_limits must be finite.")
            if bool((effort_limit <= 0.0).any()):
                raise ValueError("effort_limits must be positive.")

        # -------------------------------------------------------------
        # Box bounds:
        #
        #   acc_lower <= acc <= acc_upper
        #   lambda_inactive = 0
        #   -effort_limits <= tau_qp + tau_known <= effort_limits
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
        if effort_limit is not None and not has_servo:
            lower[:, 47:76] = -effort_limit - known_torque
            upper[:, 47:76] = effort_limit - known_torque

        if effort_limit is not None and has_servo:
            total_effort_G = torch.zeros(batch_size, 29, self.NUM_VARIABLES, device=device, dtype=dtype)
            total_effort_G[:, :, 6:35] = torch.diag_embed(servo_acc_gain)
            total_effort_G[:, :, 47:76] = torch.eye(29, device=device, dtype=dtype)
            G = torch.cat((G, total_effort_G, -total_effort_G), dim=1)
            inequality_b = torch.cat((inequality_b, effort_limit - servo_bias, effort_limit + servo_bias), dim=1)

        prediction_limit = None
        if (predicted_acc_lower is None) != (predicted_acc_upper is None):
            raise ValueError("Supply both predicted acceleration bounds.")
        if predicted_acc_limit is not None or predicted_acc_lower is not None:
            prediction_lower = _batched_vector(predicted_acc_lower, batch_size=batch_size,
                dimension=35, device=device, dtype=dtype, default=-1e6, name="predicted_acc_lower")
            prediction_upper = _batched_vector(predicted_acc_upper, batch_size=batch_size,
                dimension=35, device=device, dtype=dtype, default=1e6, name="predicted_acc_upper")
            if predicted_acc_limit is not None:
                prediction_limit = _batched_vector(predicted_acc_limit, batch_size=batch_size,
                    dimension=35, device=device, dtype=dtype, default=0., name="predicted_acc_limit")
                if not torch.isfinite(prediction_limit).all() or (prediction_limit <= 0).any():
                    raise ValueError("predicted_acc_limit must be finite and positive.")
                prediction_lower = torch.maximum(prediction_lower, -prediction_limit)
                prediction_upper = torch.minimum(prediction_upper, prediction_limit)
            if (torch.isnan(prediction_lower).any() or torch.isnan(prediction_upper).any()
                    or torch.isposinf(prediction_lower).any() or torch.isneginf(prediction_upper).any()
                    or (prediction_lower > prediction_upper).any()):
                raise ValueError("Predicted acceleration bounds must be ordered, without NaN or inward infinities.")
            prediction_G = torch.zeros(batch_size, 35, self.NUM_VARIABLES, device=device, dtype=dtype)
            prediction_G[:, :, :35] = torch.eye(35, device=device, dtype=dtype)
            prediction_bias = torch.zeros(batch_size, 35, device=device, dtype=dtype)
            if has_servo:
                prediction_G[:, :, 6:35] += acceleration_response * servo_acc_gain[:, None, :]
                prediction_bias = (acceleration_response @ servo_bias.unsqueeze(-1)).squeeze(-1)
            prediction_bias = prediction_bias - correction
            # Joint-only limits leave floating-base rows unbounded. Keep a
            # common batch shape, but encode absent bounds as finite 0 <= 0
            # rows instead of passing infinities to the QP backend.
            upper_finite = torch.isfinite(prediction_upper)
            lower_finite = torch.isfinite(prediction_lower)
            upper_G = torch.where(upper_finite[..., None], prediction_G, 0.)
            lower_G = torch.where(lower_finite[..., None], -prediction_G, 0.)
            upper_b = torch.where(upper_finite, prediction_upper - prediction_bias, 0.)
            lower_b = torch.where(lower_finite, -prediction_lower + prediction_bias, 0.)
            G = torch.cat((G, upper_G, lower_G), dim=1)
            inequality_b = torch.cat((inequality_b, upper_b, lower_b), dim=1)

        if acceleration_constraints is not None:
            constraint_C, constraint_d = acceleration_constraints
            if (constraint_C.ndim != 3 or constraint_C.shape[0] != batch_size
                    or constraint_C.shape[2] != 35
                    or constraint_d.shape != constraint_C.shape[:2]
                    or any(x.device != device or x.dtype != dtype or not torch.isfinite(x).all()
                           for x in (constraint_C, constraint_d))):
                raise ValueError('Physical acceleration constraints require finite C [B,K,35], d [B,K].')
            physical_map = torch.eye(35, device=device, dtype=dtype).expand(batch_size,-1,-1).clone()
            physical_offset = -correction
            if has_servo:
                physical_map[:, :, 6:] += acceleration_response*servo_acc_gain[:, None, :]
                physical_offset = physical_offset+(acceleration_response@servo_bias[...,None]).squeeze(-1)
            extra_G = torch.zeros(batch_size,constraint_C.shape[1],self.NUM_VARIABLES,device=device,dtype=dtype)
            extra_G[:, :, :35] = constraint_C@physical_map
            G = torch.cat((G,extra_G),dim=1)
            inequality_b = torch.cat((inequality_b,
                constraint_d-(constraint_C@physical_offset[...,None]).squeeze(-1)),dim=1)

        qp = QPData(
            H=H,
            h=linear_term,
            G=G,
            b=inequality_b,
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

        priority_G = priority_b = None
        priority_solution = None
        priority_stages = []
        self.last_priority_problems = []
        self.last_priority_problem = None
        self.last_priority_solver_status = []
        if priority_levels is not None:
            names = tuple(name for level in priority_levels for name in level)
            if (not priority_levels or any(not level for level in priority_levels)
                    or len(names) != len(set(names)) or set(names) != set(priority_tasks)):
                raise ValueError("Priority levels must partition selected tasks without duplicates.")
        if priority_rows and not set(priority_rows).issubset(priority_tasks):
            raise ValueError("Priority row selectors must name selected priority tasks.")
        if priority_projections and not set(priority_projections).issubset(priority_tasks):
            raise ValueError("Priority projections must name selected priority tasks.")
        if set(priority_rows or {}) & set(priority_projections or {}):
            raise ValueError("Use either row selection or projection for each priority task.")
        if priority_requests is not None and not priority_tasks:
            raise ValueError('Separate priority requests require priority task names.')
        if priority_tasks:
            priority_pool = task_requests if priority_requests is None else priority_requests
            if (not priority_pool or not math.isfinite(priority_tolerance)
                    or priority_tolerance <= 0):
                raise ValueError("Priority tasks require task requests and a positive finite tolerance.")
            selected = [request for request in priority_pool if request.name in priority_tasks]
            if set(priority_tasks) != {request.name for request in selected}:
                raise ValueError("Unknown priority task name.")
            if priority_rows:
                filtered = []
                for request in selected:
                    indices = priority_rows.get(request.name)
                    if indices is None:
                        filtered.append(request)
                        continue
                    if (not indices or len(set(indices)) != len(indices)
                            or any(type(i) is not int or i < 0 or i >= request.A.shape[1]
                                   for i in indices)):
                        raise ValueError("Priority row indices must be nonempty, unique and within the task.")
                    active = torch.zeros_like(request.active)
                    active[:, list(indices)] = request.active[:, list(indices)]
                    filtered.append(TaskRequest(request.name, request.A, request.bias,
                        request.desired, request.metric, active, request.priority))
                selected = filtered
            if priority_projections:
                selected = [request.project(priority_projections[request.name])
                    if request.name in priority_projections else request for request in selected]
            levels = priority_levels or (priority_tasks,)
            locks, targets = [], []
            for stage_index, level in enumerate(levels):
                stage_tasks = [request for request in selected if request.name in level]
                first_H = self.hessian_regularization * full_identity.to(objective_dtype).expand(batch_size, -1, -1).clone()
                first_h = torch.zeros_like(linear_term)
                first_H[:, :35, :35], first_h[:, :35] = task_objective(
                    stage_tasks, transform, offset, accumulation_dtype=objective_dtype)
                first_H[:, :35, :35] += self.hessian_regularization * full_identity[:, :35, :35]
                if getattr(self, 'priority_wrench_reference', False) and contact_wrench_reference is not None:
                    first_H += self.wrench_weight * (wrench_map.transpose(1, 2) @ weighted_map)
                    first_h += self.wrench_weight * (weighted_map.transpose(1, 2) @
                        (wrench_offset-contact_wrench_reference).unsqueeze(-1)).squeeze(-1)
                # qp already contains every earlier stage's output locks.
                first_qp = QPData(first_H, first_h, qp.G, qp.b, lower, upper, qp.A_eq, qp.b_eq)
                workspace = 'priority' if stage_index == 0 else f'priority_{stage_index}'
                priority_solution = self.solver.solve(first_qp, warm_z=warm_z, warm_lam_eq=warm_lam_eq,
                    **({'workspace': workspace} if self.solver_backend in ('osqp','daqp') else {}))
                priority_solution.z = priority_solution.z.to(dtype)
                priority_stages.append((first_qp, priority_solution))
                self.last_priority_problems.append(first_qp)
                statuses = list(getattr(self.solver,'last_status',[]))
                if not self.last_priority_solver_status:
                    self.last_priority_solver_status = statuses
                else:
                    self.last_priority_solver_status = [f'{before} | {after}' for before, after in
                        zip(self.last_priority_solver_status, statuses)]
                self.last_priority_problem = first_qp
                rows = torch.cat([request.A * (request.active &
                    (request.metric.diagonal(dim1=-2, dim2=-1) > 0))[..., None]
                    for request in stage_tasks], dim=1)
                lock = torch.zeros(batch_size, rows.shape[1], self.NUM_VARIABLES, device=device, dtype=dtype)
                lock[:, :, :35] = rows @ transform
                achieved = (lock @ priority_solution.z[..., None]).squeeze(-1)
                locks.append(lock); targets.append(achieved)
                all_locks, all_targets = torch.cat(locks, dim=1), torch.cat(targets, dim=1)
                priority_G = torch.cat((all_locks, -all_locks), dim=1)
                priority_b = torch.cat((all_targets + priority_tolerance, -all_targets + priority_tolerance), dim=1)
                qp = QPData(H, linear_term, G, inequality_b, lower, upper,
                    torch.cat((A_eq, all_locks), dim=1), torch.cat((b_eq, all_targets), dim=1))
                if self.solver_backend in ('osqp','daqp'):
                    qp = QPData(H, linear_term, torch.cat((G, priority_G), dim=1),
                                torch.cat((inequality_b, priority_b), dim=1),
                                lower, upper, A_eq, b_eq)
                warm_z, warm_lam_eq = priority_solution.z, None

        stop_indices = None
        self.last_joint_stop_reaction = torch.zeros(batch_size, 29, device=device, dtype=dtype)
        self.last_joint_stop_constraint_error = torch.zeros(batch_size, device=device, dtype=dtype)
        if joint_stop_directions is not None:
            if has_servo or correction.ne(0).any() or priority_stages or effort_limit is None:
                raise ValueError('Observed joint stops require direct physical acceleration, one QP and finite effort limits')
            from fabrics_sim.solver.joint_stop import augment_joint_stops
            qp, stop_indices = augment_joint_stops(qp, joint_stop_directions, effort_limit)
            if len(stop_indices):
                # A new passive-constraint mode has different variables and
                # equalities. Select one cold solve; do not retry the QP.
                warm_z = warm_lam_eq = None
        solution = self.solver.solve(
            qp,
            warm_z=warm_z,
            warm_lam_eq=warm_lam_eq,
            **({'workspace': 'tracking'} if self.solver_backend in ('osqp','daqp') else {}),
        )
        self.last_full_solution = solution.z.clone()
        stop_bound_error = torch.zeros(batch_size, device=device, dtype=dtype)
        if stop_indices is not None and len(stop_indices):
            reaction = solution.z[:, 76:]
            signs = joint_stop_directions[:, stop_indices].to(reaction)
            self.last_joint_stop_reaction[:, stop_indices] = (-signs*reaction).to(dtype)
            stop_bound_error = torch.maximum((-reaction).clamp_min(0),
                (reaction-effort_limit[:, stop_indices].to(reaction)*signs.ne(0)).clamp_min(0)).amax(1).to(dtype)
            solution.z = solution.z[:, :76]
        solution.z = solution.z.to(dtype)
        if solution.lam_eq is not None:
            solution.lam_eq = solution.lam_eq.to(dtype)
        # Preserve the exact last problem for independent failure replay.
        self.last_problem = qp

        final_acc = solution.z[:, 0:35]
        final_wrench = solution.z[:, 35:47]
        final_joint_torque = solution.z[:, 47:76]
        total_joint_torque = final_joint_torque + known_torque
        nominal_wrench = final_wrench
        nominal_acc = final_acc - correction
        predicted_acc = nominal_acc
        if has_servo:
            pd_torque = servo_bias + servo_acc_gain * final_acc[:, 6:]
            total_joint_torque = final_joint_torque + pd_torque
            predicted_acc = nominal_acc + (acceleration_response @ pd_torque.unsqueeze(-1)).squeeze(-1)
            final_wrench = nominal_wrench + (wrench_response @ pd_torque.unsqueeze(-1)).squeeze(-1)

        # Explicit residual checks are important because the referenced
        # admm_qp.py reports primal feasibility, not a full KKT certificate.
        dynamics_residual = (
            torch.bmm(mass_matrix, predicted_acc.unsqueeze(-1)).squeeze(-1)
            + bias_force
            - torch.bmm(
                Jc.transpose(1, 2),
                final_wrench.unsqueeze(-1),
            ).squeeze(-1)
        )
        dynamics_residual[:, 6:35] -= total_joint_torque
        dynamics_residual[:, 6:35] -= self.last_joint_stop_reaction
        contact_residual = (
            torch.bmm(Jc, predicted_acc.unsqueeze(-1)).squeeze(-1)
            + Jdot_nu
        )
        friction_residual = (
            torch.bmm(
                G_lambda,
                final_wrench.unsqueeze(-1),
            ).squeeze(-1)
            - b_lambda
        )
        base_error = dynamics_residual[:, 0:6].abs().amax(dim=1)
        joint_dynamics_error = (
            dynamics_residual[:, 6:35].abs().amax(dim=1)
        )
        contact_error = contact_residual.abs().amax(dim=1)
        if friction_residual.shape[1] == 0:
            friction_violation = torch.zeros(
                batch_size, device=device, dtype=dtype
            )
        else:
            friction_violation = torch.clamp_min(
                friction_residual,
                0.0,
            ).amax(dim=1)
        if effort_limit is None:
            effort_violation = torch.zeros(
                batch_size, device=device, dtype=dtype
            )
        else:
            effort_violation = torch.clamp_min(
                total_joint_torque.abs() - effort_limit,
                0.0,
            ).amax(dim=1)

        tolerance = self.validation_tolerance
        box_violation = torch.maximum(
            torch.clamp_min(lower - solution.z, 0.0).amax(dim=1),
            torch.clamp_min(solution.z - upper, 0.0).amax(dim=1),
        )
        box_violation = torch.maximum(box_violation, stop_bound_error)
        if predicted_acc_limit is not None or predicted_acc_lower is not None:
            box_violation = torch.maximum(box_violation,
                torch.maximum(prediction_lower - predicted_acc,
                              predicted_acc - prediction_upper).clamp_min(0).amax(dim=1))
        validated = (
            torch.isfinite(solution.z).all(dim=1)
            & (base_error <= tolerance)
            & (joint_dynamics_error <= tolerance)
            & (contact_error <= tolerance)
            & (
                friction_violation
                <= self.friction_validation_tolerance
            )
            & (
                effort_violation
                <= self.effort_validation_tolerance
            )
            & (box_violation <= self.box_validation_tolerance)
        )
        self.last_acceleration_constraint_error = torch.zeros(batch_size,device=device,dtype=dtype)
        if joint_stop_directions is not None:
            self.last_joint_stop_constraint_error = (
                predicted_acc[:, 6:]*joint_stop_directions.ne(0)).abs().amax(1)
            validated &= self.last_joint_stop_constraint_error <= tolerance
        if acceleration_constraints is not None and constraint_C.shape[1]:
            self.last_acceleration_constraint_error = (
                (constraint_C@predicted_acc[...,None]).squeeze(-1)-constraint_d).clamp_min(0).amax(-1)
            validated &= self.last_acceleration_constraint_error <= tolerance
        if priority_solution is not None:
            first_error = torch.zeros(batch_size, device=device, dtype=dtype)
            for stage_problem, stage_solution in priority_stages:
                first_z = stage_solution.z
                stage_error = torch.stack((
                    ((stage_problem.A_eq @ first_z[..., None]).squeeze(-1)-stage_problem.b_eq).abs().amax(-1),
                    ((stage_problem.G @ first_z[..., None]).squeeze(-1)-stage_problem.b).clamp_min(0).amax(-1),
                    (lower-first_z).clamp_min(0).amax(-1),
                    (first_z-upper).clamp_min(0).amax(-1)), dim=-1).amax(-1)
                first_error = torch.maximum(first_error, stage_error)
                validated &= torch.isfinite(first_z).all(-1) & (stage_error <= tolerance)
                if stage_solution.status_ok is not None:
                    validated &= stage_solution.status_ok
            priority_error = ((priority_G @ solution.z[..., None]).squeeze(-1)-priority_b).clamp_min(0).amax(-1)
            validated &= priority_error <= .005
            self.last_priority_errors = (first_error, priority_error)
        else:
            self.last_priority_errors = None

        if solution.status_ok is not None:
            validated &= solution.status_ok

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
            joint_torque=final_joint_torque,
            total_joint_torque=total_joint_torque,
            predicted_acc=predicted_acc,
            nominal_acc=nominal_acc,
            nominal_contact_wrench=nominal_wrench,
            converged=solution.converged & torch.stack([stage.converged for _, stage in priority_stages]).all(0)
                if priority_stages else solution.converged,
            validated=validated,
            iters=solution.iters + sum(stage.iters for _, stage in priority_stages),
            objective=solution.obj,
            base_dynamics_error=base_error,
            joint_dynamics_error=joint_dynamics_error,
            contact_acceleration_error=contact_error,
            friction_violation=friction_violation,
            effort_violation=effort_violation,
            bound_violation=box_violation,
            task_residuals=None if task_requests is None else {
                request.name: request.residual(predicted_acc) for request in task_requests
            },
            joint_stop_reaction=self.last_joint_stop_reaction,
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
