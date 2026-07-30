"""
G1 FABRICS -> GPU dynamically consistent contact projection.

Input from FABRICS
------------------
    acc_fabric: [B, 35]

G1 generalized tangent-space convention
----------------------------------------
    nu = [base linear velocity(3),
          base angular velocity(3),
          29 joint velocities]                       -> 35

Output
------
    acc_dc: [B, 35]

Contact modes
-------------
    0: flight
    1: left support
    2: right support
    3: double support

The implementation always stacks the two 6D foot Jacobians into a fixed
[B, 12, 35] tensor. Inactive contact rows are masked out. This permits flight,
single support and double support to be processed in one GPU-batched solve.

Core equation
-------------
    a_dc = a_fab - M^{-1} J_c^T
           (J_c M^{-1} J_c^T)^{-1}
           (J_c a_fab + Jdot_c nu)

No explicit matrix inverse is calculated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor


NV = 35
FOOT_TASK_DIM = 6
MAX_CONTACT_DIM = 12

FLIGHT = 0
LEFT_SUPPORT = 1
RIGHT_SUPPORT = 2
DOUBLE_SUPPORT = 3


@dataclass(frozen=True)
class ContactDetectorConfig:
    """Contact-force hysteresis and debounce settings."""

    force_on: float = 20.0
    force_off: float = 10.0
    confirm_steps: int = 2


@dataclass(frozen=True)
class ProjectionConfig:
    """Projection and optional contact-stabilization settings."""

    regularization: float = 1.0e-6
    contact_kp: float = 0.0
    contact_kd: float = 0.0
    symmetrize_mass_matrix: bool = True
    filter_alpha_jdot_v: float = 0.0


@dataclass
class ProjectionOutput:
    """Results returned by G1FabricContactProjector.step()."""

    acc_dc: Tensor
    contact_mode: Tensor
    left_contact: Tensor
    right_contact: Tensor
    active_row_mask: Tensor
    contact_residual_before: Tensor
    contact_residual_after: Tensor


class BatchedFootContactDetector:
    """
    GPU-batched left/right foot contact detector.

    The detector applies:
      1. a force-on threshold,
      2. a lower force-off threshold,
      3. a consecutive-frame confirmation counter.
    """

    def __init__(
        self,
        num_envs: int,
        device: torch.device | str,
        config: ContactDetectorConfig = ContactDetectorConfig(),
    ) -> None:
        if config.force_on <= config.force_off:
            raise ValueError("force_on must be greater than force_off.")
        if config.confirm_steps < 1:
            raise ValueError("confirm_steps must be at least one.")

        self.num_envs = num_envs
        self.device = torch.device(device)
        self.config = config

        self.contact = torch.zeros(
            num_envs, 2, dtype=torch.bool, device=self.device
        )
        self.on_counter = torch.zeros(
            num_envs, 2, dtype=torch.int32, device=self.device
        )
        self.off_counter = torch.zeros(
            num_envs, 2, dtype=torch.int32, device=self.device
        )

    def reset(self, env_ids: Optional[Tensor] = None) -> None:
        """Reset all environments or a selected set of environment indices."""
        if env_ids is None:
            self.contact.zero_()
            self.on_counter.zero_()
            self.off_counter.zero_()
            return

        self.contact[env_ids] = False
        self.on_counter[env_ids] = 0
        self.off_counter[env_ids] = 0

    def update(self, normal_forces: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            normal_forces:
                [B, 2], left/right nonnegative normal contact forces.

        Returns:
            left_contact:  [B], bool
            right_contact: [B], bool
            contact_mode:  [B], int64
        """
        if normal_forces.shape != (self.num_envs, 2):
            raise ValueError(
                f"normal_forces must be [{self.num_envs}, 2], "
                f"got {tuple(normal_forces.shape)}."
            )
        if normal_forces.device != self.device:
            raise ValueError("normal_forces is on the wrong device.")

        want_on = normal_forces > self.config.force_on
        want_off = normal_forces < self.config.force_off

        self.on_counter = torch.where(
            want_on,
            self.on_counter + 1,
            torch.zeros_like(self.on_counter),
        )
        self.off_counter = torch.where(
            want_off,
            self.off_counter + 1,
            torch.zeros_like(self.off_counter),
        )

        confirmed_on = self.on_counter >= self.config.confirm_steps
        confirmed_off = self.off_counter >= self.config.confirm_steps

        self.contact = torch.where(
            confirmed_on,
            torch.ones_like(self.contact),
            torch.where(
                confirmed_off,
                torch.zeros_like(self.contact),
                self.contact,
            ),
        )

        left_contact = self.contact[:, 0]
        right_contact = self.contact[:, 1]

        # mode = left + 2 * right
        contact_mode = (
            left_contact.to(torch.int64)
            + 2 * right_contact.to(torch.int64)
        )
        return left_contact, right_contact, contact_mode


class BatchedJdotVEstimator:
    """
    Backward finite-difference estimator of Jdot @ nu.

    Prefer analytic foot bias acceleration from the kinematics engine. Use this
    estimator only when analytic Jdot @ nu is unavailable.
    """

    def __init__(
        self,
        num_envs: int,
        device: torch.device | str,
        dtype: torch.dtype,
        dt: float,
        filter_alpha: float = 0.0,
    ) -> None:
        if dt <= 0.0:
            raise ValueError("dt must be positive.")
        if not 0.0 <= filter_alpha < 1.0:
            raise ValueError("filter_alpha must satisfy 0 <= alpha < 1.")

        self.num_envs = num_envs
        self.device = torch.device(device)
        self.dtype = dtype
        self.dt = dt
        self.filter_alpha = filter_alpha

        self.previous_jacobian = torch.zeros(
            num_envs,
            MAX_CONTACT_DIM,
            NV,
            dtype=dtype,
            device=self.device,
        )
        self.filtered_jdot_v = torch.zeros(
            num_envs,
            MAX_CONTACT_DIM,
            dtype=dtype,
            device=self.device,
        )
        self.initialized = torch.zeros(
            num_envs, dtype=torch.bool, device=self.device
        )

    def reset(self, env_ids: Optional[Tensor] = None) -> None:
        if env_ids is None:
            self.previous_jacobian.zero_()
            self.filtered_jdot_v.zero_()
            self.initialized.zero_()
            return

        self.previous_jacobian[env_ids] = 0.0
        self.filtered_jdot_v[env_ids] = 0.0
        self.initialized[env_ids] = False

    def update(self, stacked_jacobian: Tensor, nu: Tensor) -> Tensor:
        """
        Args:
            stacked_jacobian: [B, 12, 35]
            nu:                [B, 35]

        Returns:
            jdot_v:            [B, 12]
        """
        jacobian_dot = (
            stacked_jacobian - self.previous_jacobian
        ) / self.dt
        raw = torch.bmm(jacobian_dot, nu.unsqueeze(-1)).squeeze(-1)

        # The first sample has no valid previous Jacobian.
        raw = torch.where(self.initialized.unsqueeze(-1), raw, 0.0)

        if self.filter_alpha > 0.0:
            filtered = (
                self.filter_alpha * self.filtered_jdot_v
                + (1.0 - self.filter_alpha) * raw
            )
        else:
            filtered = raw

        self.previous_jacobian.copy_(stacked_jacobian)
        self.filtered_jdot_v.copy_(filtered)
        self.initialized.fill_(True)
        return filtered


def stack_foot_quantities(
    left_foot_jacobian: Tensor,
    right_foot_jacobian: Tensor,
    left_jdot_v: Tensor,
    right_jdot_v: Tensor,
) -> tuple[Tensor, Tensor]:
    """Stack two 6D foot quantities into fixed 12D contact quantities."""
    stacked_jacobian = torch.cat(
        (left_foot_jacobian, right_foot_jacobian), dim=1
    )
    stacked_jdot_v = torch.cat((left_jdot_v, right_jdot_v), dim=1)
    return stacked_jacobian, stacked_jdot_v


def build_contact_row_mask(
    left_contact: Tensor,
    right_contact: Tensor,
    dtype: torch.dtype,
) -> Tensor:
    """
    Build [B, 12] mask:
        columns 0:6   correspond to the left foot,
        columns 6:12  correspond to the right foot.
    """
    left_rows = left_contact.to(dtype).unsqueeze(-1).expand(
        -1, FOOT_TASK_DIM
    )
    right_rows = right_contact.to(dtype).unsqueeze(-1).expand(
        -1, FOOT_TASK_DIM
    )
    return torch.cat((left_rows, right_rows), dim=1)


def dynamically_consistent_projection_masked(
    acc_fabric: Tensor,
    mass_matrix: Tensor,
    stacked_contact_jacobian: Tensor,
    stacked_jdot_v: Tensor,
    active_row_mask: Tensor,
    generalized_velocity: Optional[Tensor] = None,
    stacked_contact_pose_error: Optional[Tensor] = None,
    config: ProjectionConfig = ProjectionConfig(),
) -> tuple[Tensor, Tensor, Tensor]:
    """
    One GPU-batched dynamically consistent solve for all contact modes.

    Shapes:
        acc_fabric:                    [B, 35]
        mass_matrix:                   [B, 35, 35]
        stacked_contact_jacobian:      [B, 12, 35]
        stacked_jdot_v:                [B, 12]
        active_row_mask:               [B, 12]
        generalized_velocity:          [B, 35], needed when Kd != 0
        stacked_contact_pose_error:    [B, 12], needed when Kp != 0

    Returns:
        acc_dc:                        [B, 35]
        active_residual_before:        [B, 12]
        active_residual_after:         [B, 12]
    """
    batch_size = acc_fabric.shape[0]

    if acc_fabric.shape != (batch_size, NV):
        raise ValueError("acc_fabric must have shape [B, 35].")
    if mass_matrix.shape != (batch_size, NV, NV):
        raise ValueError("mass_matrix must have shape [B, 35, 35].")
    if stacked_contact_jacobian.shape != (
        batch_size, MAX_CONTACT_DIM, NV
    ):
        raise ValueError("stacked_contact_jacobian must be [B, 12, 35].")
    if stacked_jdot_v.shape != (batch_size, MAX_CONTACT_DIM):
        raise ValueError("stacked_jdot_v must have shape [B, 12].")
    if active_row_mask.shape != (batch_size, MAX_CONTACT_DIM):
        raise ValueError("active_row_mask must have shape [B, 12].")

    if config.contact_kd != 0.0 and generalized_velocity is None:
        raise ValueError("generalized_velocity is required when Kd != 0.")
    if config.contact_kp != 0.0 and stacked_contact_pose_error is None:
        raise ValueError("contact pose error is required when Kp != 0.")

    M = mass_matrix
    if config.symmetrize_mass_matrix:
        M = 0.5 * (M + M.transpose(-1, -2))

    # Masking a row removes the corresponding inactive contact constraint.
    row_mask_3d = active_row_mask.unsqueeze(-1)
    Jc = stacked_contact_jacobian * row_mask_3d
    jdot_v_c = stacked_jdot_v * active_row_mask

    # Step 1: X = M^{-1} Jc^T, solved without explicitly inverting M.
    minv_jt = torch.linalg.solve(M, Jc.transpose(-1, -2))

    # Step 2: A = Jc M^{-1} Jc^T.
    contact_matrix = torch.bmm(Jc, minv_jt)
    contact_matrix = 0.5 * (
        contact_matrix + contact_matrix.transpose(-1, -2)
    )

    # Inactive rows of Jc are zero, making A singular. Add 1 to the inactive
    # diagonal. Their residual is also zero, so their solution remains zero.
    inactive_diagonal = 1.0 - active_row_mask
    active_regularization = config.regularization * active_row_mask
    diagonal = inactive_diagonal + active_regularization
    contact_matrix = contact_matrix + torch.diag_embed(diagonal)

    # Step 3: r = Jc a_fab + Jdot_c nu.
    residual_before = (
        torch.bmm(Jc, acc_fabric.unsqueeze(-1)).squeeze(-1)
        + jdot_v_c
    )

    # Optional Baumgarte contact stabilization.
    if config.contact_kd != 0.0:
        contact_velocity = torch.bmm(
            Jc, generalized_velocity.unsqueeze(-1)
        ).squeeze(-1)
        residual_before = (
            residual_before + config.contact_kd * contact_velocity
        )

    if config.contact_kp != 0.0:
        residual_before = (
            residual_before
            + config.contact_kp
            * stacked_contact_pose_error
            * active_row_mask
        )

    # Step 4: y = A^{-1} r.
    contact_solution = torch.linalg.solve(
        contact_matrix, residual_before.unsqueeze(-1)
    )

    # Step 5: a_dc = a_fab - M^{-1} Jc^T y.
    correction = torch.bmm(minv_jt, contact_solution).squeeze(-1)
    acc_dc = acc_fabric - correction

    # Verify the raw physical contact acceleration constraint.
    residual_after = (
        torch.bmm(Jc, acc_dc.unsqueeze(-1)).squeeze(-1)
        + jdot_v_c
    ) * active_row_mask

    return acc_dc, residual_before, residual_after


class G1FabricContactProjector:
    """
    Complete stateful module placed immediately after FABRICS.

    Pipeline:
        1. detect left/right contact from normal forces,
        2. obtain or estimate Jdot @ nu,
        3. construct fixed 12D masked contact constraints,
        4. perform one batched GPU dynamic-consistency solve,
        5. return a 35D contact-consistent generalized acceleration.
    """

    def __init__(
        self,
        num_envs: int,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
        dt: float = 0.02,
        detector_config: ContactDetectorConfig = ContactDetectorConfig(),
        projection_config: ProjectionConfig = ProjectionConfig(),
    ) -> None:
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.dtype = dtype
        self.projection_config = projection_config

        self.contact_detector = BatchedFootContactDetector(
            num_envs=num_envs,
            device=self.device,
            config=detector_config,
        )
        self.jdot_v_estimator = BatchedJdotVEstimator(
            num_envs=num_envs,
            device=self.device,
            dtype=dtype,
            dt=dt,
            filter_alpha=projection_config.filter_alpha_jdot_v,
        )

    def reset(self, env_ids: Optional[Tensor] = None) -> None:
        self.contact_detector.reset(env_ids)
        self.jdot_v_estimator.reset(env_ids)

    def step(
        self,
        acc_fabric: Tensor,
        mass_matrix: Tensor,
        generalized_velocity: Tensor,
        left_foot_jacobian: Tensor,
        right_foot_jacobian: Tensor,
        foot_normal_forces: Tensor,
        left_jdot_v: Optional[Tensor] = None,
        right_jdot_v: Optional[Tensor] = None,
        left_contact_pose_error: Optional[Tensor] = None,
        right_contact_pose_error: Optional[Tensor] = None,
    ) -> ProjectionOutput:
        """
        Run one controller step.

        Args:
            acc_fabric:              [B, 35]
            mass_matrix:             [B, 35, 35]
            generalized_velocity:    [B, 35]
            left_foot_jacobian:      [B, 6, 35]
            right_foot_jacobian:     [B, 6, 35]
            foot_normal_forces:      [B, 2], left then right
            left_jdot_v:             [B, 6], optional analytic value
            right_jdot_v:            [B, 6], optional analytic value
            left_contact_pose_error: [B, 6], optional
            right_contact_pose_error:[B, 6], optional
        """
        previous_contact = self.contact_detector.contact.clone()
        left_contact, right_contact, contact_mode = (
            self.contact_detector.update(foot_normal_forces)
        )

        # Stack left/right Jacobians.
        stacked_jacobian = torch.cat(
            (left_foot_jacobian, right_foot_jacobian), dim=1
        )

        # Prefer analytic Jdot @ nu. Otherwise estimate it on GPU.
        analytic_left_given = left_jdot_v is not None
        analytic_right_given = right_jdot_v is not None
        if analytic_left_given != analytic_right_given:
            raise ValueError(
                "Provide both left_jdot_v and right_jdot_v, or neither."
            )

        if analytic_left_given:
            stacked_jdot_v = torch.cat(
                (left_jdot_v, right_jdot_v), dim=1
            )
            # Keep finite-difference estimator history synchronized in case
            # the caller later switches to the fallback estimator.
            self.jdot_v_estimator.previous_jacobian.copy_(stacked_jacobian)
            self.jdot_v_estimator.initialized.fill_(True)
        else:
            stacked_jdot_v = self.jdot_v_estimator.update(
                stacked_jacobian, generalized_velocity
            )
            # A finite difference spanning the instant a contact becomes
            # active is not a reliable contact bias acceleration. Start the
            # newly active foot at zero and use normal differences afterward.
            rising_contact = self.contact_detector.contact & ~previous_contact
            rising_rows = torch.cat(
                (
                    rising_contact[:, 0:1].expand(-1, FOOT_TASK_DIM),
                    rising_contact[:, 1:2].expand(-1, FOOT_TASK_DIM),
                ),
                dim=1,
            )
            stacked_jdot_v = torch.where(
                rising_rows,
                torch.zeros_like(stacked_jdot_v),
                stacked_jdot_v,
            )

        active_row_mask = build_contact_row_mask(
            left_contact, right_contact, acc_fabric.dtype
        )

        if (
            left_contact_pose_error is not None
            and right_contact_pose_error is not None
        ):
            stacked_pose_error = torch.cat(
                (
                    left_contact_pose_error,
                    right_contact_pose_error,
                ),
                dim=1,
            )
        elif (
            left_contact_pose_error is None
            and right_contact_pose_error is None
        ):
            stacked_pose_error = None
        else:
            raise ValueError(
                "Provide both left/right contact pose errors, or neither."
            )

        acc_dc, residual_before, residual_after = (
            dynamically_consistent_projection_masked(
                acc_fabric=acc_fabric,
                mass_matrix=mass_matrix,
                stacked_contact_jacobian=stacked_jacobian,
                stacked_jdot_v=stacked_jdot_v,
                active_row_mask=active_row_mask,
                generalized_velocity=generalized_velocity,
                stacked_contact_pose_error=stacked_pose_error,
                config=self.projection_config,
            )
        )

        return ProjectionOutput(
            acc_dc=acc_dc,
            contact_mode=contact_mode,
            left_contact=left_contact,
            right_contact=right_contact,
            active_row_mask=active_row_mask,
            contact_residual_before=residual_before,
            contact_residual_after=residual_after,
        )


def _make_spd_mass_matrix(
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Generate synthetic positive-definite mass matrices for the self-test."""
    random_matrix = torch.randn(
        batch_size, NV, NV, device=device, dtype=dtype
    )
    identity = torch.eye(
        NV, device=device, dtype=dtype
    ).expand(batch_size, NV, NV)
    return (
        torch.bmm(random_matrix, random_matrix.transpose(-1, -2))
        + 2.0 * identity
    )


def self_test() -> None:
    """
    Synthetic numerical test.

    It deliberately creates the four modes in different environments and
    checks that active contact acceleration residuals are approximately zero.
    """
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64
    batch_size = 8

    projector = G1FabricContactProjector(
        num_envs=batch_size,
        device=device,
        dtype=dtype,
        dt=0.02,
        detector_config=ContactDetectorConfig(
            force_on=20.0,
            force_off=10.0,
            confirm_steps=1,
        ),
        projection_config=ProjectionConfig(
            regularization=1.0e-10,
            contact_kp=0.0,
            contact_kd=0.0,
        ),
    )

    acc_fabric = torch.randn(
        batch_size, NV, device=device, dtype=dtype
    )
    mass_matrix = _make_spd_mass_matrix(
        batch_size, device, dtype
    )
    generalized_velocity = torch.randn(
        batch_size, NV, device=device, dtype=dtype
    )
    left_J = torch.randn(
        batch_size, FOOT_TASK_DIM, NV, device=device, dtype=dtype
    )
    right_J = torch.randn(
        batch_size, FOOT_TASK_DIM, NV, device=device, dtype=dtype
    )
    left_jdot_v = torch.randn(
        batch_size, FOOT_TASK_DIM, device=device, dtype=dtype
    )
    right_jdot_v = torch.randn(
        batch_size, FOOT_TASK_DIM, device=device, dtype=dtype
    )

    # Modes repeat as:
    # env 0: flight, env 1: left, env 2: right, env 3: double.
    expected_mode = torch.arange(
        batch_size, device=device, dtype=torch.int64
    ) % 4
    normal_forces = torch.zeros(
        batch_size, 2, device=device, dtype=dtype
    )
    normal_forces[:, 0] = torch.where(
        (expected_mode == LEFT_SUPPORT)
        | (expected_mode == DOUBLE_SUPPORT),
        100.0,
        0.0,
    )
    normal_forces[:, 1] = torch.where(
        (expected_mode == RIGHT_SUPPORT)
        | (expected_mode == DOUBLE_SUPPORT),
        100.0,
        0.0,
    )

    output = projector.step(
        acc_fabric=acc_fabric,
        mass_matrix=mass_matrix,
        generalized_velocity=generalized_velocity,
        left_foot_jacobian=left_J,
        right_foot_jacobian=right_J,
        foot_normal_forces=normal_forces,
        left_jdot_v=left_jdot_v,
        right_jdot_v=right_jdot_v,
    )

    if not torch.equal(output.contact_mode, expected_mode):
        raise RuntimeError("Contact-mode classification failed.")

    active_error = (
        output.contact_residual_after
        * output.active_row_mask
    )
    max_active_error = active_error.abs().max().item()

    flight = output.contact_mode == FLIGHT
    max_flight_change = (
        output.acc_dc[flight] - acc_fabric[flight]
    ).abs().max().item()

    print(f"device: {device}")
    print(f"acc_dc shape: {tuple(output.acc_dc.shape)}")
    print(f"contact modes: {output.contact_mode.tolist()}")
    print(f"max active contact residual: {max_active_error:.3e}")
    print(f"max flight acceleration change: {max_flight_change:.3e}")

    if max_active_error > 1.0e-7:
        raise RuntimeError("Dynamic-consistency residual is too large.")
    if max_flight_change != 0.0:
        raise RuntimeError("Flight acceleration must remain unchanged.")

    print("self-test passed")


if __name__ == "__main__":
    self_test()
