from __future__ import annotations

import torch

from fabrics_sim.prod.kinematics import Kinematics
from fabrics_sim.taskmaps.maps_base import BaseMap
from fabrics_sim.taskmaps.robot_frame_origins_taskmap import RobotKinematics


def _normalize(vector: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    return vector / torch.linalg.vector_norm(
        vector, dim=-1, keepdim=True
    ).clamp_min(eps)


def _skew(vector: torch.Tensor) -> torch.Tensor:
    """Return the skew-symmetric matrix [v]_x for (..., 3) vectors."""
    x, y, z = vector.unbind(dim=-1)
    zero = torch.zeros_like(x)
    return torch.stack(
        (
            zero, -z, y,
            z, zero, -x,
            -y, x, zero,
        ),
        dim=-1,
    ).reshape(*vector.shape[:-1], 3, 3)


def _quat_xyzw_to_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    quaternion = _normalize(quaternion)
    x, y, z, w = quaternion.unbind(dim=-1)

    return torch.stack(
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(*quaternion.shape[:-1], 3, 3)


def _so3_log(rotation: torch.Tensor, eps: float = 1.0e-7) -> torch.Tensor:
    """Compute Log(R)^vee, including a stable small-angle branch."""
    trace = (
        rotation[..., 0, 0]
        + rotation[..., 1, 1]
        + rotation[..., 2, 2]
    )
    cos_theta = ((trace - 1.0) * 0.5).clamp(-1.0 + eps, 1.0 - eps)
    theta = torch.acos(cos_theta)

    vee = torch.stack(
        (
            rotation[..., 2, 1] - rotation[..., 1, 2],
            rotation[..., 0, 2] - rotation[..., 2, 0],
            rotation[..., 1, 0] - rotation[..., 0, 1],
        ),
        dim=-1,
    )

    regular_scale = theta / (
        2.0 * torch.sin(theta).abs().clamp_min(eps)
    )
    small_scale = 0.5 + theta.square() / 12.0
    scale = torch.where(theta.abs() < 1.0e-4, small_scale, regular_scale)
    return scale.unsqueeze(-1) * vee


def _so3_left_jacobian_inverse(
    rotation_vector: torch.Tensor,
    eps: float = 1.0e-7,
) -> torch.Tensor:
    """Inverse SO(3) left Jacobian for world-frame rotation errors."""
    theta = torch.linalg.vector_norm(
        rotation_vector, dim=-1, keepdim=True
    )
    phi_hat = _skew(rotation_vector)
    phi_hat_squared = torch.matmul(phi_hat, phi_hat)

    identity = torch.eye(
        3,
        dtype=rotation_vector.dtype,
        device=rotation_vector.device,
    ).expand(*rotation_vector.shape[:-1], 3, 3)

    theta_scalar = theta.squeeze(-1)
    sin_theta = torch.sin(theta_scalar)
    cos_theta = torch.cos(theta_scalar)

    coefficient_regular = (
        1.0 / theta_scalar.square().clamp_min(eps)
        - (1.0 + cos_theta)
        / (
            2.0
            * theta_scalar
            * sin_theta
        ).abs().clamp_min(eps)
    )
    coefficient_small = (
        torch.full_like(theta_scalar, 1.0 / 12.0)
        + theta_scalar.square() / 720.0
    )
    coefficient = torch.where(
        theta_scalar.abs() < 1.0e-4,
        coefficient_small,
        coefficient_regular,
    )

    return (
        identity
        - 0.5 * phi_hat
        + coefficient[..., None, None] * phi_hat_squared
    )


class G1EndEffectorPoseTaskMap(BaseMap):
    """SE(3) task map for a selectable group of G1 end effectors."""

    END_EFFECTOR_ORDER = (
        "left_foot",
        "right_foot",
        "left_hand",
        "right_hand",
    )

    DEFAULT_FRAME_GROUPS = {
        "left_foot": ("left_ankle_roll_link",),
        "right_foot": ("right_ankle_roll_link",),
        "left_hand": ("left_wrist_yaw_link",),
        "right_hand": ("right_wrist_yaw_link",),
    }

    def __init__(
        self,
        urdf_path: str,
        batch_size: int,
        device: str,
        frame_groups: dict[str, tuple[str, str, str, str]] | None = None,
        end_effector_order: tuple[str, ...] | None = None,
    ):
        super().__init__(device)
        self.batch_size = batch_size
        self.frame_groups = (
            self.DEFAULT_FRAME_GROUPS if frame_groups is None else frame_groups
        )
        self.end_effector_order = (
            self.END_EFFECTOR_ORDER
            if end_effector_order is None
            else tuple(end_effector_order)
        )
        if not self.end_effector_order:
            raise ValueError("end_effector_order must not be empty.")

        for name in self.end_effector_order:
            if name not in self.frame_groups:
                raise KeyError(f"Missing frame group for {name}.")
            if len(self.frame_groups[name]) < 1:
                raise ValueError(
                    f"{name} needs at least one origin frame."
                )

        self.robot_kinematics = Kinematics(
            urdf_path,
            batch_size,
            False,
            device=device,
            floating_base=True,
        )

        link_names = [
            self.frame_groups[name][0] for name in self.end_effector_order
        ]

        self.link_indices = torch.tensor(
            [
                self.robot_kinematics.get_link_index(link_name)
                for link_name in link_names
            ],
            dtype=torch.long,
            device=device,
        )

        # Mask the world-frame angular axes that affect each end effector.
        angular_mask = torch.zeros(
            len(self.end_effector_order),
            self.robot_kinematics.velocity_dim,
            dtype=torch.float32,
            device=device,
        )
        angular_mask[:, 3:6] = 1.0
        cspace_links = list(self.robot_kinematics.urdf_info.cspace2link)
        link_paths = self.robot_kinematics.link_paths_numpy
        for end_effector_index, link_index in enumerate(
            self.link_indices.tolist()
        ):
            ancestors = set(link_paths[link_index])
            for joint_index, joint_link_index in enumerate(cspace_links):
                if joint_link_index in ancestors:
                    angular_mask[end_effector_index, joint_index + 6] = 1.0
        self.angular_mask = angular_mask

        self.position_target = torch.zeros(
            batch_size, len(self.end_effector_order), 3, device=device
        )
        self.quaternion_target = torch.zeros(
            batch_size, len(self.end_effector_order), 4, device=device
        )
        self.quaternion_target[..., 3] = 1.0

    def set_target(
        self,
        position_target: torch.Tensor,
        quaternion_target_xyzw: torch.Tensor,
    ) -> None:
        """Set retargeted end-effector targets.

        position_target:          (batch_size, num_end_effectors, 3)
        quaternion_target_xyzw:   (batch_size, num_end_effectors, 4)
        """
        count = len(self.end_effector_order)
        expected_position = (self.batch_size, count, 3)
        expected_quaternion = (self.batch_size, count, 4)
        if tuple(position_target.shape) != expected_position:
            raise ValueError(
                f"position_target must be {expected_position}, "
                f"got {tuple(position_target.shape)}."
            )
        if tuple(quaternion_target_xyzw.shape) != expected_quaternion:
            raise ValueError(
                f"quaternion_target_xyzw must be {expected_quaternion}, "
                f"got {tuple(quaternion_target_xyzw.shape)}."
            )

        quaternion_target_xyzw = _normalize(quaternion_target_xyzw)

        # Keep the quaternion sequence on one hemisphere.
        same_hemisphere = (
            torch.sum(
                quaternion_target_xyzw * self.quaternion_target,
                dim=-1,
                keepdim=True,
            )
            >= 0.0
        )
        quaternion_target_xyzw = torch.where(
            same_hemisphere,
            quaternion_target_xyzw,
            -quaternion_target_xyzw,
        )

        self.position_target.copy_(position_target)
        self.quaternion_target.copy_(quaternion_target_xyzw)

    @staticmethod
    def _recover_rotation(
        origin: torch.Tensor,
        point_x: torch.Tensor,
        point_y: torch.Tensor,
        point_z: torch.Tensor,
    ) -> torch.Tensor:
        """Recover a right-handed rotation matrix from auxiliary points."""
        x_axis = _normalize(point_x - origin)

        y_raw = point_y - origin
        y_axis = _normalize(
            y_raw
            - torch.sum(y_raw * x_axis, dim=-1, keepdim=True) * x_axis
        )

        z_axis = _normalize(torch.linalg.cross(x_axis, y_axis, dim=-1))

        # Use the z point to resolve a possible sign inconsistency.
        z_hint = point_z - origin
        flip = (
            torch.sum(z_axis * z_hint, dim=-1, keepdim=True) < 0.0
        )
        y_axis = torch.where(flip, -y_axis, y_axis)
        z_axis = torch.where(flip, -z_axis, z_axis)

        return torch.stack((x_axis, y_axis, z_axis), dim=-1)

    @staticmethod
    def _recover_angular_jacobian(
        origin: torch.Tensor,
        point_x: torch.Tensor,
        point_y: torch.Tensor,
        point_z: torch.Tensor,
        jac_origin: torch.Tensor,
        jac_x: torch.Tensor,
        jac_y: torch.Tensor,
        jac_z: torch.Tensor,
    ) -> torch.Tensor:
        """Recover J_omega from point Jacobians.

        For a fixed point r on a rigid link:
            J_point - J_origin = -[r]_x J_omega.
        """
        offsets = (
            point_x - origin,
            point_y - origin,
            point_z - origin,
        )
        jacobian_differences = (
            jac_x - jac_origin,
            jac_y - jac_origin,
            jac_z - jac_origin,
        )

        coefficient = torch.cat(
            [-_skew(offset) for offset in offsets],
            dim=-2,
        )
        right_hand_side = torch.cat(jacobian_differences, dim=-2)

        return torch.matmul(
            torch.linalg.pinv(coefficient),
            right_hand_side,
        )

    def forward_position(
        self,
        q: torch.Tensor,
        features=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        link_transforms, link_jacobians = RobotKinematics.apply(
            q,
            self.robot_kinematics,
        )

        selected_transforms = link_transforms[:, self.link_indices]
        positions = selected_transforms[..., :3]
        current_rotations = _quat_xyzw_to_matrix(
            selected_transforms[..., 3:7]
        )

        # Original Warp layout:
        #   (batch, links, generalized_velocity_dim, xyz)
        # Desired layout:
        #   (batch, links, xyz, generalized_velocity_dim)
        position_jacobians = link_jacobians[
            :, self.link_indices, :, :
        ].transpose(2, 3)
        joint_axes = self.robot_kinematics.batch_joint_axes_torch
        angular_jacobians = (
            joint_axes[:, None, :, :]
            * self.angular_mask[None, :, :, None]
        ).transpose(2, 3)

        target_rotations = _quat_xyzw_to_matrix(self.quaternion_target)

        task_errors = []
        task_jacobians = []

        for end_effector_index in range(len(self.end_effector_order)):
            origin = positions[:, end_effector_index]
            current_rotation = current_rotations[:, end_effector_index]
            jac_origin = position_jacobians[:, end_effector_index]
            angular_jacobian = angular_jacobians[:, end_effector_index]

            position_error = (
                origin - self.position_target[:, end_effector_index]
            )

            # World-frame error: R_error = R_current R_target^T.
            rotation_error = torch.matmul(
                current_rotation,
                target_rotations[:, end_effector_index].transpose(1, 2),
            )
            orientation_error = _so3_log(rotation_error)

            # e_dot = J_l(e)^(-1) * omega_error.
            orientation_error_jacobian = torch.matmul(
                _so3_left_jacobian_inverse(orientation_error),
                angular_jacobian,
            )

            task_errors.append(
                torch.cat((position_error, orientation_error), dim=-1)
            )
            task_jacobians.append(
                torch.cat(
                    (jac_origin, orientation_error_jacobian),
                    dim=-2,
                )
            )

        task_error = torch.cat(task_errors, dim=-1)
        task_jacobian = torch.cat(task_jacobians, dim=-2)

        return task_error, task_jacobian
