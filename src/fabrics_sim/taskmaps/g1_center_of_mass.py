from __future__ import annotations

import torch

from fabrics_sim.prod.kinematics import Kinematics
from fabrics_sim.taskmaps.g1_end_effector_pose_taskmap import (
    _quat_xyzw_to_matrix,
    _skew,
)
from fabrics_sim.taskmaps.maps_base import BaseMap
from fabrics_sim.taskmaps.robot_frame_origins_taskmap import RobotKinematics


class G1CenterOfMassTaskMap(BaseMap):
    """Mass-weighted whole-body center-of-mass task map for floating-base G1."""

    def __init__(self, urdf_path: str, batch_size: int, device: str):
        super().__init__(device)
        self.batch_size = batch_size
        self.robot_kinematics = Kinematics(
            urdf_path,
            batch_size,
            False,
            device=device,
            floating_base=True,
        )

        model = self.robot_kinematics.model
        self.link_masses = torch.as_tensor(
            model.body_mass.numpy(), dtype=torch.float32, device=device
        )
        self.link_com_offsets = torch.as_tensor(
            model.body_com.numpy(), dtype=torch.float32, device=device
        )
        self.total_mass = self.link_masses.sum()
        if not bool(torch.isfinite(self.total_mass)) or self.total_mass <= 0.0:
            raise ValueError("The G1 URDF must contain positive finite link masses.")

        link_count = len(self.robot_kinematics.link_names)
        if self.link_masses.shape != (link_count,):
            raise RuntimeError("Warp body masses do not align with URDF links.")
        if self.link_com_offsets.shape != (link_count, 3):
            raise RuntimeError("Warp body COM offsets do not align with URDF links.")

        # Each link angular velocity consists of floating-base angular velocity
        # plus the axes of the revolute joints on its ancestor chain.
        angular_mask = torch.zeros(
            link_count,
            self.robot_kinematics.velocity_dim,
            dtype=torch.float32,
            device=device,
        )
        angular_mask[:, 3:6] = 1.0
        cspace_links = list(self.robot_kinematics.urdf_info.cspace2link)
        for link_index, link_path in enumerate(
            self.robot_kinematics.link_paths_numpy
        ):
            ancestors = set(link_path)
            for joint_index, joint_link_index in enumerate(cspace_links):
                if joint_link_index in ancestors:
                    angular_mask[link_index, joint_index + 6] = 1.0
        self.angular_mask = angular_mask

        self.position_target = torch.zeros(
            batch_size, 3, dtype=torch.float32, device=device
        )
        self.current_position = torch.zeros_like(self.position_target)

    def set_target(self, position_target: torch.Tensor) -> None:
        expected = (self.batch_size, 3)
        if tuple(position_target.shape) != expected:
            raise ValueError(
                f"position_target must be {expected}, got "
                f"{tuple(position_target.shape)}."
            )
        self.position_target.copy_(position_target)

    def _position_and_jacobian(
        self, q: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        link_transforms, link_origin_jacobians = RobotKinematics.apply(
            q, self.robot_kinematics
        )
        link_origins = link_transforms[..., :3]
        link_rotations = _quat_xyzw_to_matrix(link_transforms[..., 3:7])
        world_com_offsets = torch.matmul(
            link_rotations,
            self.link_com_offsets[None, :, :, None],
        ).squeeze(-1)
        link_com_positions = link_origins + world_com_offsets

        origin_jacobians = link_origin_jacobians.transpose(2, 3)
        angular_jacobians = (
            self.robot_kinematics.batch_joint_axes_torch[:, None, :, :]
            * self.angular_mask[None, :, :, None]
        ).transpose(2, 3)
        # v_com = v_origin + omega x r =
        #         (J_origin - [r]_x J_omega) qdot.
        link_com_jacobians = origin_jacobians - torch.matmul(
            _skew(world_com_offsets), angular_jacobians
        )

        normalized_masses = self.link_masses / self.total_mass
        center_of_mass = torch.sum(
            normalized_masses[None, :, None] * link_com_positions,
            dim=1,
        )
        center_of_mass_jacobian = torch.sum(
            normalized_masses[None, :, None, None] * link_com_jacobians,
            dim=1,
        )
        self.current_position.copy_(center_of_mass.detach())
        return center_of_mass, center_of_mass_jacobian

    def compute_position(self, q: torch.Tensor) -> torch.Tensor:
        """Return the world-frame whole-body COM without applying the target."""
        center_of_mass, _ = self._position_and_jacobian(q)
        return center_of_mass

    def forward_position(
        self, q: torch.Tensor, features=None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del features
        center_of_mass, jacobian = self._position_and_jacobian(q)
        return center_of_mass - self.position_target, jacobian
