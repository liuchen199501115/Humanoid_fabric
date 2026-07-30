from __future__ import annotations

import torch

from fabrics_sim.fabrics.fabric import BaseFabric
from fabrics_sim.utils.path_utils import get_robot_urdf_path

from fabrics_sim.taskmaps.pelvis_pose import PelvisPoseTaskMap
from fabrics_sim.fabric_terms.pelvis_pose import PelvisPoseAttractor

from fabrics_sim.taskmaps.g1_end_effector_pose_taskmap import (
    G1EndEffectorPoseTaskMap,
)
from fabrics_sim.fabric_terms.g1_end_effector_pose_attractor import (
    G1EndEffectorPoseAttractor,
)


class G1PoseFabric(BaseFabric):
    """Whole-body G1 reference-motion Fabric with two tracking tasks."""

    NUM_ACTUATED_JOINTS = 29
    CONFIGURATION_DIMENSION = 7 + NUM_ACTUATED_JOINTS
    GENERALIZED_VELOCITY_DIMENSION = 6 + NUM_ACTUATED_JOINTS
    CONTROL_FREQUENCY = 50.0
    DEFAULT_TIMESTEP = 1.0 / CONTROL_FREQUENCY

    END_EFFECTOR_ORDER = (
        "left_foot",
        "right_foot",
        "left_hand",
        "right_hand",
    )

    def __init__(
        self,
        batch_size: int,
        device: str,
        timestep: float = DEFAULT_TIMESTEP,
        graph_capturable: bool = False,
        robot_dir_name: str = "g1",
        robot_name: str = "g1_29dof",
        frame_groups: dict[
            str, tuple[str, str, str, str]
        ] | None = None,
    ):
        """Construct the two-task G1 Fabric.

        Args:
            batch_size:
                Number of G1 instances.
            device:
                PyTorch/Warp device, e.g. ``"cpu"`` or ``"cuda:0"``.
            timestep:
                Controller timestep. The default is 0.02 s (50 Hz).
            graph_capturable:
                Whether persistent graph-capture buffers are used.
            robot_dir_name:
                Robot directory below ``models/robots/urdf``.
            robot_name:
                URDF filename without ``.urdf``.
            frame_groups:
                Optional actual G1 frame names for the four end effectors.
                Every entry is ``(origin, x_point, y_point, z_point)``.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if timestep <= 0.0:
            raise ValueError("timestep must be positive.")

        self.control_timestep = float(timestep)
        self.robot_dir_name = robot_dir_name
        self.g1_robot_name = robot_name
        self.end_effector_frame_groups = frame_groups

        super().__init__(
            device=device,
            batch_size=batch_size,
            timestep=timestep,
            fabric_params_filename="g1_params.yaml",
            graph_capturable=graph_capturable,
        )

        self.urdf_path = get_robot_urdf_path(
            robot_dir_name,
            robot_name,
        )

        # This call also allocates BaseFabric root-space buffers. In the
        # floating-base-adapted version it must allocate them with size 35.
        self.load_robot(
            robot_dir_name,
            robot_name,
            batch_size,
            floating_base=True,
        )

        self._check_floating_base_support()

        # Register both tracking tasks exactly once.
        self.construct_fabric()

    def _check_floating_base_support(self) -> None:
        """Fail early when this is run with the original fixed-base class."""
        root_dimension = int(self.root_metrics.shape[-1])

        if root_dimension != self.GENERALIZED_VELOCITY_DIMENSION:
            raise RuntimeError(
                "G1PoseFabric requires a floating-base-adapted BaseFabric "
                f"with root dimension {self.GENERALIZED_VELOCITY_DIMENSION}, "
                f"but load_robot() allocated dimension {root_dimension}. "
                "The original NVlabs/FABRICS load_robot() counts only the "
                "29 revolute joints. Extend it to allocate all natural-form "
                "buffers in the 35-D generalized-velocity tangent space."
            )

        acceleration_limits = self.fabric_params["joint_limits"][
            "acceleration"
        ]
        jerk_limits = self.fabric_params["joint_limits"]["jerk"]

        if len(acceleration_limits) != self.GENERALIZED_VELOCITY_DIMENSION:
            raise ValueError(
                "g1_pose_params.yaml must contain 35 acceleration limits."
            )
        if len(jerk_limits) != self.GENERALIZED_VELOCITY_DIMENSION:
            raise ValueError(
                "g1_pose_params.yaml must contain 35 jerk limits."
            )

    # ------------------------------------------------------------------
    # Task construction
    # ------------------------------------------------------------------
    def add_pelvis_pose_task(self) -> None:
        """Add the 6-D pelvis position/orientation tracking task."""
        taskmap_name = "pelvis_pose"
        fabric_name = "pelvis_pose_attractor"

        self.pelvis_pose_taskmap = PelvisPoseTaskMap(
            batch_size=self.batch_size,
            num_joints=self.NUM_ACTUATED_JOINTS,
            device=self.device,
            track_position=(True, True, True),
            track_orientation=(True, True, True),
        )

        self.add_taskmap(
            taskmap_name,
            self.pelvis_pose_taskmap,
            graph_capturable=self.graph_capturable,
        )

        pelvis_term = PelvisPoseAttractor(
            is_forcing_policy=True,
            params=self.fabric_params[fabric_name],
            device=self.device,
            graph_capturable=self.graph_capturable,
        )

        self.add_fabric(
            taskmap_name,
            fabric_name,
            pelvis_term,
        )

    def add_end_effector_pose_task(self) -> None:
        """Add the unified 24-D feet-and-hands SE(3) tracking task."""
        taskmap_name = "end_effector_pose"
        fabric_name = "end_effector_pose_attractor"

        self.end_effector_pose_taskmap = G1EndEffectorPoseTaskMap(
            urdf_path=self.urdf_path,
            batch_size=self.batch_size,
            device=self.device,
            frame_groups=self.end_effector_frame_groups,
        )

        self.add_taskmap(
            taskmap_name,
            self.end_effector_pose_taskmap,
            graph_capturable=self.graph_capturable,
        )

        end_effector_term = G1EndEffectorPoseAttractor(
            is_forcing_policy=True,
            params=self.fabric_params[fabric_name],
            device=self.device,
            graph_capturable=self.graph_capturable,
        )

        self.add_fabric(
            taskmap_name,
            fabric_name,
            end_effector_term,
        )

    def construct_fabric(self) -> None:
        """Register the two tracking tasks.

        This function is called once from ``__init__``. Do not call it in the
        50 Hz control loop.
        """
        self.add_pelvis_pose_task()
        self.add_end_effector_pose_task()
        self.add_root_regularizer()

    def add_root_regularizer(self) -> None:
        """Keep the 35-D pullback metric invertible.

        The two tracking tasks have only 30 task dimensions in total, so their
        combined 35-D pullback cannot be full rank without regularization.
        """
        from fabrics_sim.fabric_terms.isotropic_inertia import IsotropicInertia
        from fabrics_sim.taskmaps.floating_base_identity import (
            FloatingBaseIdentityMap,
        )

        taskmap_name = "root_regularizer"
        self.add_taskmap(
            taskmap_name,
            FloatingBaseIdentityMap(
                self.NUM_ACTUATED_JOINTS,
                self.device,
            ),
            graph_capturable=self.graph_capturable,
        )
        self.add_fabric(
            taskmap_name,
            "isotropic_inertia",
            IsotropicInertia(
                isotropic_mass=1.0e-4,
                device=self.device,
                graph_capturable=self.graph_capturable,
            ),
        )

    def integrate_root_position(self, q, qd, timestep):
        """Integrate quaternion configuration by a world-frame base twist."""
        position = q[:, :3] + timestep * qd[:, :3]
        quaternion = q[:, 3:7]
        angular_step = timestep * qd[:, 3:6]

        angle = torch.linalg.vector_norm(angular_step, dim=-1, keepdim=True)
        half_angle = 0.5 * angle
        scale = torch.where(
            angle > 1.0e-8,
            torch.sin(half_angle) / angle,
            0.5 - angle.square() / 48.0,
        )
        delta = torch.cat((scale * angular_step, torch.cos(half_angle)), dim=-1)
        dv, dw = delta[:, :3], delta[:, 3:4]
        qv, qw = quaternion[:, :3], quaternion[:, 3:4]
        integrated_quaternion = torch.cat(
            (
                dw * qv + qw * dv + torch.linalg.cross(dv, qv),
                dw * qw - torch.sum(dv * qv, dim=-1, keepdim=True),
            ),
            dim=-1,
        )
        integrated_quaternion = integrated_quaternion / (
            torch.linalg.vector_norm(
                integrated_quaternion, dim=-1, keepdim=True
            ).clamp_min(1.0e-8)
        )
        joints = q[:, 7:] + timestep * qd[:, 6:]
        return torch.cat((position, integrated_quaternion, joints), dim=-1)

    # ------------------------------------------------------------------
    # Target updates
    # ------------------------------------------------------------------
    def set_pelvis_target(
        self,
        pelvis_position_target: torch.Tensor,
        pelvis_quaternion_target_xyzw: torch.Tensor,
        pelvis_linear_velocity_target: torch.Tensor | None = None,
        pelvis_angular_velocity_target: torch.Tensor | None = None,
    ) -> None:
        """Update the retargeted pelvis reference.

        Shapes:
            position:             (batch_size, 3)
            quaternion_xyzw:      (batch_size, 4)
            linear velocity:      (batch_size, 3), optional
            angular velocity:     (batch_size, 3), optional
        """
        expected_position = (self.batch_size, 3)
        expected_quaternion = (self.batch_size, 4)

        if tuple(pelvis_position_target.shape) != expected_position:
            raise ValueError(
                f"pelvis_position_target must be {expected_position}, "
                f"got {tuple(pelvis_position_target.shape)}."
            )
        if (
            tuple(pelvis_quaternion_target_xyzw.shape)
            != expected_quaternion
        ):
            raise ValueError(
                "pelvis_quaternion_target_xyzw must be "
                f"{expected_quaternion}, got "
                f"{tuple(pelvis_quaternion_target_xyzw.shape)}."
            )

        self.pelvis_pose_taskmap.set_target(
            position_target=pelvis_position_target,
            orientation_target=pelvis_quaternion_target_xyzw,
        )

        if (
            pelvis_linear_velocity_target is None
            and pelvis_angular_velocity_target is None
        ):
            target_twist = None
        elif (
            pelvis_linear_velocity_target is not None
            and pelvis_angular_velocity_target is not None
        ):
            if tuple(pelvis_linear_velocity_target.shape) != expected_position:
                raise ValueError(
                    "pelvis_linear_velocity_target must be "
                    f"{expected_position}."
                )
            if tuple(pelvis_angular_velocity_target.shape) != expected_position:
                raise ValueError(
                    "pelvis_angular_velocity_target must be "
                    f"{expected_position}."
                )
            target_twist = torch.cat(
                (
                    pelvis_linear_velocity_target,
                    pelvis_angular_velocity_target,
                ),
                dim=-1,
            )
        else:
            raise ValueError(
                "Provide both pelvis target velocities or neither one."
            )

        self.fabrics_features["pelvis_pose"][
            "pelvis_pose_attractor"
        ] = target_twist

    def set_end_effector_targets(
        self,
        position_target: torch.Tensor,
        quaternion_target_xyzw: torch.Tensor,
        linear_velocity_target: torch.Tensor | None = None,
        angular_velocity_target: torch.Tensor | None = None,
    ) -> None:
        """Update feet-and-hands retargeting references.

        End-effector order:
            left foot, right foot, left hand, right hand.

        Shapes:
            position:             (batch_size, 4, 3)
            quaternion_xyzw:      (batch_size, 4, 4)
            linear velocity:      (batch_size, 4, 3), optional
            angular velocity:     (batch_size, 4, 3), optional
        """
        expected_position = (self.batch_size, 4, 3)
        expected_quaternion = (self.batch_size, 4, 4)

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

        self.end_effector_pose_taskmap.set_target(
            position_target=position_target,
            quaternion_target_xyzw=quaternion_target_xyzw,
        )

        if (
            linear_velocity_target is None
            and angular_velocity_target is None
        ):
            target_twist = None
        elif (
            linear_velocity_target is not None
            and angular_velocity_target is not None
        ):
            if tuple(linear_velocity_target.shape) != expected_position:
                raise ValueError(
                    "linear_velocity_target must be "
                    f"{expected_position}."
                )
            if tuple(angular_velocity_target.shape) != expected_position:
                raise ValueError(
                    "angular_velocity_target must be "
                    f"{expected_position}."
                )

            # (B, 4, 3) + (B, 4, 3) -> (B, 4, 6) -> (B, 24)
            target_twist = torch.cat(
                (
                    linear_velocity_target,
                    angular_velocity_target,
                ),
                dim=-1,
            ).reshape(self.batch_size, 24)
        else:
            raise ValueError(
                "Provide both end-effector target velocities or neither one."
            )

        self.fabrics_features["end_effector_pose"][
            "end_effector_pose_attractor"
        ] = target_twist

    def set_motion_targets(
        self,
        pelvis_position_target: torch.Tensor,
        pelvis_quaternion_target_xyzw: torch.Tensor,
        end_effector_position_target: torch.Tensor,
        end_effector_quaternion_target_xyzw: torch.Tensor,
        pelvis_linear_velocity_target: torch.Tensor | None = None,
        pelvis_angular_velocity_target: torch.Tensor | None = None,
        end_effector_linear_velocity_target: torch.Tensor | None = None,
        end_effector_angular_velocity_target: torch.Tensor | None = None,
    ) -> None:
        """Update both tracking tasks in one control-loop call."""
        self.set_pelvis_target(
            pelvis_position_target=pelvis_position_target,
            pelvis_quaternion_target_xyzw=(
                pelvis_quaternion_target_xyzw
            ),
            pelvis_linear_velocity_target=(
                pelvis_linear_velocity_target
            ),
            pelvis_angular_velocity_target=(
                pelvis_angular_velocity_target
            ),
        )

        self.set_end_effector_targets(
            position_target=end_effector_position_target,
            quaternion_target_xyzw=(
                end_effector_quaternion_target_xyzw
            ),
            linear_velocity_target=(
                end_effector_linear_velocity_target
            ),
            angular_velocity_target=(
                end_effector_angular_velocity_target
            ),
        )

    # ------------------------------------------------------------------
    # Fabric evaluation
    # ------------------------------------------------------------------
    def _validate_state(
        self,
        q: torch.Tensor,
        qd: torch.Tensor,
    ) -> None:
        expected_q = (self.batch_size, self.CONFIGURATION_DIMENSION)
        expected_qd = (
            self.batch_size,
            self.GENERALIZED_VELOCITY_DIMENSION,
        )

        if tuple(q.shape) != expected_q:
            raise ValueError(
                f"q must have shape {expected_q}, got {tuple(q.shape)}."
            )
        if tuple(qd.shape) != expected_qd:
            raise ValueError(
                f"qd must have shape {expected_qd}, got {tuple(qd.shape)}."
            )

    def compute_acceleration(
        self,
        q: torch.Tensor,
        qd: torch.Tensor,
        timestep: float | None = None,
    ) -> torch.Tensor:
        """Return the combined 35-D Fabric acceleration.

        BaseFabric.forward/eval_natural returns ``(M, f, M_inv)``. The
        canonical acceleration is therefore ``qdd = -M_inv @ f``.
        """
        self._validate_state(q, qd)
        dt = self.control_timestep if timestep is None else float(timestep)

        metric, force, metric_inverse = self.forward(q, qd, dt)
        del metric

        qdd = -torch.bmm(
            metric_inverse,
            force.unsqueeze(-1),
        ).squeeze(-1)

        if tuple(qdd.shape) != (
            self.batch_size,
            self.GENERALIZED_VELOCITY_DIMENSION,
        ):
            raise RuntimeError(
                "Unexpected Fabric acceleration shape: "
                f"{tuple(qdd.shape)}."
            )
        if not torch.isfinite(qdd).all():
            raise FloatingPointError(
                "Fabric acceleration contains NaN or Inf. The combined "
                "35x35 metric may be singular; add root-space metric "
                "regularization or a low-weight joint-space stabilizer."
            )

        return qdd
