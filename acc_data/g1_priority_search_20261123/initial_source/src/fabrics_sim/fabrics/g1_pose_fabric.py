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
from fabrics_sim.taskmaps.g1_center_of_mass import G1CenterOfMassTaskMap
from fabrics_sim.fabric_terms.center_of_mass import CenterOfMassAttractor


class G1PoseFabric(BaseFabric):
    """Whole-body G1 Fabric with pelvis, COM, feet and hand tasks."""

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
        task_profile: str = "legacy",
        joint_posture_enabled: bool = True,
    ):
        """Construct the whole-body G1 Fabric.

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
            joint_posture_enabled:
                Include joint posture tracking. References can still be updated
                when disabled without reactivating the task.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if timestep <= 0.0:
            raise ValueError("timestep must be positive.")
        if task_profile not in ("legacy", "dynamics-aware"):
            raise ValueError("task_profile must be legacy or dynamics-aware.")
        if task_profile == "dynamics-aware" and graph_capturable:
            raise ValueError("Dynamics-aware task evaluation does not yet support CUDA graph capture.")
        self.task_profile = task_profile
        self.joint_posture_enabled = joint_posture_enabled

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

        # Register all tracking tasks exactly once.
        self.construct_fabric()
        self.dynamics_tasks = None
        if task_profile == "dynamics-aware":
            from fabrics_sim.fabrics.dynamics_aware import DynamicsAwareTasks
            self.dynamics_tasks = DynamicsAwareTasks(self)

    def compute_task_requests(self, q, qd):
        """Physical acceleration/momentum-rate intentions for task-level WBC."""
        self._validate_state(q, qd)
        if self.dynamics_tasks is None:
            raise RuntimeError("Task requests require task_profile='dynamics-aware'.")
        return self.dynamics_tasks.compute(q, qd)

    def set_contact_state(self, active):
        if self.dynamics_tasks is None:
            raise RuntimeError("Contact gating requires the dynamics-aware profile.")
        self.dynamics_tasks.set_contact_state(active)

    def set_centroidal_momentum_target(self, reference, reference_rate=None):
        if self.dynamics_tasks is None:
            raise RuntimeError("Momentum tracking requires the dynamics-aware profile.")
        self.dynamics_tasks.set_momentum_target(reference, reference_rate)

    def set_body_orientation_target(self, quaternion_xyzw, omega_ref=None, alpha_ref=None, link="pelvis"):
        if self.dynamics_tasks is None:
            raise RuntimeError("Body orientation requests require the dynamics-aware profile.")
        self.dynamics_tasks.set_body_target(quaternion_xyzw, omega_ref, alpha_ref, link)

    def set_collision_obstacles(self, spheres):
        if self.dynamics_tasks is None:
            raise RuntimeError("Collision requests require the dynamics-aware profile.")
        self.dynamics_tasks.set_obstacles(spheres)

    def set_hand_task_active(self, active):
        if self.dynamics_tasks is None:
            raise RuntimeError("Hand gating requires the dynamics-aware profile.")
        mask = self.dynamics_tasks.hand_active
        if active.shape != mask.shape or active.device != mask.device or active.dtype != torch.bool:
            raise ValueError("Hand activation must be boolean [B,2] on the model device.")
        mask.copy_(active)

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

    def add_center_of_mass_task(self) -> None:
        """Add whole-body COM tracking using URDF link inertial data."""
        taskmap_name = "center_of_mass"
        fabric_name = "center_of_mass_attractor"
        self.center_of_mass_taskmap = G1CenterOfMassTaskMap(
            urdf_path=self.urdf_path,
            batch_size=self.batch_size,
            device=self.device,
        )
        self.add_taskmap(
            taskmap_name,
            self.center_of_mass_taskmap,
            graph_capturable=self.graph_capturable,
        )
        self.add_fabric(
            taskmap_name,
            fabric_name,
            CenterOfMassAttractor(
                is_forcing_policy=True,
                params=self.fabric_params[fabric_name],
                device=self.device,
                graph_capturable=self.graph_capturable,
            ),
        )

    def _add_end_effector_group_task(
        self,
        taskmap_name: str,
        fabric_name: str,
        end_effector_order: tuple[str, str],
    ) -> G1EndEffectorPoseTaskMap:
        """Add one independent 12-D two-end-effector SE(3) task."""
        taskmap = G1EndEffectorPoseTaskMap(
            urdf_path=self.urdf_path,
            batch_size=self.batch_size,
            device=self.device,
            frame_groups=self.end_effector_frame_groups,
            end_effector_order=end_effector_order,
        )

        self.add_taskmap(
            taskmap_name,
            taskmap,
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
        return taskmap

    def add_feet_pose_task(self) -> None:
        """Add the independent 12-D left/right-foot pose task."""
        self.feet_pose_taskmap = self._add_end_effector_group_task(
            "feet_pose",
            "feet_pose_attractor",
            ("left_foot", "right_foot"),
        )

    def add_hand_pose_task(self) -> None:
        """Add the independent 12-D left/right-hand pose task."""
        self.hand_pose_taskmap = self._add_end_effector_group_task(
            "hand_pose",
            "hand_pose_attractor",
            ("left_hand", "right_hand"),
        )

    def construct_fabric(self) -> None:
        """Register pelvis, COM, feet and hand tracking tasks.

        This function is called once from ``__init__``. Do not call it in the
        50 Hz control loop.
        """
        self.add_pelvis_pose_task()
        self.add_center_of_mass_task()
        self.add_feet_pose_task()
        self.add_hand_pose_task()
        self.add_joint_posture_task()
        self.add_root_regularizer()

    def add_joint_posture_task(self) -> None:
        from fabrics_sim.taskmaps.joint_posture import JointPostureTaskMap
        from fabrics_sim.fabric_terms.joint_posture import JointPostureAttractor

        self.joint_posture_taskmap = JointPostureTaskMap(self.batch_size, self.device)
        self.add_taskmap("joint_posture", self.joint_posture_taskmap,
                         graph_capturable=self.graph_capturable)
        self.joint_posture_term = JointPostureAttractor(
            is_forcing_policy=True,
            params=self.fabric_params["joint_posture_attractor"],
            device=self.device, graph_capturable=self.graph_capturable,
        )
        # Other users of G1PoseFabric retain the previous behavior until
        # they explicitly provide a joint posture target.
        self.joint_posture_term.params["toggle"] = 0.0
        self.add_fabric("joint_posture", "joint_posture_attractor", self.joint_posture_term)

    def set_joint_posture_target(self, position, velocity=None, acceleration=None) -> None:
        self.joint_posture_taskmap.set_target(position)
        if velocity is not None and (velocity.shape != position.shape or not torch.isfinite(velocity).all()):
            raise ValueError("Joint velocity target must be finite and match position shape.")
        if acceleration is not None and (acceleration.shape != position.shape or not torch.isfinite(acceleration).all()):
            raise ValueError("Joint acceleration target must be finite and match position shape.")
        self.fabrics_features["joint_posture"]["joint_posture_attractor"] = velocity if acceleration is None else {
            "velocity": torch.zeros_like(position) if velocity is None else velocity,
            "acceleration": acceleration,
        }
        self.joint_posture_term.params["toggle"] = float(self.joint_posture_enabled)

    def add_root_regularizer(self) -> None:
        """Keep the 35-D pullback metric invertible.

        The tracking-task pullback is not guaranteed to be full rank in the
        35-D tangent space, so retain a small isotropic regularizer.
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
    def set_center_of_mass_target(
        self,
        position_target: torch.Tensor,
        velocity_target: torch.Tensor | None = None,
        acceleration_target: torch.Tensor | None = None,
    ) -> None:
        """Set world-frame COM position, velocity and acceleration references."""
        expected = (self.batch_size, 3)
        if tuple(position_target.shape) != expected:
            raise ValueError(f"position_target must be {expected}.")
        if velocity_target is not None and tuple(velocity_target.shape) != expected:
            raise ValueError(f"velocity_target must be {expected}.")
        if acceleration_target is not None and tuple(acceleration_target.shape) != expected:
            raise ValueError(f"acceleration_target must be {expected}.")
        self.center_of_mass_taskmap.set_target(position_target)
        self.fabrics_features["center_of_mass"]["center_of_mass_attractor"] = (
            velocity_target if acceleration_target is None else {
                "velocity": torch.zeros_like(position_target) if velocity_target is None else velocity_target,
                "acceleration": acceleration_target,
            })

    def compute_center_of_mass(self, q: torch.Tensor) -> torch.Tensor:
        """Compute the G1 whole-body COM represented by the Fabric URDF."""
        return self.center_of_mass_taskmap.compute_position(q)

    def set_pelvis_target(
        self,
        pelvis_position_target: torch.Tensor,
        pelvis_quaternion_target_xyzw: torch.Tensor,
        pelvis_linear_velocity_target: torch.Tensor | None = None,
        pelvis_angular_velocity_target: torch.Tensor | None = None,
        pelvis_linear_acceleration_target: torch.Tensor | None = None,
        pelvis_angular_acceleration_target: torch.Tensor | None = None,
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

        references = (
            pelvis_linear_velocity_target,
            pelvis_angular_velocity_target,
            pelvis_linear_acceleration_target,
            pelvis_angular_acceleration_target,
        )
        if all(value is None for value in references):
            target_twist = None
        elif (
            pelvis_linear_velocity_target is not None
            and pelvis_angular_velocity_target is not None
            and pelvis_linear_acceleration_target is None
            and pelvis_angular_acceleration_target is None
        ):
            if tuple(pelvis_linear_velocity_target.shape) != expected_position:
                raise ValueError(f"pelvis_linear_velocity_target must be {expected_position}.")
            if tuple(pelvis_angular_velocity_target.shape) != expected_position:
                raise ValueError(f"pelvis_angular_velocity_target must be {expected_position}.")
            target_twist = torch.cat(
                (pelvis_linear_velocity_target, pelvis_angular_velocity_target), dim=-1
            )
        elif all(value is not None for value in references):
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
            if tuple(pelvis_linear_acceleration_target.shape) != expected_position:
                raise ValueError("pelvis_linear_acceleration_target must be (batch_size, 3).")
            if tuple(pelvis_angular_acceleration_target.shape) != expected_position:
                raise ValueError("pelvis_angular_acceleration_target must be (batch_size, 3).")
            target_twist = {
                "velocity": torch.cat((pelvis_linear_velocity_target, pelvis_angular_velocity_target), dim=-1),
                "acceleration": torch.cat((pelvis_linear_acceleration_target, pelvis_angular_acceleration_target), dim=-1),
            }
        else:
            raise ValueError(
                "Provide pelvis target velocity and acceleration references together."
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
        linear_acceleration_target: torch.Tensor | None = None,
        angular_acceleration_target: torch.Tensor | None = None,
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

        references = (
            linear_velocity_target, angular_velocity_target,
            linear_acceleration_target, angular_acceleration_target,
        )
        velocity_only = (
            linear_velocity_target is not None and angular_velocity_target is not None
            and linear_acceleration_target is None and angular_acceleration_target is None
        )
        if not (all(value is None for value in references) or velocity_only or all(value is not None for value in references)):
            raise ValueError(
                "Provide all end-effector velocity and acceleration references or neither."
            )
        if linear_velocity_target is not None:
            if tuple(linear_velocity_target.shape) != expected_position:
                raise ValueError(f"linear_velocity_target must be {expected_position}.")
            if tuple(angular_velocity_target.shape) != expected_position:
                raise ValueError(f"angular_velocity_target must be {expected_position}.")
            if linear_acceleration_target is not None:
                if tuple(linear_acceleration_target.shape) != expected_position:
                    raise ValueError(f"linear_acceleration_target must be {expected_position}.")
                if tuple(angular_acceleration_target.shape) != expected_position:
                    raise ValueError(f"angular_acceleration_target must be {expected_position}.")

        self.set_feet_targets(
            position_target[:, :2], quaternion_target_xyzw[:, :2],
            None if linear_velocity_target is None else linear_velocity_target[:, :2],
            None if angular_velocity_target is None else angular_velocity_target[:, :2],
            None if linear_acceleration_target is None else linear_acceleration_target[:, :2],
            None if angular_acceleration_target is None else angular_acceleration_target[:, :2],
        )
        self.set_hand_targets(
            position_target[:, 2:], quaternion_target_xyzw[:, 2:],
            None if linear_velocity_target is None else linear_velocity_target[:, 2:],
            None if angular_velocity_target is None else angular_velocity_target[:, 2:],
            None if linear_acceleration_target is None else linear_acceleration_target[:, 2:],
            None if angular_acceleration_target is None else angular_acceleration_target[:, 2:],
        )

    def _set_group_targets(
        self, task_name, fabric_name, taskmap, position_target,
        quaternion_target_xyzw, linear_velocity_target=None,
        angular_velocity_target=None, linear_acceleration_target=None,
        angular_acceleration_target=None,
    ) -> None:
        expected_position = (self.batch_size, 2, 3)
        expected_quaternion = (self.batch_size, 2, 4)
        if tuple(position_target.shape) != expected_position:
            raise ValueError(f"position_target must be {expected_position}.")
        if tuple(quaternion_target_xyzw.shape) != expected_quaternion:
            raise ValueError(f"quaternion_target_xyzw must be {expected_quaternion}.")
        taskmap.set_target(position_target, quaternion_target_xyzw)
        references = (linear_velocity_target, angular_velocity_target,
                      linear_acceleration_target, angular_acceleration_target)
        if all(value is None for value in references):
            target_twist = None
        elif (
            linear_velocity_target is not None
            and angular_velocity_target is not None
            and linear_acceleration_target is None
            and angular_acceleration_target is None
        ):
            if tuple(linear_velocity_target.shape) != expected_position:
                raise ValueError(f"linear_velocity_target must be {expected_position}.")
            if tuple(angular_velocity_target.shape) != expected_position:
                raise ValueError(f"angular_velocity_target must be {expected_position}.")
            target_twist = torch.cat(
                (linear_velocity_target, angular_velocity_target), dim=-1
            ).reshape(self.batch_size, 12)
        elif all(value is not None for value in references):
            if tuple(linear_velocity_target.shape) != expected_position:
                raise ValueError(f"linear_velocity_target must be {expected_position}.")
            if tuple(angular_velocity_target.shape) != expected_position:
                raise ValueError(f"angular_velocity_target must be {expected_position}.")
            if tuple(linear_acceleration_target.shape) != expected_position:
                raise ValueError(f"linear_acceleration_target must be {expected_position}.")
            if tuple(angular_acceleration_target.shape) != expected_position:
                raise ValueError(f"angular_acceleration_target must be {expected_position}.")
            target_twist = {
                "velocity": torch.cat((linear_velocity_target, angular_velocity_target), dim=-1).reshape(self.batch_size, 12),
                "acceleration": torch.cat((linear_acceleration_target, angular_acceleration_target), dim=-1).reshape(self.batch_size, 12),
            }
        else:
            raise ValueError("Provide all group velocity and acceleration references or neither.")
        self.fabrics_features[task_name][fabric_name] = target_twist

    def set_feet_targets(self, position_target, quaternion_target_xyzw,
                         linear_velocity_target=None, angular_velocity_target=None,
                         linear_acceleration_target=None, angular_acceleration_target=None):
        """Update the independent left/right-foot tracking references."""
        self._set_group_targets(
            "feet_pose", "feet_pose_attractor", self.feet_pose_taskmap,
            position_target, quaternion_target_xyzw,
            linear_velocity_target, angular_velocity_target,
            linear_acceleration_target, angular_acceleration_target,
        )

    def anchor_support_feet(self, anchors, active):
        """Replace planted-foot references, preserving every swing-foot reference.

        Call after setting motion targets and refreshing touchdown anchors.
        Anchors are world-frame xyz + xyzw, shaped (batch, 2, 7).
        """
        if tuple(anchors.shape) != (self.batch_size, 2, 7):
            raise ValueError("anchors must have shape (batch_size, 2, 7).")
        if tuple(active.shape) != (self.batch_size, 2) or active.dtype != torch.bool:
            raise ValueError("active must be a boolean (batch_size, 2) tensor.")
        if not torch.isfinite(anchors).all():
            raise ValueError("anchors must be finite.")
        mask = active.unsqueeze(-1)
        position = torch.where(mask, anchors[..., :3], self.feet_pose_taskmap.position_target)
        quaternion = torch.where(mask, anchors[..., 3:], self.feet_pose_taskmap.quaternion_target)
        features = self.fabrics_features["feet_pose"]["feet_pose_attractor"]
        velocity = features.get("velocity") if isinstance(features, dict) else features
        acceleration = features.get("acceleration") if isinstance(features, dict) else None

        def masked_parts(reference):
            if reference is None:
                return None, None
            reference = reference.reshape(self.batch_size, 2, 6)
            reference = torch.where(mask, torch.zeros_like(reference), reference)
            return reference[..., :3], reference[..., 3:]

        linear_v, angular_v = masked_parts(velocity)
        linear_a, angular_a = masked_parts(acceleration)
        self.set_feet_targets(position, quaternion, linear_v, angular_v, linear_a, angular_a)

    def set_hand_targets(self, position_target, quaternion_target_xyzw,
                         linear_velocity_target=None, angular_velocity_target=None,
                         linear_acceleration_target=None, angular_acceleration_target=None):
        """Update the independent left/right-hand tracking references."""
        self._set_group_targets(
            "hand_pose", "hand_pose_attractor", self.hand_pose_taskmap,
            position_target, quaternion_target_xyzw,
            linear_velocity_target, angular_velocity_target,
            linear_acceleration_target, angular_acceleration_target,
        )

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
        pelvis_linear_acceleration_target: torch.Tensor | None = None,
        pelvis_angular_acceleration_target: torch.Tensor | None = None,
        end_effector_linear_acceleration_target: torch.Tensor | None = None,
        end_effector_angular_acceleration_target: torch.Tensor | None = None,
    ) -> None:
        """Update pelvis, feet and hand tracking tasks in one call."""
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
            pelvis_linear_acceleration_target=pelvis_linear_acceleration_target,
            pelvis_angular_acceleration_target=pelvis_angular_acceleration_target,
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
            linear_acceleration_target=end_effector_linear_acceleration_target,
            angular_acceleration_target=end_effector_angular_acceleration_target,
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
        if self.dynamics_tasks is not None:
            requests = self.compute_task_requests(q, qd)
            # Prioritized Jacobians can be nearly rank deficient. Keep their
            # natural-form accumulation in double precision through the solve.
            compose_dtype = torch.float64 if self.dynamics_tasks.priority_mode == 'paper' else q.dtype
            metric = self.dynamics_tasks.config["regularization"] * torch.eye(
                35, device=q.device, dtype=compose_dtype).expand(q.shape[0], -1, -1).clone()
            force = torch.zeros_like(qd, dtype=compose_dtype)
            for request in requests:
                task_metric, task_force = request.natural_form()
                metric += task_metric
                force += task_force
            acceleration = torch.linalg.solve(metric, -force.unsqueeze(-1)).squeeze(-1)
            if not torch.isfinite(acceleration).all():
                raise FloatingPointError("Dynamics-aware acceleration contains NaN or Inf.")
            # A constrained downstream solve needs both the minimizer and
            # its metric to preserve the original soft-task tradeoffs.
            self.dynamics_tasks.last_snapshot['fabric_acc_metric'] = metric
            return acceleration.to(q.dtype)
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
