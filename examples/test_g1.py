"""Test the two G1 Fabric tasks followed by contact-consistent projection.

Data flow:
    qdd_fabric [1, 35]
        -> Isaac Sim M [1, 35, 35] and foot J [1, 12, 35]
        -> dynamically consistent projection
        -> qdd_contact_consistent [1, 35]
        -> floating-base dynamics acceleration QP
        -> qdd_dynamics_feasible [1, 35]
        -> actuated-joint semi-implicit integration [1, 29]

This is a one-step numerical integration test. It intentionally takes the
contact mode as an argument; force-based contact detection belongs in the
closed-loop simulator, where real foot contact forces are available.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import traceback
import warnings

import numpy as np
import torch
import warp as wp

# Warp 1.8.1 probes ``tensor.grad`` while wrapping Torch tensors. PyTorch
# warns when that tensor is non-leaf even though this controller never calls
# backward(). Suppress only this known bridge warning; all other warnings stay
# visible.
warnings.filterwarnings(
    "ignore",
    message=(
        r"The \.grad attribute of a Tensor that is not a leaf Tensor "
        r"is being accessed\..*"
    ),
    category=UserWarning,
    module=r"warp\.torch",
)

from fabrics_sim.fabrics.g1_pose_fabric import G1PoseFabric
from fabrics_sim.solver.AccelerationQP import G1DynamicsAccelerationQP
from fabrics_sim.solver.dynamic_consistent_gpu import (
    ContactDetectorConfig,
    G1FabricContactProjector,
    ProjectionConfig,
)

from test_g1_two_tasks import (
    FABRIC_TIMESTEP,
    G1_JOINT_NAMES,
    G1_USD_PATH,
    PD_TIMESTEP,
    load_motion_trajectory,
    unitree_g1_effort_limits,
    unitree_g1_pd_gains,
)


LEFT_FOOT_BODY = "left_ankle_roll_link"
RIGHT_FOOT_BODY = "right_ankle_roll_link"
OUTPUT_PATH = (
    Path(__file__).resolve().parents[1]
    / "acc_data/g1_two_task_dynamics_feasible.npz"
)


def configure_two_task_targets(
    fabric: G1PoseFabric,
    q: torch.Tensor,
) -> None:
    """Use one complete retargeted pose as the two tracking targets."""
    fabric.end_effector_pose_taskmap(q, None)
    transforms = (
        fabric.end_effector_pose_taskmap
        .robot_kinematics.batch_link_transforms_torch[
            :, fabric.end_effector_pose_taskmap.link_indices
        ]
        .clone()
    )
    fabric.set_motion_targets(
        pelvis_position_target=q[:, :3].clone(),
        pelvis_quaternion_target_xyzw=q[:, 3:7].clone(),
        end_effector_position_target=transforms[..., :3].clone(),
        end_effector_quaternion_target_xyzw=transforms[..., 3:7].clone(),
    )


def _to_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _clone_contiguous(value):
    """Own a contiguous dynamics buffer for either Torch or NumPy backends."""
    if isinstance(value, torch.Tensor):
        return value.clone().contiguous()
    return np.ascontiguousarray(np.asarray(value).copy())


def read_isaac_dynamics(robot):
    """Read full floating-base G1 dynamics from Isaac/PhysX."""
    # Isaac Lab assets expose ``root_physx_view``. Isaac Sim Core's
    # ``Articulation`` stores the same PhysX tensor view as ``_physics_view``.
    view = getattr(robot, "root_physx_view", None)
    if view is None:
        view = getattr(robot, "_physics_view", None)
    if view is None:
        raise RuntimeError(
            "Isaac articulation has no initialized PhysX view. "
            "Call world.reset() before reading dynamics."
        )

    mass_matrix = _clone_contiguous(
        view.get_generalized_mass_matrices()
    )
    coriolis = _clone_contiguous(
        view.get_coriolis_and_centrifugal_compensation_forces()
    )
    gravity = _clone_contiguous(
        view.get_gravity_compensation_forces()
    )
    bias_force = coriolis + gravity
    batch_size = mass_matrix.shape[0]

    if mass_matrix.shape != (batch_size, 35, 35):
        raise RuntimeError(
            "Expected floating-base mass matrix "
            f"[B,35,35], got {tuple(mass_matrix.shape)}"
        )
    if coriolis.shape != (batch_size, 35):
        raise RuntimeError(
            "Expected floating-base Coriolis force "
            f"[B,35], got {tuple(coriolis.shape)}"
        )
    if gravity.shape != (batch_size, 35):
        raise RuntimeError(
            "Expected floating-base gravity force "
            f"[B,35], got {tuple(gravity.shape)}"
        )
    if bias_force.shape != (batch_size, 35):
        raise RuntimeError(
            "Expected floating-base bias force "
            f"[B,35], got {tuple(bias_force.shape)}"
        )

    return mass_matrix, bias_force


def read_live_isaac_snapshot(
    robot,
    isaac_joint_names: list[str],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Read the live state and dynamics in FABRICS generalized order."""
    isaac_to_fabric_joint = np.asarray(
        [isaac_joint_names.index(name) for name in G1_JOINT_NAMES],
        dtype=np.int64,
    )
    coordinate_index = np.asarray(
        list(range(6))
        + [6 + index for index in isaac_to_fabric_joint],
        dtype=np.int64,
    )

    positions, orientations_wxyz = robot.get_world_poses()
    positions = _to_numpy(positions)
    orientations_wxyz = _to_numpy(orientations_wxyz)
    joint_positions_isaac = _to_numpy(robot.get_joint_positions())
    base_velocity = _to_numpy(robot.get_velocities())
    joint_velocity_isaac = _to_numpy(robot.get_joint_velocities())

    q_numpy = np.concatenate(
        (
            positions,
            orientations_wxyz[:, 1:4],
            orientations_wxyz[:, 0:1],
            joint_positions_isaac[:, isaac_to_fabric_joint],
        ),
        axis=1,
    )
    qd_numpy = np.concatenate(
        (
            base_velocity[:, :6],
            joint_velocity_isaac[:, isaac_to_fabric_joint],
        ),
        axis=1,
    )

    mass_raw, bias_raw = read_isaac_dynamics(robot)
    mass_isaac = _to_numpy(mass_raw)
    bias_isaac = _to_numpy(bias_raw)
    mass_fabric = mass_isaac[
        :, coordinate_index[:, None], coordinate_index[None, :]
    ]
    bias_fabric = bias_isaac[:, coordinate_index]

    jacobians = _to_numpy(robot.get_jacobians())
    body_names = list(robot.body_names)
    if jacobians.shape[1] == len(body_names):
        jacobian_body_names = body_names
    elif jacobians.shape[1] == len(body_names) - 1:
        jacobian_body_names = body_names[1:]
    else:
        raise RuntimeError(
            f"Cannot align {jacobians.shape[1]} Jacobian rows with "
            f"{len(body_names)} Isaac body names."
        )
    left_index = jacobian_body_names.index(LEFT_FOOT_BODY)
    right_index = jacobian_body_names.index(RIGHT_FOOT_BODY)
    left_jacobian = jacobians[
        :, left_index, :, :
    ][:, :, coordinate_index]
    right_jacobian = jacobians[
        :, right_index, :, :
    ][:, :, coordinate_index]

    def tensor(value) -> torch.Tensor:
        return torch.as_tensor(value, device=device, dtype=dtype)

    return (
        tensor(q_numpy),
        tensor(qd_numpy),
        tensor(mass_fabric),
        tensor(bias_fabric),
        tensor(left_jacobian),
        tensor(right_jacobian),
    )


def initialize_isaac_dynamics(
    q: torch.Tensor,
    qd: torch.Tensor,
    headless: bool,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[str],
    object,
    object,
    object,
    object,
]:
    """Read M, h, and both foot Jacobians at exactly the supplied state."""
    if not G1_USD_PATH.is_file():
        raise FileNotFoundError(G1_USD_PATH)

    # SimulationApp must be constructed before importing other Isaac modules.
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": headless})
    completed = False
    try:
        from isaacsim.core.api import World
        from isaacsim.core.api.objects import GroundPlane
        from isaacsim.core.prims import Articulation, RigidPrim
        from isaacsim.core.utils.stage import add_reference_to_stage
        from isaacsim.storage.native import get_assets_root_path
        import omni.usd
        from pxr import UsdLux

        if World.instance():
            World.instance().clear_instance()
        world = World(
            physics_dt=PD_TIMESTEP,
            rendering_dt=FABRIC_TIMESTEP,
        )
        world.scene.add(
            GroundPlane(
                prim_path="/World/GroundPlane",
                name="ground_plane",
                size=20.0,
                z_position=0.0,
            )
        )
        add_reference_to_stage(str(G1_USD_PATH), "/World/G1")

        # Match unitreerobotics/unitree_rl_lab's G1-29DOF scene:
        # a 750-intensity dome light with the Kloofendal clear-sky HDRI.
        stage = omni.usd.get_context().get_stage()
        sky_light = UsdLux.DomeLight.Define(stage, "/World/skyLight")
        sky_light.CreateIntensityAttr(750.0)
        assets_root = get_assets_root_path(skip_check=True)
        sky_light.CreateTextureFileAttr(
            f"{assets_root}/Isaac/Materials/Textures/Skies/PolyHaven/"
            "kloofendal_43d_clear_puresky_4k.hdr"
        )

        robot = world.scene.add(
            Articulation(prim_paths_expr="/World/G1", name="g1")
        )
        foot_paths = {}
        for prim in stage.Traverse():
            name = prim.GetName()
            if name in {LEFT_FOOT_BODY, RIGHT_FOOT_BODY}:
                foot_paths[name] = str(prim.GetPath())
        missing_feet = {
            LEFT_FOOT_BODY,
            RIGHT_FOOT_BODY,
        } - set(foot_paths)
        if missing_feet:
            raise RuntimeError(
                f"Cannot create foot contact view; missing prims: "
                f"{sorted(missing_feet)}"
            )
        foot_contact_view = world.scene.add(
            RigidPrim(
                prim_paths_expr=[
                    foot_paths[LEFT_FOOT_BODY],
                    foot_paths[RIGHT_FOOT_BODY],
                ],
                name="g1_foot_contacts",
                track_contact_forces=True,
                prepare_contact_sensors=True,
                # Isaac Sim 5.1 requires one filter-pattern group per sensor
                # pattern. There are two sensor patterns (left/right foot);
                # the default [] is normalized to only one group and makes
                # PhysX reject the contact view.
                contact_filter_prim_paths_expr=[[], []],
            )
        )
        world.reset()

        q_np = q.detach().cpu().numpy()
        qd_np = qd.detach().cpu().numpy()
        isaac_joint_names = list(robot.dof_names)
        missing = sorted(set(G1_JOINT_NAMES) - set(isaac_joint_names))
        if missing:
            raise RuntimeError(f"Isaac articulation is missing joints: {missing}")
        fabric_to_isaac_joint = np.asarray(
            [G1_JOINT_NAMES.index(name) for name in isaac_joint_names],
            dtype=np.int64,
        )

        robot.set_world_poses(
            positions=q_np[:, :3],
            orientations=q_np[:, [6, 3, 4, 5]],  # xyzw -> wxyz
        )
        robot.set_velocities(qd_np[:, :6])
        robot.set_joint_positions(
            q_np[:, 7:36][:, fabric_to_isaac_joint]
        )
        robot.set_joint_velocities(
            qd_np[:, 6:35][:, fabric_to_isaac_joint]
        )

        mass_matrix, bias_force = read_isaac_dynamics(robot)
        mass_isaac = _to_numpy(mass_matrix).copy()
        bias_isaac = _to_numpy(bias_force).copy()
        jacobians = _to_numpy(robot.get_jacobians()).copy()
        body_names = list(robot.body_names)

        expected_mass_shape = (1, 35, 35)
        if mass_isaac.shape != expected_mass_shape:
            raise RuntimeError(
                f"Expected floating-base mass matrix {expected_mass_shape}, "
                f"got {mass_isaac.shape}."
            )
        if bias_isaac.shape != (1, 35):
            raise RuntimeError(
                "Expected floating-base bias force [1, 35], got "
                f"{bias_isaac.shape}."
            )
        if jacobians.shape[0] != 1 or jacobians.shape[-2:] != (6, 35):
            raise RuntimeError(
                "Expected Isaac Jacobians [1, bodies, 6, 35], got "
                f"{jacobians.shape}."
            )

        # A floating-base articulation contains one Jacobian per body. Keep a
        # defensive fixed-base branch so an accidentally fixed USD fails by
        # body name rather than silently selecting the wrong row.
        if jacobians.shape[1] == len(body_names):
            jacobian_body_names = body_names
        elif jacobians.shape[1] == len(body_names) - 1:
            jacobian_body_names = body_names[1:]
        else:
            raise RuntimeError(
                f"Cannot align {jacobians.shape[1]} Jacobian rows with "
                f"{len(body_names)} Isaac body names."
            )
        try:
            left_index = jacobian_body_names.index(LEFT_FOOT_BODY)
            right_index = jacobian_body_names.index(RIGHT_FOOT_BODY)
        except ValueError as exc:
            raise RuntimeError(
                f"Foot body missing; Isaac bodies are {body_names}"
            ) from exc

        # Isaac generalized order is [base6, Isaac DOFs]. Reorder both axes of
        # M and the columns of J into [base6, G1_JOINT_NAMES].
        isaac_joint_index = {
            name: index for index, name in enumerate(isaac_joint_names)
        }
        coordinate_index = np.asarray(
            list(range(6))
            + [
                6 + isaac_joint_index[name]
                for name in G1_JOINT_NAMES
            ],
            dtype=np.int64,
        )
        mass_fabric = mass_isaac[
            :, coordinate_index[:, None], coordinate_index[None, :]
        ]
        bias_fabric = bias_isaac[:, coordinate_index]
        left_jacobian = jacobians[
            :, left_index, :, :
        ][:, :, coordinate_index]
        right_jacobian = jacobians[
            :, right_index, :, :
        ][:, :, coordinate_index]
        result = (
            mass_fabric,
            bias_fabric,
            left_jacobian,
            right_jacobian,
            isaac_joint_names,
            simulation_app,
            world,
            robot,
            foot_contact_view,
        )
        completed = True
        return result
    except BaseException:
        # Isaac Sim 5.1 shutdown can terminate the process before Python gets
        # a chance to display the uncaught exception. Print it first so a
        # failed dynamics query remains diagnosable.
        traceback.print_exc()
        sys.stderr.flush()
        raise
    finally:
        # Isaac Sim 5.1 fast shutdown can terminate the Python process. Keep
        # the app alive after a successful query and close it only after the
        # projection results have been printed and saved.
        if not completed:
            simulation_app.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the two G1 Fabric tasks and directly project their 35-D "
            "C-space acceleration through foot-contact constraints."
        )
    )
    parser.add_argument("--motion-pkl", type=Path, default=None)
    parser.add_argument(
        "--motion-frame",
        type=int,
        default=-1,
        help="-1 selects the middle frame.",
    )
    parser.add_argument(
        "--contact-mode",
        choices=("flight", "left", "right", "double"),
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--regularization",
        type=float,
        default=1.0e-6,
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Projection/Fabric device: auto, cpu, cuda, or cuda:0.",
    )
    parser.add_argument(
        "--show-isaac-window",
        action="store_true",
        help="Show Isaac while the PD control loop runs.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="PD/physics simulation duration in seconds (default: 10).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.contact_mode is not None:
        print(
            "warning: --contact-mode is deprecated and ignored; "
            "contact mode is detected from Isaac Sim foot forces."
        )
    if args.regularization < 0.0:
        raise ValueError("--regularization must be nonnegative.")
    if args.duration <= 0.0:
        raise ValueError("--duration must be positive.")
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)

    wp.init()
    fabric = G1PoseFabric(
        batch_size=1,
        device=str(device),
        timestep=FABRIC_TIMESTEP,
        graph_capturable=False,
    )
    motion_trajectory = None
    motion_fps = None
    motion_start_frame = 0
    if args.motion_pkl is None:
        q = torch.zeros((1, 36), dtype=torch.float32, device=device)
        q[:, 2] = 0.8
        q[:, 6] = 1.0
    else:
        motion_trajectory_cpu, motion_fps = load_motion_trajectory(
            args.motion_pkl
        )
        motion_start_frame = (
            motion_trajectory_cpu.shape[0] // 2
            if args.motion_frame == -1
            else args.motion_frame
        )
        if not 0 <= motion_start_frame < motion_trajectory_cpu.shape[0]:
            raise IndexError(
                f"Motion frame {motion_start_frame} is outside "
                f"[0, {motion_trajectory_cpu.shape[0] - 1}]"
            )
        motion_trajectory = motion_trajectory_cpu.to(device)
        q = motion_trajectory[
            motion_start_frame:motion_start_frame + 1
        ].clone()
        print(
            f"loaded motion: {args.motion_pkl} "
            f"(frames={motion_trajectory.shape[0]}, fps={motion_fps:g}, "
            f"start={motion_start_frame})"
        )
    qd = torch.zeros((1, 35), dtype=torch.float32, device=device)

    configure_two_task_targets(fabric, q)
    qdd_fabric = fabric.compute_acceleration(q, qd)

    (
        mass_np,
        bias_np,
        left_j_np,
        right_j_np,
        isaac_joint_names,
        simulation_app,
        world,
        robot,
        foot_contact_view,
    ) = (
        initialize_isaac_dynamics(
            q,
            qd,
            headless=not args.show_isaac_window,
        )
    )
    mass = torch.as_tensor(mass_np, device=device, dtype=q.dtype)
    bias_force = torch.as_tensor(bias_np, device=device, dtype=q.dtype)
    left_j = torch.as_tensor(left_j_np, device=device, dtype=q.dtype)
    right_j = torch.as_tensor(right_j_np, device=device, dtype=q.dtype)
    projector = G1FabricContactProjector(
        num_envs=1,
        device=device,
        dtype=q.dtype,
        dt=FABRIC_TIMESTEP,
        detector_config=ContactDetectorConfig(
            force_on=20.0,
            force_off=10.0,
            confirm_steps=1,
        ),
        projection_config=ProjectionConfig(
            regularization=args.regularization,
        ),
    )
    foot_normal_forces = torch.zeros((1, 2), device=device, dtype=q.dtype)
    # At this initial test state qd=0, hence Jdot*qdot=0 exactly. Supplying
    # zeros avoids treating the estimator's intentionally-zero first sample as
    # an approximation.
    zero_bias = torch.zeros((1, 6), device=device, dtype=q.dtype)
    output = projector.step(
        acc_fabric=qdd_fabric,
        mass_matrix=mass,
        generalized_velocity=qd,
        left_foot_jacobian=left_j,
        right_foot_jacobian=right_j,
        foot_normal_forces=foot_normal_forces,
        left_jdot_v=zero_bias,
        right_jdot_v=zero_bias,
    )
    qdd_dc = output.acc_dc
    contact_jacobian = torch.cat((left_j, right_j), dim=1)
    jdot_qdot = torch.cat((zero_bias, zero_bias), dim=1)
    acceleration_qp = G1DynamicsAccelerationQP(
        max_iter=500,
        validation_tolerance=1.0e-3,
        friction_validation_tolerance=0.25,
    )
    qp_output = acceleration_qp.solve(
        acc_reference=qdd_dc,
        mass_matrix=mass,
        bias_force=bias_force,
        contact_jacobian=contact_jacobian,
        jdot_qdot=jdot_qdot,
        contact_mode=output.contact_mode,
    )
    if not bool(qp_output.validated.all()):
        raise RuntimeError(
            "Acceleration QP failed validation: "
            f"converged={qp_output.converged.tolist()}, "
            f"base_error={qp_output.base_dynamics_error.tolist()}, "
            "contact_error="
            f"{qp_output.contact_acceleration_error.tolist()}, "
            f"friction_violation={qp_output.friction_violation.tolist()}."
        )
    qdd_dynamics_feasible = qp_output.acc
    # The floating base is unactuated. Integrate only the final 29 actuated
    # coordinates instead of constructing an unused desired root state.
    joint_velocity = qd[:, 6:35]
    joint_velocity_desired = (
        joint_velocity
        + FABRIC_TIMESTEP * qdd_dynamics_feasible[:, 6:35]
    )
    joint_position = q[:, 7:36]
    joint_position_desired = (
        joint_position
        + FABRIC_TIMESTEP * joint_velocity_desired
    )

    # Apply joint-space PD to the 29 G1 actuators, then enforce effort limits.
    kp, kd = unitree_g1_pd_gains(device)
    effort_limits = torch.as_tensor(
        unitree_g1_effort_limits(),
        device=device,
        dtype=q.dtype,
    )
    joint_torque_unclipped = (
        kp * (joint_position_desired - joint_position)
        + kd * (joint_velocity_desired - joint_velocity)
    )
    joint_torque = torch.clamp(
        joint_torque_unclipped,
        min=-effort_limits,
        max=effort_limits,
    )

    # Send the 29-D effort command in Isaac DOF order. Physics and joint PD run
    # at 500 Hz; the complete state/dynamics/FABRICS/QP calculation is refreshed
    # every ten physics steps (50 Hz).
    fabric_to_isaac_joint = torch.tensor(
        [G1_JOINT_NAMES.index(name) for name in isaac_joint_names],
        device=device,
        dtype=torch.long,
    )
    position_desired_isaac = joint_position_desired[
        :, fabric_to_isaac_joint
    ]
    velocity_desired_isaac = joint_velocity_desired[
        :, fabric_to_isaac_joint
    ]
    kp_isaac = kp[fabric_to_isaac_joint]
    kd_isaac = kd[fabric_to_isaac_joint]
    effort_limits_isaac = effort_limits[fabric_to_isaac_joint]
    physics_steps = int(round(args.duration / PD_TIMESTEP))
    joint_torque_isaac_tensor = torch.zeros(
        (1, len(G1_JOINT_NAMES)),
        device=device,
        dtype=q.dtype,
    )
    fabric_steps_per_update = int(round(FABRIC_TIMESTEP / PD_TIMESTEP))
    if fabric_steps_per_update < 1:
        raise RuntimeError("FABRIC_TIMESTEP must be at least PD_TIMESTEP.")
    executed_steps = 0
    control_updates = 0
    reported_nonconvergence = False
    for step in range(physics_steps):
        if not simulation_app.is_running():
            break

        if step % fabric_steps_per_update == 0:
            (
                q,
                qd,
                mass,
                bias_force,
                left_j,
                right_j,
            ) = read_live_isaac_snapshot(
                robot,
                isaac_joint_names,
                device,
                q.dtype,
            )

            if motion_trajectory is not None:
                elapsed_motion_time = control_updates * FABRIC_TIMESTEP
                motion_frame = min(
                    motion_start_frame
                    + int(round(elapsed_motion_time * motion_fps)),
                    motion_trajectory.shape[0] - 1,
                )
                reference_q = motion_trajectory[
                    motion_frame:motion_frame + 1
                ]
                configure_two_task_targets(fabric, reference_q)

            net_foot_forces = torch.as_tensor(
                _to_numpy(
                    foot_contact_view.get_net_contact_forces(
                        dt=PD_TIMESTEP
                    )
                ),
                device=device,
                dtype=q.dtype,
            )
            if net_foot_forces.shape != (2, 3):
                raise RuntimeError(
                    "Expected left/right net foot forces [2,3], got "
                    f"{tuple(net_foot_forces.shape)}."
                )
            # Ground normal is world +Z. Magnitude makes the detector robust
            # to the API's body-on-environment force sign convention.
            foot_normal_forces = (
                net_foot_forces[:, 2].abs().reshape(1, 2)
            )

            qdd_fabric = fabric.compute_acceleration(q, qd)
            output = projector.step(
                acc_fabric=qdd_fabric,
                mass_matrix=mass,
                generalized_velocity=qd,
                left_foot_jacobian=left_j,
                right_foot_jacobian=right_j,
                foot_normal_forces=foot_normal_forces,
            )
            qdd_dc = output.acc_dc
            contact_jacobian = torch.cat((left_j, right_j), dim=1)
            # The projector's finite-difference estimator and the QP must use
            # the same Jdot*v value. Recover it from the reported residual:
            # residual_before = J*a_fabric + Jdot*v (no stabilization here).
            jdot_qdot = (
                output.contact_residual_before
                - torch.bmm(
                    contact_jacobian * output.active_row_mask.unsqueeze(-1),
                    qdd_fabric.unsqueeze(-1),
                ).squeeze(-1)
            )
            qp_output = acceleration_qp.solve(
                acc_reference=qdd_dc,
                mass_matrix=mass,
                bias_force=bias_force,
                contact_jacobian=contact_jacobian,
                jdot_qdot=jdot_qdot,
                contact_mode=output.contact_mode,
            )
            if not bool(qp_output.validated.all()):
                raise RuntimeError(
                    "50 Hz acceleration QP failed validation: "
                    f"converged={qp_output.converged.tolist()}, "
                    f"base_error={qp_output.base_dynamics_error.tolist()}, "
                    "contact_error="
                    f"{qp_output.contact_acceleration_error.tolist()}, "
                    "friction_violation="
                    f"{qp_output.friction_violation.tolist()}."
                )
            if (
                not bool(qp_output.converged.all())
                and not reported_nonconvergence
            ):
                print(
                    "warning: ADMM reached a physically validated solution "
                    "without satisfying its stricter numerical convergence "
                    f"flag after {qp_output.iters} iterations."
                )
                reported_nonconvergence = True
            qdd_dynamics_feasible = qp_output.acc

            joint_velocity = qd[:, 6:35]
            joint_velocity_desired = (
                joint_velocity
                + FABRIC_TIMESTEP * qdd_dynamics_feasible[:, 6:35]
            )
            joint_position = q[:, 7:36]
            joint_position_desired = (
                joint_position
                + FABRIC_TIMESTEP * joint_velocity_desired
            )
            position_desired_isaac = joint_position_desired[
                :, fabric_to_isaac_joint
            ]
            velocity_desired_isaac = joint_velocity_desired[
                :, fabric_to_isaac_joint
            ]
            joint_torque_unclipped = (
                kp * (joint_position_desired - joint_position)
                + kd * (joint_velocity_desired - joint_velocity)
            )
            joint_torque = torch.clamp(
                joint_torque_unclipped,
                min=-effort_limits,
                max=effort_limits,
            )
            control_updates += 1

        actual_position_isaac = torch.as_tensor(
            _to_numpy(robot.get_joint_positions()),
            device=device,
            dtype=q.dtype,
        )
        actual_velocity_isaac = torch.as_tensor(
            _to_numpy(robot.get_joint_velocities()),
            device=device,
            dtype=q.dtype,
        )
        joint_torque_isaac_tensor = (
            kp_isaac * (position_desired_isaac - actual_position_isaac)
            + kd_isaac * (
                velocity_desired_isaac - actual_velocity_isaac
            )
        )
        joint_torque_isaac_tensor = torch.clamp(
            joint_torque_isaac_tensor,
            min=-effort_limits_isaac,
            max=effort_limits_isaac,
        )
        robot.set_joint_efforts(
            joint_torque_isaac_tensor.detach().cpu().numpy()
        )
        world.step(render=args.show_isaac_window)
        executed_steps = step + 1
    joint_torque_isaac = (
        joint_torque_isaac_tensor.detach().cpu().numpy()
    )
    joint_position_after_control = torch.as_tensor(
        _to_numpy(robot.get_joint_positions()),
        device=device,
        dtype=q.dtype,
    )
    joint_velocity_after_control = torch.as_tensor(
        _to_numpy(robot.get_joint_velocities()),
        device=device,
        dtype=q.dtype,
    )

    active = output.active_row_mask.bool()
    before_max = (
        output.contact_residual_before[active].abs().max().item()
        if active.any()
        else 0.0
    )
    after_max = (
        output.contact_residual_after[active].abs().max().item()
        if active.any()
        else 0.0
    )
    values = {
        "qdd_fabric": (qdd_fabric, (1, 35)),
        "mass_matrix": (mass, (1, 35, 35)),
        "bias_force": (bias_force, (1, 35)),
        "left_foot_jacobian": (left_j, (1, 6, 35)),
        "right_foot_jacobian": (right_j, (1, 6, 35)),
        "qdd_contact_consistent": (qdd_dc, (1, 35)),
        "qdd_dynamics_feasible": (
            qdd_dynamics_feasible,
            (1, 35),
        ),
        "joint_velocity_desired": (
            joint_velocity_desired,
            (1, 29),
        ),
        "joint_position_desired": (
            joint_position_desired,
            (1, 29),
        ),
        "joint_torque": (joint_torque, (1, 29)),
        "joint_position_after_control": (
            joint_position_after_control,
            (1, 29),
        ),
        "joint_velocity_after_control": (
            joint_velocity_after_control,
            (1, 29),
        ),
    }
    for name, (value, shape) in values.items():
        assert tuple(value.shape) == shape
        assert torch.isfinite(value).all(), f"{name} contains NaN/Inf"
        print(f"{name:27s} shape={tuple(value.shape)}")

    print(f"device={device}")
    print(
        "contact_mode="
        f"{int(output.contact_mode.item())} "
        f"(left_fz={foot_normal_forces[0, 0].item():.3f} N, "
        f"right_fz={foot_normal_forces[0, 1].item():.3f} N)"
    )
    print(
        f"PD simulation={executed_steps}/{physics_steps} steps, "
        f"50 Hz control updates={control_updates}, "
        f"requested duration={args.duration:.3f}s, "
        f"dt={PD_TIMESTEP:.4f}s"
    )
    print(f"|qdd_fabric|={torch.linalg.vector_norm(qdd_fabric).item():.6f}")
    print(f"|qdd_dc|={torch.linalg.vector_norm(qdd_dc).item():.6f}")
    print(
        "|qdd_dynamics_feasible|="
        f"{torch.linalg.vector_norm(qdd_dynamics_feasible).item():.6f}"
    )
    print(
        "Acceleration QP: "
        f"iters={qp_output.iters}, "
        f"base_error={qp_output.base_dynamics_error.max().item():.6e}, "
        "contact_error="
        f"{qp_output.contact_acceleration_error.max().item():.6e}, "
        "friction_violation="
        f"{qp_output.friction_violation.max().item():.6e}"
    )
    print(
        "|joint_torque|="
        f"{torch.linalg.vector_norm(joint_torque).item():.6f} Nm"
    )
    print(
        "max |joint_torque|="
        f"{joint_torque.abs().max().item():.6f} Nm"
    )
    print(f"max active residual before={before_max:.6e}")
    print(f"max active residual after ={after_max:.6e}")
    print(f"Isaac DOF order={isaac_joint_names}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT_PATH,
        q=q.detach().cpu().numpy(),
        qd=qd.detach().cpu().numpy(),
        qdd_fabric=qdd_fabric.detach().cpu().numpy(),
        qdd_contact_consistent=qdd_dc.detach().cpu().numpy(),
        qdd_dynamics_feasible=(
            qdd_dynamics_feasible.detach().cpu().numpy()
        ),
        contact_wrench=qp_output.contact_wrench.detach().cpu().numpy(),
        qp_converged=qp_output.converged.detach().cpu().numpy(),
        qp_validated=qp_output.validated.detach().cpu().numpy(),
        qp_objective=qp_output.objective.detach().cpu().numpy(),
        qp_base_dynamics_error=(
            qp_output.base_dynamics_error.detach().cpu().numpy()
        ),
        qp_contact_acceleration_error=(
            qp_output.contact_acceleration_error.detach().cpu().numpy()
        ),
        qp_friction_violation=(
            qp_output.friction_violation.detach().cpu().numpy()
        ),
        joint_position_desired=(
            joint_position_desired.detach().cpu().numpy()
        ),
        joint_velocity_desired=(
            joint_velocity_desired.detach().cpu().numpy()
        ),
        joint_torque_unclipped=(
            joint_torque_unclipped.detach().cpu().numpy()
        ),
        joint_torque=joint_torque.detach().cpu().numpy(),
        kp=kp.detach().cpu().numpy(),
        kd=kd.detach().cpu().numpy(),
        effort_limits=effort_limits.detach().cpu().numpy(),
        fabric_dt=np.asarray(FABRIC_TIMESTEP, dtype=np.float32),
        pd_dt=np.asarray(PD_TIMESTEP, dtype=np.float32),
        joint_torque_isaac=joint_torque_isaac,
        joint_position_after_control=(
            joint_position_after_control.detach().cpu().numpy()
        ),
        joint_velocity_after_control=(
            joint_velocity_after_control.detach().cpu().numpy()
        ),
        mass_matrix=mass.detach().cpu().numpy(),
        bias_force=bias_force.detach().cpu().numpy(),
        left_foot_jacobian=left_j.detach().cpu().numpy(),
        right_foot_jacobian=right_j.detach().cpu().numpy(),
        active_row_mask=output.active_row_mask.detach().cpu().numpy(),
        contact_mode=output.contact_mode.detach().cpu().numpy(),
        foot_normal_forces=foot_normal_forces.detach().cpu().numpy(),
    )
    print(f"saved: {OUTPUT_PATH}")
    print(
        "[pass] two-task Fabric -> contact projection -> "
        "dynamics acceleration QP -> integration"
    )
    simulation_app.close()


if __name__ == "__main__":
    main()
