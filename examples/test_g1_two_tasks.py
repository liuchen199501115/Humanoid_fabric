"""Smoke-test the pelvis and four-end-effector G1 Fabric tasks."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
import warp as wp

from fabrics_sim.fabrics.g1_pose_fabric import G1PoseFabric


FABRIC_FREQUENCY = 50.0
FABRIC_TIMESTEP = 1.0 / FABRIC_FREQUENCY
PD_FREQUENCY = 500.0
PD_TIMESTEP = 1.0 / PD_FREQUENCY
PD_STEPS_PER_FABRIC_STEP = int(PD_FREQUENCY / FABRIC_FREQUENCY)
OUTPUT_PATH = (
    Path(__file__).resolve().parents[1]
    / "acc_data/g1_two_task_pd_step.npz"
)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
G1_USD_PATH = (
    PROJECT_ROOT
    / "src/fabrics_sim/models/robots/USD/g1_sim/29dof/usd"
    / "g1_29dof_rev_1_0/g1_29dof_rev_1_0.usd"
)

G1_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)


def unitree_g1_pd_gains(
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """UNITREE_G1_29DOF_CFG stiffness/damping in G1 joint order."""
    kp = []
    kd = []
    for name in G1_JOINT_NAMES:
        if "_knee_joint" in name:
            gains = (150.0, 4.0)
        elif "_hip_" in name:
            gains = (100.0, 2.0)
        elif "_ankle_" in name:
            gains = (40.0, 2.0)
        elif name == "waist_yaw_joint":
            gains = (200.0, 5.0)
        elif name in {"waist_roll_joint", "waist_pitch_joint"}:
            gains = (40.0, 5.0)
        else:
            gains = (40.0, 1.0)
        kp.append(gains[0])
        kd.append(gains[1])
    return (
        torch.tensor(kp, dtype=torch.float32, device=device),
        torch.tensor(kd, dtype=torch.float32, device=device),
    )


def unitree_g1_effort_limits() -> np.ndarray:
    """UNITREE_G1_29DOF_CFG simulated effort limits in G1 joint order."""
    limits = []
    for name in G1_JOINT_NAMES:
        if "_hip_roll_joint" in name or "_knee_joint" in name:
            limit = 139.0
        elif "_hip_pitch_joint" in name or "_hip_yaw_joint" in name:
            limit = 88.0
        elif name == "waist_yaw_joint":
            limit = 88.0
        elif "_wrist_pitch_joint" in name or "_wrist_yaw_joint" in name:
            limit = 5.0
        else:
            limit = 25.0
        limits.append(limit)
    return np.asarray(limits, dtype=np.float32)


def as_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def load_motion_frame(
    motion_path: Path,
    frame: int,
) -> tuple[torch.Tensor, int]:
    """Load one floating-base G1 configuration from a retargeted motion PKL."""
    trajectory, _ = load_motion_trajectory(motion_path)
    frame = trajectory.shape[0] // 2 if frame == -1 else frame
    if not 0 <= frame < trajectory.shape[0]:
        raise IndexError(
            f"Motion frame {frame} is outside [0, {trajectory.shape[0] - 1}]"
        )
    return trajectory[frame:frame + 1], frame


def load_motion_trajectory(
    motion_path: Path,
) -> tuple[torch.Tensor, float]:
    """Load the complete floating-base G1 trajectory and its sample rate."""
    if not motion_path.is_file():
        raise FileNotFoundError(f"Motion PKL does not exist: {motion_path}")
    with motion_path.open("rb") as motion_file:
        motion = pickle.load(motion_file)

    required = {"root_pos", "root_rot", "dof_pos"}
    missing = sorted(required - set(motion))
    if missing:
        raise KeyError(f"Motion PKL is missing keys: {missing}")

    root_pos = as_numpy(motion["root_pos"])
    root_rot = as_numpy(motion["root_rot"])
    dof_pos = as_numpy(motion["dof_pos"])
    if (
        root_pos.ndim != 2
        or root_pos.shape[1] != 3
        or root_rot.shape != (root_pos.shape[0], 4)
        or dof_pos.shape != (root_pos.shape[0], len(G1_JOINT_NAMES))
    ):
        raise ValueError(
            "Expected root_pos (T,3), root_rot (T,4), and "
            f"dof_pos (T,{len(G1_JOINT_NAMES)}); got "
            f"{root_pos.shape}, {root_rot.shape}, {dof_pos.shape}"
        )

    dof_names = motion.get("dof_names")
    if dof_names is not None:
        dof_names = [str(name) for name in dof_names]
        if len(dof_names) != dof_pos.shape[1]:
            raise ValueError(
                "dof_names and dof_pos disagree: "
                f"{len(dof_names)} names for {dof_pos.shape[1]} columns"
            )
        missing_joints = sorted(set(G1_JOINT_NAMES) - set(dof_names))
        if missing_joints:
            raise KeyError(f"Motion PKL is missing G1 joints: {missing_joints}")
        indices = [dof_names.index(name) for name in G1_JOINT_NAMES]
        dof_pos = dof_pos[:, indices]
    else:
        dof_pos = dof_pos

    q = np.concatenate(
        (root_pos, root_rot, dof_pos),
        axis=1,
    ).astype(np.float32, copy=False)
    if not np.isfinite(q).all():
        raise ValueError("Motion trajectory contains NaN/Inf")
    quaternion_norm = np.linalg.norm(q[:, 3:7], axis=1, keepdims=True)
    if np.any(quaternion_norm < 1e-8):
        raise ValueError("Motion trajectory contains a zero root quaternion")
    q[:, 3:7] /= quaternion_norm

    fps = float(motion.get("fps", 30.0))
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"Motion fps must be positive, got {fps}")
    return torch.from_numpy(q), fps


def visualize_isaacsim(
    fabric: G1PoseFabric,
    q_desired: torch.Tensor,
    qd_desired: torch.Tensor,
    kp: torch.Tensor,
    kd: torch.Tensor,
    duration: float,
    headless: bool,
) -> None:
    """Run 50 Hz Fabric and 500 Hz explicit joint PD in Isaac Sim."""
    if not G1_USD_PATH.is_file():
        raise FileNotFoundError(G1_USD_PATH)

    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": headless})

    # Isaac Sim 5.1 API.
    from isaacsim.core.api import World
    from isaacsim.core.api.objects import GroundPlane
    from isaacsim.core.prims import Articulation
    from isaacsim.core.utils.stage import add_reference_to_stage
    from isaacsim.core.utils.viewports import set_camera_view
    from isaacsim.storage.native import get_assets_root_path
    import omni.usd
    from pxr import UsdLux

    if World.instance():
        World.instance().clear_instance()
    world = World(
        physics_dt=PD_TIMESTEP,
        rendering_dt=1.0 / 60.0,
    )
    # Create the plane locally; add_default_ground_plane() may query a remote
    # Isaac asset server and block on machines without Nucleus connectivity.
    world.scene.add(
        GroundPlane(
            prim_path="/World/GroundPlane",
            name="ground_plane",
            size=20.0,
            z_position=0.0,
        )
    )
    add_reference_to_stage(str(G1_USD_PATH), "/World/G1")
    robot = world.scene.add(
        Articulation(prim_paths_expr="/World/G1", name="g1")
    )

    # Match unitreerobotics/unitree_rl_lab's G1 scene lighting: one
    # 750-intensity dome light using the Kloofendal clear-sky HDRI.
    stage = omni.usd.get_context().get_stage()
    sky_light = UsdLux.DomeLight.Define(
        stage,
        "/World/skyLight",
    )
    sky_light.CreateIntensityAttr(750.0)
    assets_root = get_assets_root_path(skip_check=True)
    sky_light.CreateTextureFileAttr(
        f"{assets_root}/Isaac/Materials/Textures/Skies/PolyHaven/"
        "kloofendal_43d_clear_puresky_4k.hdr"
    )

    world.reset()
    set_camera_view(
        eye=np.asarray([3.0, 3.0, 2.2]),
        target=np.asarray([0.0, 0.0, 1.0]),
        camera_prim_path="/OmniverseKit_Persp",
    )

    isaac_joint_names = list(robot.dof_names)
    missing = sorted(set(isaac_joint_names) - set(G1_JOINT_NAMES))
    if missing:
        raise RuntimeError(f"Unmapped Isaac joints: {missing}")
    reference_indices = np.asarray(
        [G1_JOINT_NAMES.index(name) for name in isaac_joint_names],
        dtype=np.int64,
    )

    kp_isaac = kp.detach().cpu().numpy()[reference_indices]
    kd_isaac = kd.detach().cpu().numpy()[reference_indices]
    effort_limits = unitree_g1_effort_limits()[reference_indices]
    zeros = np.zeros((1, len(isaac_joint_names)), dtype=np.float32)

    # Disable the USD's implicit drives: torque comes only from explicit PD.
    robot.set_gains(kps=zeros, kds=zeros)
    initial_q = q_desired.detach().cpu().numpy()[0]
    robot.set_world_poses(
        positions=initial_q[None, :3],
        orientations=initial_q[None, [6, 3, 4, 5]],  # xyzw -> wxyz
    )
    robot.set_joint_positions(
        initial_q[None, 7:36][:, reference_indices]
    )
    robot.set_joint_velocities(zeros)
    world.play()

    render_stride = max(
        int(round((1.0 / 60.0) / PD_TIMESTEP)),
        1,
    )
    steps = int(round(duration * PD_FREQUENCY))
    print(
        f"[Isaac Sim] Fabric={FABRIC_FREQUENCY:.0f} Hz, "
        f"PD/physics={PD_FREQUENCY:.0f} Hz, "
        f"duration={duration:.2f}s, floating-base control=False"
    )

    try:
        with torch.no_grad():
            for step in range(steps):
                if not simulation_app.is_running():
                    break

                # Update Fabric and its integrated desired state once every
                # ten 500 Hz PD/physics steps.
                if (
                    step > 0
                    and step % PD_STEPS_PER_FABRIC_STEP == 0
                ):
                    qdd_desired = fabric.compute_acceleration(
                        q_desired,
                        qd_desired,
                    )
                    qd_desired = (
                        qd_desired
                        + FABRIC_TIMESTEP * qdd_desired
                    )
                    q_desired = fabric.integrate_root_position(
                        q_desired,
                        qd_desired,
                        FABRIC_TIMESTEP,
                    )

                desired_position = (
                    q_desired[0, 7:36]
                    .detach()
                    .cpu()
                    .numpy()[reference_indices]
                )
                desired_velocity = (
                    qd_desired[0, 6:35]
                    .detach()
                    .cpu()
                    .numpy()[reference_indices]
                )
                actual_position = as_numpy(
                    robot.get_joint_positions()
                )[0]
                actual_velocity = as_numpy(
                    robot.get_joint_velocities()
                )[0]

                torque = (
                    kp_isaac * (desired_position - actual_position)
                    + kd_isaac * (desired_velocity - actual_velocity)
                )
                torque = np.clip(
                    torque,
                    -effort_limits,
                    effort_limits,
                )
                robot.set_joint_efforts(torque[None, :])
                world.step(render=(step % render_stride == 0))

                if step % int(PD_FREQUENCY) == 0:
                    error = np.sqrt(
                        np.mean(
                            (desired_position - actual_position) ** 2
                        )
                    )
                    print(
                        f"t={step / PD_FREQUENCY:.2f}s "
                        f"joint_rmse={error:.5f} rad "
                        f"max_torque={np.max(np.abs(torque)):.2f} Nm"
                    )
    finally:
        world.stop()
        simulation_app.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="launch Isaac Sim and apply 500 Hz joint PD torque",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Isaac Sim visualization duration in seconds",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="run Isaac Sim without a window (for validation)",
    )
    parser.add_argument(
        "--motion-pkl",
        type=Path,
        default=None,
        help="initialize G1 from a retargeted B2A motion PKL",
    )
    parser.add_argument(
        "--motion-frame",
        type=int,
        default=-1,
        help="motion frame to use; -1 selects the middle frame",
    )
    args = parser.parse_args()
    if args.duration <= 0.0:
        raise ValueError("--duration must be positive.")

    wp.init()
    device = torch.device("cpu")
    fabric = G1PoseFabric(
        batch_size=1,
        device=str(device),
        timestep=FABRIC_TIMESTEP,
        graph_capturable=False,
    )

    if args.motion_pkl is None:
        q = torch.zeros((1, 36), dtype=torch.float32)
        q[:, 2] = 0.8
        q[:, 6] = 1.0  # base quaternion xyzw
    else:
        q, selected_frame = load_motion_frame(
            args.motion_pkl,
            args.motion_frame,
        )
        print(
            f"loaded motion: {args.motion_pkl} "
            f"(frame={selected_frame}, root_z={q[0, 2].item():.4f} m)"
        )
    qd = torch.zeros((1, 35), dtype=torch.float32)

    # Run FK once and use the current poses as consistent reference poses.
    fabric.end_effector_pose_taskmap(q, None)
    transforms = (
        fabric.end_effector_pose_taskmap
        .robot_kinematics.batch_link_transforms_torch[
            :, fabric.end_effector_pose_taskmap.link_indices
        ]
        .clone()
    )

    pelvis_position_target = q[:, :3].clone()
    pelvis_position_target[:, 0] += 0.02
    pelvis_quaternion_target = q[:, 3:7].clone()

    end_effector_position_target = transforms[..., :3].clone()
    end_effector_position_target[:, 2, 2] += 0.03  # left hand upward
    end_effector_quaternion_target = transforms[..., 3:7].clone()

    fabric.set_motion_targets(
        pelvis_position_target=pelvis_position_target,
        pelvis_quaternion_target_xyzw=pelvis_quaternion_target,
        end_effector_position_target=end_effector_position_target,
        end_effector_quaternion_target_xyzw=end_effector_quaternion_target,
    )

    pelvis_x, pelvis_j = fabric.pelvis_pose_taskmap(q, None)
    pelvis_xd = torch.bmm(pelvis_j, qd.unsqueeze(-1)).squeeze(-1)
    pelvis_m, pelvis_f = (
        fabric.taskmap_containers["pelvis_pose"]
        .get_fabric("pelvis_pose_attractor")(
            pelvis_x,
            pelvis_xd,
            fabric.fabrics_features["pelvis_pose"][
                "pelvis_pose_attractor"
            ],
        )
    )

    ee_x, ee_j = fabric.end_effector_pose_taskmap(q, None)
    ee_xd = torch.bmm(ee_j, qd.unsqueeze(-1)).squeeze(-1)
    ee_m, ee_f = (
        fabric.taskmap_containers["end_effector_pose"]
        .get_fabric("end_effector_pose_attractor")(
            ee_x,
            ee_xd,
            fabric.fabrics_features["end_effector_pose"][
                "end_effector_pose_attractor"
            ],
        )
    )

    qdd = fabric.compute_acceleration(q, qd)

    # Semi-implicit Euler integration:
    #   qdd (35) -> qd_next (35) -> q_next (36)
    # integrate_root_position updates the floating-base quaternion on SO(3).
    dt = FABRIC_TIMESTEP
    qd_next = qd + dt * qdd
    q_next = fabric.integrate_root_position(q, qd_next, dt)

    # This desired state is updated at 50 Hz. Explicit joint-space PD runs at
    # 500 Hz in Isaac Sim. Floating-base q[0:7] and qd[0:6]
    # are deliberately excluded; only the 29 actuated joints receive torque.
    kp, kd = unitree_g1_pd_gains(device)
    joint_position = q[:, 7:36]
    joint_velocity = qd[:, 6:35]
    desired_joint_position = q_next[:, 7:36]
    desired_joint_velocity = qd_next[:, 6:35]
    joint_torque = (
        kp * (desired_joint_position - joint_position)
        + kd * (desired_joint_velocity - joint_velocity)
    )

    expected_shapes = {
        "pelvis x": (1, 6),
        "pelvis J": (1, 6, 35),
        "pelvis M": (1, 6, 6),
        "pelvis f": (1, 6),
        "end-effector x": (1, 24),
        "end-effector J": (1, 24, 35),
        "end-effector M": (1, 24, 24),
        "end-effector f": (1, 24),
        "combined qdd": (1, 35),
        "integrated qd": (1, 35),
        "integrated q": (1, 36),
        "joint torque": (1, 29),
    }
    values = {
        "pelvis x": pelvis_x,
        "pelvis J": pelvis_j,
        "pelvis M": pelvis_m,
        "pelvis f": pelvis_f,
        "end-effector x": ee_x,
        "end-effector J": ee_j,
        "end-effector M": ee_m,
        "end-effector f": ee_f,
        "combined qdd": qdd,
        "integrated qd": qd_next,
        "integrated q": q_next,
        "joint torque": joint_torque,
    }
    for name, expected in expected_shapes.items():
        value = values[name]
        assert tuple(value.shape) == expected, (
            f"{name}: expected {expected}, got {tuple(value.shape)}"
        )
        assert torch.isfinite(value).all(), f"{name} contains NaN/Inf"
        print(f"{name:20s} shape={tuple(value.shape)}")

    print(f"|pelvis error|={torch.linalg.vector_norm(pelvis_x).item():.6f}")
    print(f"|end-effector error|={torch.linalg.vector_norm(ee_x).item():.6f}")
    print(f"|combined qdd|={torch.linalg.vector_norm(qdd).item():.6f}")
    print(f"integrated qd={qd_next[0].tolist()}")
    print(f"integrated q={q_next[0].tolist()}")
    print(
        "integrated quaternion norm="
        f"{torch.linalg.vector_norm(q_next[0, 3:7]).item():.8f}"
    )
    print(
        f"Fabric frequency={FABRIC_FREQUENCY:.1f} Hz, "
        f"dt={FABRIC_TIMESTEP:.6f} s"
    )
    print(
        f"PD frequency={PD_FREQUENCY:.1f} Hz, "
        f"dt={PD_TIMESTEP:.6f} s"
    )
    print(f"joint torque={joint_torque[0].tolist()}")
    print(
        "floating-base controlled="
        f"{False} (torque dimension={joint_torque.shape[-1]})"
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT_PATH,
        q_desired=q_next.detach().cpu().numpy(),
        qd_desired=qd_next.detach().cpu().numpy(),
        joint_position_desired=(
            desired_joint_position.detach().cpu().numpy()
        ),
        joint_velocity_desired=(
            desired_joint_velocity.detach().cpu().numpy()
        ),
        joint_torque=joint_torque.detach().cpu().numpy(),
        joint_names=np.asarray(G1_JOINT_NAMES),
        kp=kp.detach().cpu().numpy(),
        kd=kd.detach().cpu().numpy(),
        fabric_dt=np.asarray(FABRIC_TIMESTEP, dtype=np.float32),
        pd_dt=np.asarray(PD_TIMESTEP, dtype=np.float32),
    )
    print(f"saved PD command: {OUTPUT_PATH}")
    print("[pass] pelvis_pose + g1_end_effector_pose")

    if args.visualize:
        visualize_isaacsim(
            fabric=fabric,
            q_desired=q_next.clone(),
            qd_desired=qd_next.clone(),
            kp=kp,
            kd=kd,
            duration=args.duration,
            headless=args.headless,
        )


if __name__ == "__main__":
    main()
