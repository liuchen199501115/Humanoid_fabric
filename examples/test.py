from __future__ import annotations

import argparse
import inspect
import pickle
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import warp as wp

from fabrics_sim.fabrics.g1_pose_fabric import G1PoseFabric


NUM_JOINTS = 29
Q_DIM = 7 + NUM_JOINTS
V_DIM = 6 + NUM_JOINTS
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ACC_OUTPUT = PROJECT_ROOT / "acc_data" / "g1_fabric_acc.npz"


# Add the exact names used by your retargeting pipeline here when necessary.
ALIASES = {
    "pelvis_position": (
        "pelvis_position",
        "pelvis_pos",
        "root_position",
        "root_pos",
        "root_translation",
        "trans",
    ),
    "pelvis_quaternion": (
        "pelvis_quaternion",
        "pelvis_quat",
        "root_quaternion",
        "root_quat",
        "root_rotation",
        "root_rot",
    ),
    "joint_position": (
        "joint_position",
        "joint_positions",
        "joint_pos",
        "joint_q",
        "dof_pos",
        "qpos",
    ),
    "generalized_position": (
        "generalized_position",
        "configuration",
        "robot_q",
        "q",
    ),
    "base_linear_velocity": (
        "base_linear_velocity",
        "pelvis_linear_velocity",
        "root_linear_velocity",
        "root_lin_vel",
    ),
    "base_angular_velocity": (
        "base_angular_velocity",
        "pelvis_angular_velocity",
        "root_angular_velocity",
        "root_ang_vel",
    ),
    "joint_velocity": (
        "joint_velocity",
        "joint_velocities",
        "joint_vel",
        "joint_qd",
        "dof_vel",
        "qvel_joint",
    ),
    "generalized_velocity": (
        "generalized_velocity",
        "robot_qd",
        "nu",
        "qd",
        "qvel",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pkl",
        type=Path,
        required=True,
        help="Retargeting PKL file under outputs/ or another directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_ACC_OUTPUT,
        help="Output NPZ containing q, qd, targets, and qdd_fabric.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--motion-fps",
        type=float,
        default=30.0,
        help="Used when the PKL does not contain fps/dt.",
    )
    parser.add_argument(
        "--quat-order",
        choices=("xyzw", "wxyz"),
        default="xyzw",
        help="Quaternion order in the PKL. Fabric receives xyzw.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional short test limit.",
    )
    parser.add_argument(
        "--graph-capturable",
        action="store_true",
        help="Enable CUDA graph-compatible Fabric buffers.",
    )
    return parser.parse_args()


def load_pickle(path: Path) -> Any:
    with path.open("rb") as file:
        return pickle.load(file)


def describe(value: Any, prefix: str = "root", depth: int = 0) -> None:
    """Print a compact PKL structure to help resolve field-name mismatches."""
    indent = "  " * depth
    if isinstance(value, dict):
        print(f"{indent}{prefix}: dict({len(value)})")
        for key, child in value.items():
            if depth < 2:
                describe(child, str(key), depth + 1)
    elif isinstance(value, (list, tuple)):
        print(f"{indent}{prefix}: {type(value).__name__}(len={len(value)})")
        if value and depth < 2:
            describe(value[0], "[0]", depth + 1)
    elif hasattr(value, "shape"):
        print(
            f"{indent}{prefix}: {type(value).__name__}"
            f"(shape={tuple(value.shape)}, dtype={getattr(value, 'dtype', None)})"
        )
    else:
        print(f"{indent}{prefix}: {type(value).__name__} = {value!r}")


def unwrap_motion(data: Any) -> Any:
    """Unwrap common outer PKL containers without assuming one exact producer."""
    if not isinstance(data, dict):
        return data

    for key in ("motion", "motions", "frames", "trajectory", "data"):
        candidate = data.get(key)
        if isinstance(candidate, (dict, list, tuple)):
            return candidate
    return data


def find_key(mapping: dict[str, Any], logical_name: str) -> str | None:
    for key in ALIASES[logical_name]:
        if key in mapping:
            return key
    return None


def get_value(frame: dict[str, Any], logical_name: str) -> Any | None:
    key = find_key(frame, logical_name)
    return None if key is None else frame[key]


def frame_count(motion: Any) -> int:
    if isinstance(motion, (list, tuple)):
        return len(motion)
    if not isinstance(motion, dict):
        raise TypeError("Motion must be a dict-of-arrays or a list-of-dicts.")

    ignored = {"fps", "motion_fps", "dt", "time_step"}
    for key, value in motion.items():
        if key in ignored:
            continue
        if hasattr(value, "shape") and len(value.shape) >= 1:
            return int(value.shape[0])
        if isinstance(value, (list, tuple)) and len(value) > 1:
            return len(value)
    raise ValueError("Could not infer the number of frames from the PKL.")


def frame_at(motion: Any, index: int) -> dict[str, Any]:
    if isinstance(motion, (list, tuple)):
        frame = motion[index]
        if not isinstance(frame, dict):
            raise TypeError("Each item in a list-style motion must be a dict.")
        return frame

    result = {}
    for key, value in motion.items():
        if key in {"fps", "motion_fps", "dt", "time_step"}:
            continue
        try:
            result[key] = value[index]
        except (IndexError, KeyError, TypeError):
            result[key] = value
    return result


def as_numpy(value: Any, name: str) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or Inf.")
    return array


def normalize_quaternion_xyzw(quaternion: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(quaternion))
    if norm < 1.0e-8:
        raise ValueError("Encountered a zero-norm pelvis quaternion.")
    return quaternion / norm


def quaternion_to_xyzw(quaternion: Any, order: str) -> np.ndarray:
    quaternion = as_numpy(quaternion, "pelvis quaternion")
    if quaternion.size != 4:
        raise ValueError(
            f"Pelvis quaternion must contain 4 values; got {quaternion.size}."
        )
    if order == "wxyz":
        quaternion = quaternion[[1, 2, 3, 0]]
    return normalize_quaternion_xyzw(quaternion)


def quaternion_conjugate_xyzw(quaternion: np.ndarray) -> np.ndarray:
    result = quaternion.copy()
    result[:3] *= -1.0
    return result


def quaternion_multiply_xyzw(
    left: np.ndarray, right: np.ndarray
) -> np.ndarray:
    lv, lw = left[:3], left[3]
    rv, rw = right[:3], right[3]
    vector = lw * rv + rw * lv + np.cross(lv, rv)
    scalar = lw * rw - np.dot(lv, rv)
    return np.concatenate((vector, np.array([scalar], dtype=np.float32)))


def angular_velocity_world(
    previous_xyzw: np.ndarray,
    current_xyzw: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Finite-difference world-frame quaternion angular velocity."""
    current = current_xyzw.copy()
    if np.dot(previous_xyzw, current) < 0.0:
        current *= -1.0

    delta = quaternion_multiply_xyzw(
        current,
        quaternion_conjugate_xyzw(previous_xyzw),
    )
    delta = normalize_quaternion_xyzw(delta)

    if delta[3] < 0.0:
        delta *= -1.0

    vector_norm = float(np.linalg.norm(delta[:3]))
    if vector_norm < 1.0e-8:
        return np.zeros(3, dtype=np.float32)

    angle = 2.0 * np.arctan2(vector_norm, np.clip(delta[3], -1.0, 1.0))
    axis = delta[:3] / vector_norm
    return (axis * angle / dt).astype(np.float32)


def resolve_dt(raw_data: Any, motion: Any, fallback_fps: float) -> float:
    for container in (raw_data, motion):
        if not isinstance(container, dict):
            continue
        if "dt" in container:
            return float(np.asarray(container["dt"]).reshape(-1)[0])
        if "time_step" in container:
            return float(np.asarray(container["time_step"]).reshape(-1)[0])
        for key in ("fps", "motion_fps"):
            if key in container:
                fps = float(np.asarray(container[key]).reshape(-1)[0])
                return 1.0 / fps
    return 1.0 / fallback_fps


def extract_pose(
    frame: dict[str, Any], quaternion_order: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    generalized_position = get_value(frame, "generalized_position")

    if generalized_position is not None:
        q = as_numpy(generalized_position, "generalized position")
        if q.size != Q_DIM:
            raise ValueError(
                f"Generalized q must have {Q_DIM} values; got {q.size}. "
                "If the PKL q/qpos contains only joint angles, rename it to "
                "'joint_position' or update ALIASES."
            )
        position = q[:3]
        quaternion = quaternion_to_xyzw(q[3:7], quaternion_order)
        joints = q[7:]
        q = np.concatenate((position, quaternion, joints)).astype(np.float32)
        return position, quaternion, q

    position_value = get_value(frame, "pelvis_position")
    quaternion_value = get_value(frame, "pelvis_quaternion")
    joint_value = get_value(frame, "joint_position")

    missing = []
    if position_value is None:
        missing.append("pelvis_position")
    if quaternion_value is None:
        missing.append("pelvis_quaternion")
    if joint_value is None:
        missing.append("joint_position")
    if missing:
        raise KeyError(
            f"Missing required retarget fields: {missing}. "
            "Add the actual PKL keys to ALIASES near the top of test.py."
        )

    position = as_numpy(position_value, "pelvis position")
    joints = as_numpy(joint_value, "joint position")
    quaternion = quaternion_to_xyzw(quaternion_value, quaternion_order)

    if position.size != 3:
        raise ValueError(f"Pelvis position must have 3 values; got {position.size}.")
    if joints.size != NUM_JOINTS:
        raise ValueError(
            f"G1 joint position must have {NUM_JOINTS} values; got {joints.size}."
        )

    q = np.concatenate((position, quaternion, joints)).astype(np.float32)
    return position, quaternion, q


def extract_velocity(
    frame: dict[str, Any],
    previous_q: np.ndarray | None,
    current_q: np.ndarray,
    dt: float,
) -> np.ndarray:
    generalized_velocity = get_value(frame, "generalized_velocity")
    if generalized_velocity is not None:
        qd = as_numpy(generalized_velocity, "generalized velocity")
        if qd.size == V_DIM:
            return qd.astype(np.float32)
        if qd.size != NUM_JOINTS:
            raise ValueError(
                f"Generalized velocity must have {V_DIM} values; got {qd.size}."
            )

    base_linear = get_value(frame, "base_linear_velocity")
    base_angular = get_value(frame, "base_angular_velocity")
    joint_velocity = get_value(frame, "joint_velocity")

    if base_linear is not None and base_angular is not None:
        linear = as_numpy(base_linear, "base linear velocity")
        angular = as_numpy(base_angular, "base angular velocity")
        if joint_velocity is not None:
            joints = as_numpy(joint_velocity, "joint velocity")
        elif generalized_velocity is not None:
            joints = as_numpy(generalized_velocity, "joint velocity")
        elif previous_q is not None:
            joints = (current_q[7:] - previous_q[7:]) / dt
        else:
            joints = np.zeros(NUM_JOINTS, dtype=np.float32)
        return np.concatenate((linear, angular, joints)).astype(np.float32)

    if previous_q is None:
        return np.zeros(V_DIM, dtype=np.float32)

    linear = (current_q[:3] - previous_q[:3]) / dt
    angular = angular_velocity_world(previous_q[3:7], current_q[3:7], dt)
    joints = (current_q[7:] - previous_q[7:]) / dt
    return np.concatenate((linear, angular, joints)).astype(np.float32)


def to_batch_tensor(array: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(
        array,
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)


def compute_fabric_acceleration(
    fabric: G1PoseFabric,
    q: torch.Tensor,
    qd: torch.Tensor,
    dt: float,
) -> torch.Tensor:
    """
    Prefer the explicit wrapper defined in g1_pose_fabric.py. The fallbacks
    support common BaseFabric variants.
    """
    if hasattr(fabric, "compute_acceleration"):
        return fabric.compute_acceleration(q=q, qd=qd, timestep=dt)

    if hasattr(fabric, "forward"):
        signature = inspect.signature(fabric.forward)
        if "timestep" in signature.parameters:
            result = fabric(q, qd, timestep=dt)
        else:
            result = fabric(q, qd)
    else:
        result = fabric.eval_natural(q, qd, dt)

    if torch.is_tensor(result):
        return result

    if isinstance(result, (tuple, list)):
        # Common return: (metric, force, metric_inv)
        if len(result) >= 3:
            metric, force, metric_inv = result[:3]
            del metric
            return -torch.bmm(
                metric_inv,
                force.unsqueeze(-1),
            ).squeeze(-1)

    raise TypeError(
        "Could not interpret Fabric output. Add a compute_acceleration(q, qd, "
        "timestep) method to G1PoseFabric that returns a (batch, 35) tensor."
    )


def main() -> None:
    args = parse_args()
    if not args.pkl.is_file():
        raise FileNotFoundError(f"PKL file does not exist: {args.pkl}")
    if args.motion_fps <= 0.0:
        raise ValueError("--motion-fps must be positive.")

    raw_data = load_pickle(args.pkl)
    motion = unwrap_motion(raw_data)

    print("PKL structure:")
    describe(raw_data)

    dt = resolve_dt(raw_data, motion, args.motion_fps)
    count = frame_count(motion)
    if args.max_frames is not None:
        count = min(count, args.max_frames)

    device = torch.device(args.device)
    wp.init()
    fabric = G1PoseFabric(
        batch_size=1,
        device=str(device),
        timestep=dt,
        graph_capturable=args.graph_capturable,
    )
    fabric.eval()

    q_history = []
    qd_history = []
    position_targets = []
    quaternion_targets = []
    acceleration_history = []
    integrated_q_history = []
    integrated_qd_history = []

    previous_q = None
    integrated_q = None
    integrated_qd = None

    with torch.no_grad():
        for frame_id in range(count):
            frame = frame_at(motion, frame_id)

            pelvis_position, pelvis_quaternion, q_np = extract_pose(
                frame,
                args.quat_order,
            )
            qd_np = extract_velocity(
                frame,
                previous_q,
                q_np,
                dt,
            )

            position_target = to_batch_tensor(pelvis_position, device)
            quaternion_target = to_batch_tensor(pelvis_quaternion, device)
            reference_q = to_batch_tensor(q_np, device)
            reference_qd = to_batch_tensor(qd_np, device)

            if integrated_q is None:
                integrated_q = reference_q.clone()
                integrated_qd = reference_qd.clone()

            # Recover the four desired end-effector poses from the retargeted
            # reference configuration.
            fabric.end_effector_pose_taskmap(reference_q, None)
            end_effector_transforms = (
                fabric.end_effector_pose_taskmap
                .robot_kinematics.batch_link_transforms_torch[
                    :, fabric.end_effector_pose_taskmap.link_indices
                ]
                .clone()
            )

            # Update both reference-motion tasks.
            fabric.set_motion_targets(
                pelvis_position_target=position_target,
                pelvis_quaternion_target_xyzw=quaternion_target,
                end_effector_position_target=end_effector_transforms[..., :3],
                end_effector_quaternion_target_xyzw=(
                    end_effector_transforms[..., 3:7]
                ),
            )

            # Integrated state + current reference targets -> 35-D acceleration.
            qdd = compute_fabric_acceleration(
                fabric=fabric,
                q=integrated_q,
                qd=integrated_qd,
                dt=dt,
            )

            if qdd.shape != (1, V_DIM):
                raise ValueError(
                    f"Expected Fabric acceleration shape (1, {V_DIM}); "
                    f"got {tuple(qdd.shape)}."
                )
            if not torch.isfinite(qdd).all():
                raise FloatingPointError(
                    f"Fabric produced NaN/Inf at frame {frame_id}."
                )

            # Semi-implicit Euler:
            #   qd[k+1] = qd[k] + dt*qdd[k]
            #   q[k+1]  = integrate(q[k], qd[k+1], dt)
            # The configuration integration handles the base quaternion on
            # SO(3), so q has 36 values while qd/qdd have 35.
            integrated_qd = integrated_qd + dt * qdd
            integrated_q = fabric.integrate_root_position(
                integrated_q,
                integrated_qd,
                dt,
            )

            q_history.append(q_np.copy())
            qd_history.append(qd_np.copy())
            position_targets.append(pelvis_position.copy())
            quaternion_targets.append(pelvis_quaternion.copy())
            acceleration_history.append(qdd[0].detach().cpu().numpy())
            integrated_q_history.append(
                integrated_q[0].detach().cpu().numpy().copy()
            )
            integrated_qd_history.append(
                integrated_qd[0].detach().cpu().numpy().copy()
            )
            previous_q = q_np.copy()

            if frame_id == 0 or (frame_id + 1) % 100 == 0:
                print(
                    f"frame {frame_id + 1:6d}/{count}: "
                    f"|qdd|={torch.linalg.vector_norm(qdd).item():.6f}"
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        q=np.stack(q_history),
        qd=np.stack(qd_history),
        pelvis_position_target=np.stack(position_targets),
        pelvis_quaternion_target_xyzw=np.stack(quaternion_targets),
        qdd_fabric=np.stack(acceleration_history),
        qd_integrated=np.stack(integrated_qd_history),
        q_integrated=np.stack(integrated_q_history),
        joint_names=np.asarray(
            (
                raw_data.get("dof_names", [])
                if isinstance(raw_data, dict)
                else []
            ),
            dtype=str,
        ),
        dt=np.array(dt, dtype=np.float32),
    )

    print(f"Saved {count} frames to: {args.output}")
    print(f"qdd_fabric shape: ({count}, {V_DIM})")
    print(f"qd_integrated shape: ({count}, {V_DIM})")
    print(f"q_integrated shape: ({count}, {Q_DIM})")


if __name__ == "__main__":
    main()
