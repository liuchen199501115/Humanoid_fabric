from pathlib import Path
import pickle

import numpy as np
import torch

from .rotation import matrix_to_quaternion_xyzw
from .urdf import DEFAULT_FEET, URDFKinematicSpec, load_urdf_spec


def _as_rotation_matrices(rot_mats):
    rot = np.asarray(rot_mats, dtype=np.float64)
    if rot.ndim == 3 and rot.shape[-1] == 9:
        return rot.reshape(rot.shape[0], rot.shape[1], 3, 3)
    if rot.ndim == 4 and rot.shape[-2:] == (3, 3):
        return rot
    raise ValueError(f"Unexpected rotation shape for pkl export: {rot.shape}")


def _rotation_vectors_from_matrix(matrix):
    """Convert SO(3) matrices to shortest-path rotation vectors."""
    quaternion = matrix_to_quaternion_xyzw(
        torch.as_tensor(matrix, dtype=torch.float64)
    ).cpu().numpy()
    quaternion = np.where(quaternion[:, 3:4] < 0.0, -quaternion, quaternion)
    vector = quaternion[:, :3]
    sin_half = np.linalg.norm(vector, axis=1)
    angle = 2.0 * np.arctan2(sin_half, np.clip(quaternion[:, 3], 0.0, 1.0))
    scale = np.divide(
        angle,
        sin_half,
        out=np.full_like(angle, 2.0),
        where=sin_half > 1e-12,
    )
    return vector * scale[:, None]


def project_rotations_to_urdf_dofs(rot_mats, joint_names, urdf_path):
    """Project arbitrary local SO(3) outputs onto signed URDF hinge axes."""
    names = [n.decode("utf-8") if isinstance(n, bytes) else str(n) for n in joint_names]
    rot = _as_rotation_matrices(rot_mats)

    if names and names[-1] == "root_velocity_node":
        rot = rot[:, :-1]
        names = names[:-1]

    if len(names) != rot.shape[1]:
        raise ValueError(
            f"joint_names has {len(names)} entries but rotations has {rot.shape[1]} joints"
        )
    if len(set(names)) != len(names):
        raise ValueError("joint_names contains duplicates and cannot be aligned safely")

    spec = load_urdf_spec(urdf_path)
    name_to_index = {name: idx for idx, name in enumerate(names)}
    missing = [name for name in spec.joint_names if name not in name_to_index]
    if missing:
        raise ValueError(f"Generated rotation tensor is missing URDF joints: {missing}")

    root_matrix = rot[:, name_to_index[spec.joint_names[0]]]
    root_quaternion = matrix_to_quaternion_xyzw(
        torch.as_tensor(root_matrix, dtype=torch.float64)
    ).cpu().numpy()
    dof_pos = np.empty((rot.shape[0], len(spec.dof_names)), dtype=np.float64)
    for dof_idx, name in enumerate(spec.dof_names):
        local_rotation = rot[:, name_to_index[name]]
        relative_rotation = np.einsum(
            "ij,tjk->tik", spec.base_rotations[dof_idx + 1].T, local_rotation
        )
        rotation_vector = _rotation_vectors_from_matrix(relative_rotation)
        axis = spec.axes[dof_idx + 1]
        axis = axis / np.linalg.norm(axis).clip(min=1e-12)
        dof_pos[:, dof_idx] = np.einsum("ti,i->t", rotation_vector, axis)
    return root_matrix, root_quaternion, dof_pos, spec


def _axis_angle_matrices(axes, angles):
    axes = np.asarray(axes, dtype=np.float64)
    axes = axes / np.linalg.norm(axes, axis=-1, keepdims=True).clip(min=1e-12)
    x, y, z = axes.T
    zeros = np.zeros_like(x)
    skew = np.stack(
        (
            zeros, -z, y,
            z, zeros, -x,
            -y, x, zeros,
        ),
        axis=-1,
    ).reshape(-1, 3, 3)
    identity = np.eye(3, dtype=np.float64)[None, None]
    sin = np.sin(angles)[..., None, None]
    cos = np.cos(angles)[..., None, None]
    return identity + sin * skew[None] + (1.0 - cos) * np.matmul(skew, skew)[None]


def _forward_kinematics(root_pos, root_matrix, dof_pos, spec: URDFKinematicSpec):
    frame_count = dof_pos.shape[0]
    joint_count = len(spec.joint_names)
    positions = np.zeros((frame_count, joint_count, 3), dtype=np.float64)
    orientations = np.zeros((frame_count, joint_count, 3, 3), dtype=np.float64)
    positions[:, 0] = root_pos + spec.offsets[0]
    orientations[:, 0] = root_matrix

    axis_rotations = _axis_angle_matrices(spec.axes[1:], dof_pos)
    local_rotations = np.matmul(spec.base_rotations[None, 1:], axis_rotations)
    for joint_idx in range(1, joint_count):
        parent_idx = int(spec.parents[joint_idx])
        if parent_idx < 0 or parent_idx >= joint_idx:
            raise ValueError(
                f"URDF chain is not parent-before-child at joint {joint_idx}: "
                f"parent={parent_idx}"
            )
        parent_orientation = orientations[:, parent_idx]
        positions[:, joint_idx] = positions[:, parent_idx] + np.einsum(
            "tij,j->ti", parent_orientation, spec.offsets[joint_idx]
        )
        orientations[:, joint_idx] = np.matmul(
            parent_orientation, local_rotations[:, joint_idx - 1]
        )
    return positions, orientations


def align_projected_motion_to_ground(root_pos, root_matrix, dof_pos, spec):
    """Shift the trajectory so the first-frame lowest foot sphere is at z=0."""
    root_pos = np.asarray(root_pos, dtype=np.float64).copy()
    if root_pos.shape[0] == 0:
        return root_pos
    positions, orientations = _forward_kinematics(
        root_pos[:1], root_matrix[:1], dof_pos[:1], spec
    )
    name_to_index = {name: idx for idx, name in enumerate(spec.joint_names)}
    minimum_heights = []
    for joint_name in DEFAULT_FEET:
        collision = spec.foot_collisions[joint_name]
        joint_idx = name_to_index[joint_name]
        world_centers = positions[:, joint_idx, None, :] + np.einsum(
            "tij,sj->tsi", orientations[:, joint_idx], collision.centers
        )
        minimum_heights.append(
            np.min(world_centers[..., 2] - collision.radii[None, :])
        )
    root_pos[:, 2] -= min(minimum_heights)
    return root_pos


def _remove_short_true_runs(mask, minimum_frames):
    """Remove True runs shorter than ``minimum_frames``."""
    result = np.asarray(mask, dtype=bool).copy()
    if result.ndim != 1:
        raise ValueError(f"contact mask must be one-dimensional, got {result.shape}")
    if minimum_frames < 1:
        raise ValueError("minimum_frames must be at least one")
    padded = np.pad(result.astype(np.int8), (1, 1))
    edges = np.flatnonzero(np.diff(padded))
    for start, stop in edges.reshape(-1, 2):
        if stop - start < minimum_frames:
            result[start:stop] = False
    return result


def core_contact_sequence(
    root_pos,
    root_matrix,
    dof_pos,
    spec,
    fps,
    velocity_threshold=0.08,
    minimum_duration=0.10,
    smoothing_duration=0.10,
    height_threshold=0.04,
    support_fallback_height=0.08,
):
    """Infer physically plausible left/right contacts from retargeted feet.

    CoRe smooths each retargeted toe's vertical trajectory, thresholds its
    absolute vertical velocity, and rejects contact runs that are too short.
    Velocity alone labels a stationary swing-foot apex as contact and can turn
    retarget jitter into long false flight phases.  We additionally require the
    foot collision geometry to be near the ground.  When neither velocity mask
    survives but a foot is still close to the floor, the lower foot is retained
    as support; true flight remains possible only above
    ``support_fallback_height``.  The returned columns follow ``DEFAULT_FEET``.
    """
    if fps <= 0.0:
        raise ValueError("fps must be positive")
    if velocity_threshold <= 0.0:
        raise ValueError("velocity_threshold must be positive")
    if minimum_duration < 0.0 or smoothing_duration < 0.0:
        raise ValueError("contact durations must be nonnegative")
    if height_threshold <= 0.0:
        raise ValueError("height_threshold must be positive")
    if support_fallback_height < height_threshold:
        raise ValueError(
            "support_fallback_height must be at least height_threshold"
        )

    positions, orientations = _forward_kinematics(
        np.asarray(root_pos, dtype=np.float64),
        np.asarray(root_matrix, dtype=np.float64),
        np.asarray(dof_pos, dtype=np.float64),
        spec,
    )
    frame_count = positions.shape[0]
    if frame_count == 0:
        return np.empty((0, 2), dtype=bool)

    name_to_index = {name: idx for idx, name in enumerate(spec.joint_names)}
    window = max(1, int(round(smoothing_duration * fps)))
    if window % 2 == 0:
        window += 1
    kernel = np.full(window, 1.0 / window, dtype=np.float64)
    minimum_frames = max(1, int(np.ceil(minimum_duration * fps)))
    contacts = []
    sole_heights = []
    for foot_name in DEFAULT_FEET:
        collision = spec.foot_collisions[foot_name]
        # +x is forward in the supplied G1 foot model. Average the lateral
        # pair so foot roll does not arbitrarily select one toe corner.
        max_x = np.max(collision.centers[:, 0])
        toe_centers = collision.centers[
            np.isclose(collision.centers[:, 0], max_x)
        ]
        toe_local = np.mean(toe_centers, axis=0)
        foot_idx = name_to_index[foot_name]
        toe_world = positions[:, foot_idx] + np.einsum(
            "tij,j->ti", orientations[:, foot_idx], toe_local
        )
        collision_centers_world = positions[:, foot_idx, None, :] + np.einsum(
            "tij,sj->tsi", orientations[:, foot_idx], collision.centers
        )
        sole_height = np.min(
            collision_centers_world[..., 2] - collision.radii[None, :],
            axis=1,
        )
        sole_heights.append(sole_height)
        toe_z = toe_world[:, 2]
        if window > 1:
            pad = window // 2
            toe_z = np.convolve(
                np.pad(toe_z, (pad, pad), mode="edge"), kernel, mode="valid"
            )
        vertical_velocity = (
            np.zeros_like(toe_z)
            if frame_count == 1
            else np.gradient(toe_z, 1.0 / fps)
        )
        contacts.append(
            _remove_short_true_runs(
                (np.abs(vertical_velocity) < velocity_threshold)
                & (sole_height <= height_threshold),
                minimum_frames,
            )
        )
    sequence = np.stack(contacts, axis=1)
    sole_heights = np.stack(sole_heights, axis=1)

    # A locomotion controller must not lose every support constraint merely
    # because the retargeted planted foot jitters faster than the velocity
    # threshold. Preserve genuine flight, but keep the lower near-ground foot
    # for all other frames.
    neither = ~sequence.any(axis=1)
    lower_foot = np.argmin(sole_heights, axis=1)
    use_fallback = neither & (
        np.min(sole_heights, axis=1) <= support_fallback_height
    )
    fallback_frames = np.flatnonzero(use_fallback)
    sequence[fallback_frames, lower_foot[fallback_frames]] = True
    return sequence


def export_robot_pkl(
    rot_mats,
    root_pos,
    joint_names,
    urdf_path,
    fps,
    out_path,
    info=None,
    align_ground=True,
    clamp_limits=True,
    contact_velocity_threshold=0.08,
    contact_minimum_duration=0.10,
    contact_smoothing_duration=0.10,
    contact_height_threshold=0.04,
    contact_support_fallback_height=0.08,
):
    del info
    root_matrix, root_quaternion, dof_pos, spec = project_rotations_to_urdf_dofs(
        rot_mats, joint_names, urdf_path
    )
    frame_count = dof_pos.shape[0]
    if root_pos is None:
        root_pos = np.zeros((frame_count, 3), dtype=np.float64)
    else:
        root_pos = np.asarray(root_pos, dtype=np.float64)
        if root_pos.ndim != 2 or root_pos.shape[1] != 3:
            raise ValueError(f"root_pos must have shape [T,3], got {root_pos.shape}")
        if root_pos.shape[0] < frame_count:
            raise ValueError(
                f"root_pos has {root_pos.shape[0]} frames, expected at least {frame_count}"
            )
        root_pos = root_pos[:frame_count].copy()

    if clamp_limits:
        violations = (dof_pos < spec.lower[None]) | (dof_pos > spec.upper[None])
        if np.any(violations):
            affected_joints = int(np.any(violations, axis=0).sum())
            print(
                f"[warn] Clamped {int(violations.sum())} joint values across "
                f"{affected_joints} joints to URDF limits."
            )
        dof_pos = np.clip(dof_pos, spec.lower[None], spec.upper[None])

    if align_ground:
        root_pos = align_projected_motion_to_ground(
            root_pos, root_matrix, dof_pos, spec
        )

    contact_sequence = core_contact_sequence(
        root_pos,
        root_matrix,
        dof_pos,
        spec,
        fps,
        velocity_threshold=contact_velocity_threshold,
        minimum_duration=contact_minimum_duration,
        smoothing_duration=contact_smoothing_duration,
        height_threshold=contact_height_threshold,
        support_fallback_height=contact_support_fallback_height,
    )

    motion = {
        "fps": float(fps),
        "root_pos": root_pos.astype(np.float32),
        "root_rot": root_quaternion.astype(np.float32),
        "dof_pos": dof_pos.astype(np.float32),
        "dof_names": list(spec.dof_names),
        "contact_sequence": contact_sequence,
        "contact_names": ["left_foot", "right_foot"],
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        pickle.dump(motion, f)
    return motion
