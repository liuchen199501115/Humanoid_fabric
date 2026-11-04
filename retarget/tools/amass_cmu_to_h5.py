#!/usr/bin/env python3
"""
Convert AMASS CMU SMPL-X npz files into Pose-to-Motion h5 format.

This version swaps axes to match MotionX (Z-up, X-forward):
  - rotvecs, trans, offsets, and T_pose are axis-swapped.
  - applies a fixed inverse quaternion correction to align the root orientation
    from the raw AMASS space (approx [0.5, 0.5, 0.5, 0.5]) to a canonical forward direction,
    expressed in the swapped coordinate basis.
"""

import argparse
import glob
import os
import sys
from typing import Dict, Tuple

import numpy as np

# ensure repo root is on sys.path so we can import tools.motionx_to_h5
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tools.motionx_to_h5 import (  # pylint: disable=wrong-import-position
    KEEP_JOINT_IDX,
    load_smplx_template,
    mat_to_quat,
    prune_smplx,
    rotvec_to_mat,
    write_h5,
)

# Axis swap used for motion and rest pose (Z-up).
AXIS_SWAP_ORDER = (2, 0, 1)  # new X <- old Z, new Y <- old X, new Z <- old Y
AXIS_SWAP_SIGN = np.array([1.0, 1.0, 1.0], dtype=np.float64)
GROUND_AXIS = 2

# Fixed inverse quaternion to correct alignment.
# Targets the observed rotation of approx [0.5, 0.5, 0.5, 0.5] -> Identity.
FIXED_Q_INV = np.array([-0.5, -0.5, -0.5, 0.5], dtype=np.float32)


def swap_axes(arr: np.ndarray) -> np.ndarray:
    return arr[..., AXIS_SWAP_ORDER] * AXIS_SWAP_SIGN


def _axis_swap_matrix() -> np.ndarray:
    mat = np.zeros((3, 3), dtype=np.float64)
    for i, j in enumerate(AXIS_SWAP_ORDER):
        mat[i, j] = AXIS_SWAP_SIGN[i]
    return mat


AXIS_SWAP_MAT = _axis_swap_matrix()
AXIS_SWAP_MAT_T = AXIS_SWAP_MAT.T


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    """q: (...,4) xyz w -> (...,3,3)."""
    q = np.asarray(q, dtype=np.float64)
    if q.ndim == 1:
        q = q[None, :]
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    m = np.empty((q.shape[0], 3, 3), dtype=np.float64)
    m[:, 0, 0] = 1 - 2 * (yy + zz)
    m[:, 0, 1] = 2 * (xy - wz)
    m[:, 0, 2] = 2 * (xz + wy)
    m[:, 1, 0] = 2 * (xy + wz)
    m[:, 1, 1] = 1 - 2 * (xx + zz)
    m[:, 1, 2] = 2 * (yz - wx)
    m[:, 2, 0] = 2 * (xz - wy)
    m[:, 2, 1] = 2 * (yz + wx)
    m[:, 2, 2] = 1 - 2 * (xx + yy)
    if m.shape[0] == 1:
        return m[0]
    return m


FIXED_Q_INV_SWAPPED = mat_to_quat(AXIS_SWAP_MAT @ quat_to_mat(FIXED_Q_INV) @ AXIS_SWAP_MAT_T)


def _to_str(value) -> str:
    if isinstance(value, np.ndarray):
        if value.shape == ():
            value = value.item()
        elif value.size == 1:
            value = value.reshape(()).item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    return str(value)


def _read_frame_rate(data: Dict[str, np.ndarray], override: float = None) -> float:
    value = override
    if value is None:
        for key in ("mocap_frame_rate", "mocap_framerate", "fps", "frame_rate"):
            if key in data:
                value = float(np.asarray(data[key]).reshape(()))
                break
    if value is None:
        raise ValueError("Source npz has no frame-rate metadata; pass --fps explicitly.")
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"fps must be positive, got {value}")
    return value


# --- Quaternion Math Helpers ---
def q_mul_broadcast(q1, q2):
    """
    Quaternion multiplication q1 * q2 with broadcasting.
    Inputs can be (4,) or (N, 4). Returns (N, 4).
    """
    # Ensure inputs are at least 2D for broadcasting
    if q1.ndim == 1: q1 = q1[None, :]
    if q2.ndim == 1: q2 = q2[None, :]

    x1, y1, z1, w1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    x2, y2, z2, w2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return np.stack([x, y, z, w], axis=-1)


def q_rot_vec_broadcast(q, v):
    """
    Rotate vector v using quaternion q.
    q: (4,) or (N, 4)
    v: (N, 3)
    Returns (N, 3)
    """
    if q.ndim == 1: q = q[None, :]
    
    q_vec = q[:, :3]
    q_w = q[:, 3:]  # (N, 1)

    uv = np.cross(q_vec, v)
    uuv = np.cross(q_vec, uv)
    return v + 2 * (q_w * uv + uuv)
# -------------------------------


def _ensure_2d(name: str, arr: np.ndarray, t: int = None) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 1D or 2D, got shape {arr.shape}")
    if t is None:
        return arr
    if arr.shape[0] == t:
        return arr
    if arr.shape[0] == 1:
        return np.repeat(arr, t, axis=0)
    raise ValueError(f"{name} has {arr.shape[0]} frames, expected {t}")


def _infer_num_frames(data: Dict[str, np.ndarray]) -> int:
    for key in (
        "trans",
        "root_orient",
        "pose_body",
        "poses",
        "pose_hand",
        "pose_lhand",
        "pose_rhand",
        "pose_jaw",
        "pose_eye",
        "pose_leye",
        "pose_reye",
    ):
        if key in data:
            arr = np.asarray(data[key])
            return 1 if arr.ndim == 1 else arr.shape[0]
    raise ValueError("Cannot infer number of frames from npz keys.")


def _split_hands(data: Dict[str, np.ndarray], t: int) -> Tuple[np.ndarray, np.ndarray]:
    left = right = None
    if "pose_lhand" in data:
        left = _ensure_2d("pose_lhand", data["pose_lhand"], t)
    if "pose_rhand" in data:
        right = _ensure_2d("pose_rhand", data["pose_rhand"], t)
    if (left is None or right is None) and "pose_hand" in data:
        pose_hand = _ensure_2d("pose_hand", data["pose_hand"], t)
        if pose_hand.shape[1] == 90:
            if left is None:
                left = pose_hand[:, :45]
            if right is None:
                right = pose_hand[:, 45:]
        elif pose_hand.shape[1] == 45:
            if left is None and right is None:
                left = pose_hand
                right = np.zeros_like(pose_hand)
            elif left is None:
                left = pose_hand
            elif right is None:
                right = pose_hand
        else:
            raise ValueError(f"pose_hand should have 45 or 90 dims, got {pose_hand.shape[1]}")
    if left is None:
        left = np.zeros((t, 45), dtype=np.float32)
    if right is None:
        right = np.zeros((t, 45), dtype=np.float32)
    return left, right


def _extract_rotvecs_and_trans(
    data: Dict[str, np.ndarray], allow_non_smplx: bool
) -> Tuple[np.ndarray, np.ndarray]:
    model_type = None
    if "surface_model_type" in data:
        model_type = _to_str(data["surface_model_type"])
    elif "model_type" in data:
        model_type = _to_str(data["model_type"])
    if model_type and "smplx" not in model_type.lower() and not allow_non_smplx:
        raise ValueError(f"Model type is '{model_type}', expected SMPL-X.")

    t = _infer_num_frames(data)

    if "root_orient" in data and "pose_body" in data:
        root = _ensure_2d("root_orient", data["root_orient"], t)
        body = _ensure_2d("pose_body", data["pose_body"], t)
        if body.shape[1] != 63:
            raise ValueError(f"pose_body should have 63 dims, got {body.shape[1]}")

        rotvecs = np.zeros((t, 55, 3), dtype=np.float32)
        rotvecs[:, 0, :] = root
        rotvecs[:, 1:22, :] = body.reshape(t, 21, 3)

        if "pose_jaw" in data:
            jaw = _ensure_2d("pose_jaw", data["pose_jaw"], t)
            if jaw.shape[1] != 3:
                raise ValueError(f"pose_jaw should have 3 dims, got {jaw.shape[1]}")
            rotvecs[:, 22, :] = jaw

        if "pose_eye" in data:
            eye = _ensure_2d("pose_eye", data["pose_eye"], t)
            if eye.shape[1] != 6:
                raise ValueError(f"pose_eye should have 6 dims, got {eye.shape[1]}")
            rotvecs[:, 23, :] = eye[:, 0:3]
            rotvecs[:, 24, :] = eye[:, 3:6]
        else:
            if "pose_leye" in data:
                leye = _ensure_2d("pose_leye", data["pose_leye"], t)
                if leye.shape[1] != 3:
                    raise ValueError(f"pose_leye should have 3 dims, got {leye.shape[1]}")
                rotvecs[:, 23, :] = leye
            if "pose_reye" in data:
                reye = _ensure_2d("pose_reye", data["pose_reye"], t)
                if reye.shape[1] != 3:
                    raise ValueError(f"pose_reye should have 3 dims, got {reye.shape[1]}")
                rotvecs[:, 24, :] = reye

        left_hand, right_hand = _split_hands(data, t)
        rotvecs[:, 25:40, :] = left_hand.reshape(t, 15, 3)
        rotvecs[:, 40:55, :] = right_hand.reshape(t, 15, 3)
    elif "poses" in data:
        poses = _ensure_2d("poses", data["poses"], t)
        if poses.shape[1] != 165:
            raise ValueError(f"poses should have 165 dims for SMPL-X, got {poses.shape[1]}")
        rotvecs = poses.reshape(t, 55, 3).astype(np.float32)
    else:
        raise ValueError("Missing SMPL-X pose keys: need root_orient/pose_body or poses.")

    if "trans" in data:
        trans = _ensure_2d("trans", data["trans"], t)
        if trans.shape[1] != 3:
            raise ValueError(f"trans should have 3 dims, got {trans.shape[1]}")
    else:
        trans = np.zeros((t, 3), dtype=np.float32)
    return rotvecs.astype(np.float32), trans.astype(np.float32)


def process_file(
    npz_path: str,
    out_root: str,
    amass_root: str,
    joint_names,
    parents,
    offsets,
    t_pose,
    allow_non_smplx: bool,
    fps_override: float = None,
) -> None:
    data = np.load(npz_path, allow_pickle=True)
    fps = _read_frame_rate(data, fps_override)
    rotvecs, trans = _extract_rotvecs_and_trans(data, allow_non_smplx)

    rotvecs = rotvecs[:, KEEP_JOINT_IDX, :]
    joint_names, parents, offsets, t_pose = prune_smplx(
        joint_names, parents, offsets, t_pose, KEEP_JOINT_IDX
    )

    # recompute end sites after pruning
    name2idx = {n: i for i, n in enumerate(joint_names)}
    end_sites = []
    for n in ("L_Foot", "R_Foot", "L_Wrist", "R_Wrist"):
        if n in name2idx:
            end_sites.append(name2idx[n])
    end_sites = np.array(end_sites, dtype=np.int64)

    # Swap motion axes to match MotionX.
    rotvecs = swap_axes(rotvecs)
    rotmats = rotvec_to_mat(rotvecs)
    rotat = mat_to_quat(rotmats)  # (T, J, 4)

    trans = swap_axes(trans)
    offsets = swap_axes(offsets)
    t_pose = swap_axes(t_pose)
    t_pose[:, GROUND_AXIS] -= t_pose[:, GROUND_AXIS].min()

    # -----------------------------------------------------------
    # [Correction] Apply fixed inverse to align Root and Translation
    # -----------------------------------------------------------
    # Fix root orientation (in swapped basis)
    rotat[:, 0, :] = q_mul_broadcast(FIXED_Q_INV_SWAPPED, rotat[:, 0, :]).squeeze()

    # Fix translation trajectory (must rotate the path too)
    trans = q_rot_vec_broadcast(FIXED_Q_INV_SWAPPED, trans)
    # -----------------------------------------------------------

    joint_translation = np.zeros((rotvecs.shape[0], len(joint_names), 3), dtype=np.float32)
    joint_translation[:, 0, :] = trans

    rel = os.path.relpath(npz_path, amass_root)
    out_path = os.path.join(out_root, os.path.splitext(rel)[0] + ".h5")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    write_h5(
        out_path,
        rotat,
        parents,
        offsets,
        t_pose,
        joint_names,
        end_sites,
        joint_translation,
        fps=fps,
    )
    print(
        f"Saved {out_path} "
        f"({rotvecs.shape[0]} frames, {len(joint_names)} joints, {fps:g} FPS)"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert AMASS CMU SMPL-X npz to Pose-to-Motion h5.")
    ap.add_argument("--amass_root", required=True, help="Root directory for AMASS CMU data")
    ap.add_argument(
        "--smplx_model",
        default="motionx/models_smplx_v1_1/models/smplx/SMPLX_NEUTRAL.npz",
        help="SMPL-X model npz (used for joint order / offsets)",
    )
    ap.add_argument("--out_root", default="data/CMU_h5", help="Output directory for h5 files")
    ap.add_argument(
        "--pattern",
        default=None,
        help="Glob pattern under amass_root (default: **/*_stageii.npz, fallback to **/*_poses.npz)",
    )
    ap.add_argument("--max_files", type=int, default=None, help="Optional limit for debugging")
    ap.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Override source frame-rate metadata.",
    )
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing h5 files")
    ap.add_argument(
        "--allow_non_smplx",
        action="store_true",
        help="Allow non-SMPLX model_type in npz (not recommended)",
    )
    args = ap.parse_args()

    joint_names, parents, offsets, t_pose = load_smplx_template(args.smplx_model)

    if args.pattern:
        patterns = [args.pattern]
    else:
        patterns = ["**/*_stageii.npz", "**/*_poses.npz"]

    npz_files = []
    for pat in patterns:
        npz_files = sorted(glob.glob(os.path.join(args.amass_root, pat), recursive=True))
        if npz_files:
            break
    if not npz_files:
        raise SystemExit(f"No npz files found under {args.amass_root} with patterns: {patterns}")

    if args.max_files:
        npz_files = npz_files[: args.max_files]

    for npz_path in npz_files:
        rel = os.path.relpath(npz_path, args.amass_root)
        out_path = os.path.join(args.out_root, os.path.splitext(rel)[0] + ".h5")
        if os.path.exists(out_path) and not args.overwrite:
            print(f"[skip] {out_path} exists")
            continue
        process_file(
            npz_path,
            args.out_root,
            args.amass_root,
            joint_names,
            parents,
            offsets,
            t_pose,
            args.allow_non_smplx,
            args.fps,
        )


if __name__ == "__main__":
    main()
