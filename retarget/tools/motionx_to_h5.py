#!/usr/bin/env python3
"""
Convert Motion-X SMPL-X (322-dim) npy files into Pose-to-Motion h5 format, with a built-in轴交换以匹配机器人坐标系。

Input (per .npy):
    motion[t, 322] with fields (axis-angle unless stated):
        0:3       root_orient
        3:66      pose_body (21 joints)
        66:156    pose_hand (45 left, 45 right)
        156:159   pose_jaw
        159:209   face_expr (ignored here)
        209:309   face_shape (ignored here)
        309:312   trans (root translation)
        312:      betas (ignored here)

Output (per .h5):
    rotat: (T, 55, 4) quaternion per joint (xyz, w) following SMPL-X joint order
    joint_parents_idx: (55,) parent indices from SMPL-X kinematic tree (root = -1)
    joint_offsets: (55, 3) offsets from parent to child in the rest pose
    end_sites: (4,) [L_Foot, R_Foot, L_Wrist, R_Wrist] indices for contact loss
    T_pose: (55, 3) rest joints shifted so the lowest point on the chosen up-axis is at 0
    joint_names: (55,) byte names from the SMPL-X model
    joint_translation: (T, 55, 3) only root filled with `trans`, others zero

Usage:
    python tools/motionx_to_h5.py \\
        --motion_root motionx/motion_data/smplx_322 \\
        --smplx_model motionx/models_smplx_v1_1/models/smplx/SMPLX_NEUTRAL.npz \\
        --out_root data/MotionX_h5
"""

import argparse
import glob
import os
from typing import List, Tuple

# keep BLAS threads low to avoid OMP issues in constrained envs
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("KMP_AFFINITY", "disabled")
os.environ.setdefault("KMP_BLOCKTIME", "0")
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")

import h5py
import numpy as np

# 直接交换坐标轴：不做旋转矩阵变换，直接重排 xyz
# new_xyz = old_xyz[..., AXIS_SWAP_ORDER] * AXIS_SWAP_SIGN
AXIS_SWAP_ORDER = (2, 0, 1)  # new X <- old Z, new Y <- old X, new Z <- old Y (竖直)
AXIS_SWAP_SIGN = np.array([1.0, 1.0, 1.0], dtype=np.float64)
GROUND_AXIS = 2  # 竖直轴（交换后为 z）
# 保留的关节索引（去除手掌/手指/眼睛/颈部/头/下颌），基于 SMPL-X 原始 55 关节顺序
# 仅保留躯干+四肢：Pelvis, 双腿, 脊柱, Collar/Shoulder/Elbow/Wrist
KEEP_JOINT_IDX = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,  # pelvis/legs/spine
    13, 14, 16, 17, 18, 19, 20, 21         # collar/shoulder/elbow/wrist
]


def axis_angle_to_quat(rotvecs: np.ndarray) -> np.ndarray:
    """
    Convert axis-angle (..., 3) to quaternion (xyz, w).
    Small angles are clamped to identity to avoid numerical noise.
    """
    rv = rotvecs.reshape(-1, 3).astype(np.float64)
    angles = np.linalg.norm(rv, axis=1)
    quats = np.zeros((rv.shape[0], 4), dtype=np.float64)
    small = angles < 1e-8
    quats[small, 3] = 1.0

    mask = ~small
    if np.any(mask):
        axis = rv[mask] / angles[mask, None]
        half = 0.5 * angles[mask]
        sin_half = np.sin(half)
        quats[mask, :3] = axis * sin_half[:, None]
        quats[mask, 3] = np.cos(half)

    norm = np.linalg.norm(quats, axis=1, keepdims=True)
    quats /= np.clip(norm, 1e-8, None)
    return quats.reshape(rotvecs.shape[:-1] + (4,)).astype(np.float32)


def rotvec_to_mat(rotvecs: np.ndarray) -> np.ndarray:
    """Rodrigues formula. rotvecs: (...,3)"""
    rv = rotvecs.astype(np.float64)
    angle = np.linalg.norm(rv, axis=-1, keepdims=True)
    small = angle < 1e-8
    axis = np.zeros_like(rv)
    axis[~small[..., 0]] = rv[~small[..., 0]] / angle[~small[..., 0]]
    x, y, z = axis[..., 0:1], axis[..., 1:2], axis[..., 2:3]
    c = np.cos(angle)
    s = np.sin(angle)
    C = 1.0 - c
    mats = np.zeros(rv.shape[:-1] + (3, 3), dtype=np.float64)
    mats[..., 0, 0] = c[..., 0] + x[..., 0] * x[..., 0] * C[..., 0]
    mats[..., 0, 1] = x[..., 0] * y[..., 0] * C[..., 0] - z[..., 0] * s[..., 0]
    mats[..., 0, 2] = x[..., 0] * z[..., 0] * C[..., 0] + y[..., 0] * s[..., 0]
    mats[..., 1, 0] = y[..., 0] * x[..., 0] * C[..., 0] + z[..., 0] * s[..., 0]
    mats[..., 1, 1] = c[..., 0] + y[..., 0] * y[..., 0] * C[..., 0]
    mats[..., 1, 2] = y[..., 0] * z[..., 0] * C[..., 0] - x[..., 0] * s[..., 0]
    mats[..., 2, 0] = z[..., 0] * x[..., 0] * C[..., 0] - y[..., 0] * s[..., 0]
    mats[..., 2, 1] = z[..., 0] * y[..., 0] * C[..., 0] + x[..., 0] * s[..., 0]
    mats[..., 2, 2] = c[..., 0] + z[..., 0] * z[..., 0] * C[..., 0]
    # for tiny angles, use identity
    mats[small[..., 0]] = np.eye(3, dtype=np.float64)
    return mats


def mat_to_quat(mats: np.ndarray) -> np.ndarray:
    """mats: (...,3,3) -> (...,4) (xyz, w)."""
    m = mats.reshape(-1, 3, 3)
    qw = np.empty(m.shape[0], dtype=np.float64)
    qx = np.empty_like(qw)
    qy = np.empty_like(qw)
    qz = np.empty_like(qw)
    trace = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]
    cond1 = trace > 0
    t = np.sqrt(np.maximum(trace[cond1] + 1.0, 1e-12)) * 2
    qw[cond1] = 0.25 * t
    qx[cond1] = (m[cond1, 2, 1] - m[cond1, 1, 2]) / t
    qy[cond1] = (m[cond1, 0, 2] - m[cond1, 2, 0]) / t
    qz[cond1] = (m[cond1, 1, 0] - m[cond1, 0, 1]) / t

    cond2 = (m[:, 0, 0] > m[:, 1, 1]) & (m[:, 0, 0] > m[:, 2, 2]) & (~cond1)
    t = np.sqrt(np.maximum(1.0 + m[cond2, 0, 0] - m[cond2, 1, 1] - m[cond2, 2, 2], 1e-12)) * 2
    qw[cond2] = (m[cond2, 2, 1] - m[cond2, 1, 2]) / t
    qx[cond2] = 0.25 * t
    qy[cond2] = (m[cond2, 0, 1] + m[cond2, 1, 0]) / t
    qz[cond2] = (m[cond2, 0, 2] + m[cond2, 2, 0]) / t

    cond3 = (m[:, 1, 1] > m[:, 2, 2]) & (~cond1) & (~cond2)
    t = np.sqrt(np.maximum(1.0 + m[cond3, 1, 1] - m[cond3, 0, 0] - m[cond3, 2, 2], 1e-12)) * 2
    qw[cond3] = (m[cond3, 0, 2] - m[cond3, 2, 0]) / t
    qx[cond3] = (m[cond3, 0, 1] + m[cond3, 1, 0]) / t
    qy[cond3] = 0.25 * t
    qz[cond3] = (m[cond3, 1, 2] + m[cond3, 2, 1]) / t

    cond4 = (~cond1) & (~cond2) & (~cond3)
    t = np.sqrt(np.maximum(1.0 + m[cond4, 2, 2] - m[cond4, 0, 0] - m[cond4, 1, 1], 1e-12)) * 2
    qw[cond4] = (m[cond4, 1, 0] - m[cond4, 0, 1]) / t
    qx[cond4] = (m[cond4, 0, 2] + m[cond4, 2, 0]) / t
    qy[cond4] = (m[cond4, 1, 2] + m[cond4, 2, 1]) / t
    qz[cond4] = 0.25 * t

    quat = np.stack([qx, qy, qz, qw], axis=-1)
    return quat.reshape(mats.shape[:-2] + (4,)).astype(np.float32)


def swap_axes(arr: np.ndarray) -> np.ndarray:
    """Direct axis swap for vectors/rotvecs."""
    return arr[..., AXIS_SWAP_ORDER] * AXIS_SWAP_SIGN


def fk_positions(rotmats: np.ndarray, parents: np.ndarray, offsets: np.ndarray, root_trans: np.ndarray) -> np.ndarray:
    """Forward kinematics for单帧，返回关节全局位置 (J,3)."""
    J = rotmats.shape[0]
    pos = np.zeros((J, 3), dtype=np.float64)
    ori = np.zeros((J, 3, 3), dtype=np.float64)
    for j in range(J):
        p = parents[j]
        if p < 0:
            ori[j] = rotmats[j]
            pos[j] = root_trans + offsets[j]
        else:
            ori[j] = ori[p] @ rotmats[j]
            pos[j] = pos[p] + ori[p] @ offsets[j]
    return pos


def load_smplx_template(model_path: str) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray]:
    """
    Load SMPL-X template to get joint names, kinematic tree, offsets and T-pose.
    """
    model = np.load(model_path, allow_pickle=True)
    joint2num = model["joint2num"].item()
    joint_names = [name for name, _ in sorted(joint2num.items(), key=lambda kv: kv[1])]

    kintree = model["kintree_table"]
    parents = np.zeros(kintree.shape[1], dtype=np.int64)
    parents[kintree[1]] = kintree[0]
    parents[parents == 4294967295] = -1

    # rest joints from template mesh
    v_template = model["v_template"]
    j_regressor = model["J_regressor"]
    joints = (j_regressor[..., None] * v_template[None, :, :]).sum(axis=1)

    offsets = np.zeros_like(joints, dtype=np.float32)
    for j, p in enumerate(parents):
        if p >= 0:
            offsets[j] = joints[j] - joints[p]

    t_pose = joints.astype(np.float32)
    t_pose[:, 1] -= t_pose[:, 1].min()  # lift so the lowest joint sits on the ground
    return joint_names, parents, offsets, t_pose


def prune_smplx(joint_names, parents, offsets, t_pose, keep_idx):
    keep_set = set(keep_idx)
    old_to_new = {old: new for new, old in enumerate(keep_idx)}

    positions = t_pose  # global positions
    parents_new = []
    for old in keep_idx:
        p = parents[old]
        while p >= 0 and p not in keep_set:
            p = parents[p]
        parents_new.append(-1 if p < 0 else old_to_new[p])
    parents_new = np.array(parents_new, dtype=np.int64)

    pos_keep = positions[keep_idx]
    offsets_new = np.zeros_like(pos_keep, dtype=np.float32)
    for i, p in enumerate(parents_new):
        if p >= 0:
            offsets_new[i] = pos_keep[i] - pos_keep[p]

    joint_names_new = [joint_names[i] for i in keep_idx]
    t_pose_new = pos_keep.copy()
    return joint_names_new, parents_new, offsets_new, t_pose_new


def write_h5(
    out_path: str,
    rotat: np.ndarray,
    parents: np.ndarray,
    offsets: np.ndarray,
    t_pose: np.ndarray,
    joint_names: List[str],
    end_sites: np.ndarray,
    joint_translation: np.ndarray,
    fps: float | None = None,
) -> None:
    if fps is not None and (not np.isfinite(fps) or fps <= 0):
        raise ValueError(f"fps must be positive, got {fps}")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with h5py.File(out_path, "w") as f:
        if fps is not None:
            f.attrs["fps"] = float(fps)
        f.create_dataset("rotat", data=rotat.astype(np.float32))
        f.create_dataset("joint_parents_idx", data=parents.astype(np.int64))
        f.create_dataset("joint_offsets", data=offsets.astype(np.float32))
        f.create_dataset("end_sites", data=end_sites.astype(np.int64))
        f.create_dataset("T_pose", data=t_pose.astype(np.float32))
        f.create_dataset("joint_names", data=np.array([n.encode("utf-8") for n in joint_names]))
        f.create_dataset("joint_translation", data=joint_translation.astype(np.float32))


def motion_to_rotvecs(motion: np.ndarray, joint_num: int) -> np.ndarray:
    """
    Map Motion-X 322-dim vector to per-joint axis-angles following SMPL-X order.
    Missing eye poses are set to identity (zeros).
    """
    if motion.shape[1] != 322:
        raise ValueError(f"Expected motion dim 322, got {motion.shape[1]}")
    T = motion.shape[0]
    rotvecs = np.zeros((T, joint_num, 3), dtype=np.float32)

    rotvecs[:, 0, :] = motion[:, :3]  # root
    rotvecs[:, 1:22, :] = motion[:, 3:66].reshape(T, 21, 3)  # body (21 joints)

    left_hand = motion[:, 66:111].reshape(T, 15, 3)
    right_hand = motion[:, 111:156].reshape(T, 15, 3)
    rotvecs[:, 25:40, :] = left_hand
    rotvecs[:, 40:55, :] = right_hand

    rotvecs[:, 22, :] = motion[:, 156:159]  # jaw
    # eyes (23, 24) remain zeros (identity rotations)
    return rotvecs


def process_file(
    npy_path: str,
    out_root: str,
    motion_root: str,
    joint_names: List[str],
    parents: np.ndarray,
    offsets: np.ndarray,
    t_pose: np.ndarray,
    end_sites: np.ndarray,
    ground_axis: int,
    fps: float,
) -> None:
    motion = np.load(npy_path)
    rotvecs = motion_to_rotvecs(motion, len(joint_names))

    # 去除手掌/手指等多余关节，并重新映射父节点/偏移
    rotvecs = rotvecs[:, KEEP_JOINT_IDX, :]
    joint_names, parents, offsets, t_pose = prune_smplx(joint_names, parents, offsets, t_pose, KEEP_JOINT_IDX)
    rotvecs = swap_axes(rotvecs)

    # 依据保留后的关节名重算 end_sites（优先脚和手腕）
    name2idx = {n: i for i, n in enumerate(joint_names)}
    end_sites = []
    for n in ["L_Foot", "R_Foot", "L_Wrist", "R_Wrist"]:
        if n in name2idx:
            end_sites.append(name2idx[n])
    end_sites = np.array(end_sites, dtype=np.int64)

    rotmats = rotvec_to_mat(rotvecs)
    rotat = mat_to_quat(rotmats)

    joint_translation = np.zeros((motion.shape[0], len(joint_names), 3), dtype=np.float32)
    # 使用原始全局平移（旋转后）
    trans = motion[:, 309:312].copy()
    # 对照 Motion-X 官方处理：AIST 子集需要缩放 /94 并翻转 z 轴
    if "aist" in npy_path.lower():
        trans = trans / 94.0
        trans[:, 2] *= -1
    trans = swap_axes(trans)
    offsets = swap_axes(offsets)
    t_pose = swap_axes(t_pose)
    t_pose[:, ground_axis] -= t_pose[:, ground_axis].min()

    # 用第一帧最低点对齐地面：计算第一帧 FK 位置的最小 z，并整体平移
    root_trans = trans
    pos0 = fk_positions(rotmats[0], parents, offsets, root_trans[0])
    ground_offset = pos0[:, ground_axis].min()
    root_trans[:, ground_axis] -= ground_offset
    joint_translation[:, 0, :] = root_trans

    rel = os.path.relpath(npy_path, motion_root)
    out_path = os.path.join(out_root, os.path.splitext(rel)[0] + ".h5")
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
        f"({motion.shape[0]} frames, {len(joint_names)} joints, {fps:g} FPS)"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert Motion-X smplx_322 npy to Pose-to-Motion h5.")
    ap.add_argument("--motion_root", default="motionx/motion_data/smplx_322", help="Root directory containing Motion-X smplx_322 npy files")
    ap.add_argument(
        "--smplx_model",
        default="motionx/models_smplx_v1_1/models/smplx/SMPLX_NEUTRAL.npz",
        help="SMPL-X model npz (used for joint order / offsets)",
    )
    ap.add_argument("--out_root", default="data/MotionX_h5", help="Output directory for h5 files")
    ap.add_argument("--fps", type=float, default=30.0, help="Frame rate stored in each output h5.")
    ap.add_argument("--max_files", type=int, default=None, help="Optional limit for debugging")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing h5 files")
    args = ap.parse_args()

    joint_names, parents, offsets, t_pose = load_smplx_template(args.smplx_model)
    joint2num = {name: idx for idx, name in enumerate(joint_names)}
    end_sites = np.array(
        [joint2num["L_Foot"], joint2num["R_Foot"], joint2num["L_Wrist"], joint2num["R_Wrist"]],
        dtype=np.int64,
    )

    npy_files = sorted(glob.glob(os.path.join(args.motion_root, "**", "*.npy"), recursive=True))
    if args.max_files:
        npy_files = npy_files[: args.max_files]
    if not npy_files:
        raise SystemExit(f"No npy files found under {args.motion_root}")

    for npy_path in npy_files:
        rel = os.path.relpath(npy_path, args.motion_root)
        out_path = os.path.join(args.out_root, os.path.splitext(rel)[0] + ".h5")
        if os.path.exists(out_path) and not args.overwrite:
            print(f"[skip] {out_path} exists")
            continue
        process_file(
            npy_path,
            args.out_root,
            args.motion_root,
            joint_names,
            parents,
            offsets,
            t_pose,
            end_sites,
            GROUND_AXIS,
            args.fps,
        )


if __name__ == "__main__":
    main()
