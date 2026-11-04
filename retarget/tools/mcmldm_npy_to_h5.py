#!/usr/bin/env python3
"""
Convert MCM-LDM / HumanML3D motions from `(T, 22, 3)` joint positions to
Pose2Motion-compatible human h5 files (`CMU_h5`-style 20-joint skeleton).

Input:
  - single `.npy` file with shape `(T, 22, 3)`, or a directory containing such files.

Output (per file):
  - `rotat`: `(T, 20, 4)` local joint quaternions in `xyzw`
  - `joint_parents_idx`: `(20,)`
  - `joint_offsets`: `(20, 3)`
  - `end_sites`: `(4,)`
  - `T_pose`: `(20, 3)`
  - `joint_names`: `(20,)` bytes
  - `joint_translation`: `(T, 20, 3)` (root translation only)

Notes:
  - This script assumes MCM-LDM outputs are y-up and applies axis swap
    `new = [z, x, y]` so that z is the vertical axis (matching current repo data).
  - The skeleton metadata (joint names / parents / end_sites and a reference pose)
    is loaded from a template h5 file from this repo.
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
from typing import List, Sequence, Tuple

import h5py
import numpy as np

# MCM-LDM / HumanML3D 22-joint order used here (T2M canonical):
# 0 pelvis, 1 r_hip, 2 l_hip, 3 spine1, 4 r_knee, 5 l_knee, 6 spine2, 7 r_ankle,
# 8 l_ankle, 9 spine3, 10 r_foot, 11 l_foot, 12 neck, 13 r_shoulder, 14 l_shoulder,
# 15 head, 16 r_elbow, 17 l_elbow, 18 r_wrist, 19 l_wrist, 20 r_hand, 21 l_hand

# Target 20-joint order (CMU_h5 / MotionX_h5 style):
# Pelvis, L_Hip, R_Hip, Spine1, L_Knee, R_Knee, Spine2, L_Ankle, R_Ankle, Spine3,
# L_Foot, R_Foot, L_Collar, R_Collar, L_Shoulder, R_Shoulder, L_Elbow, R_Elbow,
# L_Wrist, R_Wrist
SRC_TO_DST_DIRECT = {
    0: 0,
    1: 2,
    2: 1,
    3: 3,
    4: 5,
    5: 4,
    6: 6,
    7: 8,
    8: 7,
    9: 9,
    10: 11,
    11: 10,
    13: 15,
    14: 14,
    16: 17,
    17: 16,
    18: 19,
    19: 18,
}

LEFT_SHOULDER_SRC = 14
RIGHT_SHOULDER_SRC = 13
SPINE3_SRC = 9
LEFT_COLLAR_DST = 12
RIGHT_COLLAR_DST = 13

AXIS_SWAP_ORDER = (2, 0, 1)  # new_x <- old_z, new_y <- old_x, new_z <- old_y
AXIS_SWAP_SIGN = np.array([1.0, 1.0, 1.0], dtype=np.float32)
UP_AXIS = 2


def decode_names(raw: np.ndarray) -> List[str]:
    out = []
    for item in raw:
        if isinstance(item, bytes):
            out.append(item.decode("utf-8"))
        elif isinstance(item, np.bytes_):
            out.append(item.tobytes().decode("utf-8"))
        else:
            out.append(str(item))
    return out


def normalize_vectors(v: np.ndarray, fallback: np.ndarray | None = None, eps: float = 1e-8) -> np.ndarray:
    norms = np.linalg.norm(v, axis=-1, keepdims=True)
    safe = np.where(norms < eps, 1.0, norms)
    out = v / safe
    if fallback is not None:
        mask = norms[..., 0] < eps
        if np.any(mask):
            fb = np.asarray(fallback, dtype=np.float32)
            if fb.ndim == 1:
                out[mask] = fb
            else:
                out[mask] = fb[mask]
    return out


def qnormalize(q: np.ndarray) -> np.ndarray:
    return q / np.clip(np.linalg.norm(q, axis=-1, keepdims=True), 1e-8, None)


def qinv(q: np.ndarray) -> np.ndarray:
    out = q.copy()
    out[..., 1:] *= -1.0
    return out


def qmul(q: np.ndarray, r: np.ndarray) -> np.ndarray:
    # Hamilton product for quaternions in wxyz order.
    w1, x1, y1, z1 = np.split(q, 4, axis=-1)
    w2, x2, y2, z2 = np.split(r, 4, axis=-1)
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.concatenate([w, x, y, z], axis=-1)


def qrot(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    qvec = q[..., 1:]
    uv = np.cross(qvec, v)
    uuv = np.cross(qvec, uv)
    return v + 2.0 * (q[..., :1] * uv + uuv)


def qbetween(v0: np.ndarray, v1: np.ndarray) -> np.ndarray:
    v0n = normalize_vectors(v0, fallback=np.array([1.0, 0.0, 0.0], dtype=np.float32))
    v1n = normalize_vectors(v1, fallback=np.array([1.0, 0.0, 0.0], dtype=np.float32))
    cross = np.cross(v0n, v1n)
    dot = np.sum(v0n * v1n, axis=-1, keepdims=True)
    q = np.concatenate([1.0 + dot, cross], axis=-1)

    small = np.linalg.norm(q, axis=-1) < 1e-8
    if np.any(small):
        bad_v = v0n[small]
        ortho = np.zeros_like(bad_v)
        choose_x = np.abs(bad_v[:, 0]) < np.abs(bad_v[:, 2])
        ortho[choose_x] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        ortho[~choose_x] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        axis = normalize_vectors(np.cross(bad_v, ortho), fallback=np.array([0.0, 1.0, 0.0], dtype=np.float32))
        q[small, 0] = 0.0
        q[small, 1:] = axis
    return qnormalize(q.astype(np.float32))


def qfix(q: np.ndarray) -> np.ndarray:
    out = q.copy()
    for t in range(1, out.shape[0]):
        dots = np.sum(out[t] * out[t - 1], axis=-1)
        flip = dots < 0.0
        if np.any(flip):
            out[t, flip] *= -1.0
    return out


def quat_from_axis_angle_wxyz(axis: np.ndarray, angle: np.ndarray) -> np.ndarray:
    """axis: (...,3), angle: (...) -> quaternion (...,4) in wxyz."""
    axis = normalize_vectors(axis, fallback=np.array([0.0, 0.0, 1.0], dtype=np.float32))
    half = 0.5 * angle
    s = np.sin(half)
    quat = np.concatenate([np.cos(half)[..., None], axis * s[..., None]], axis=-1)
    return qnormalize(quat.astype(np.float32))


def solve_two_direction_quat_wxyz(
    ref1: np.ndarray,
    tgt1: np.ndarray,
    ref2: np.ndarray,
    tgt2: np.ndarray,
) -> np.ndarray:
    """
    Solve a rotation (wxyz) aligning two direction pairs in batch.
    ref*/tgt*: (T, 3)
    """
    q1 = qbetween(ref1, tgt1)
    ref2_after_q1 = qrot(q1, ref2)

    axis = normalize_vectors(tgt1, fallback=np.array([1.0, 0.0, 0.0], dtype=np.float32))
    proj_a = ref2_after_q1 - axis * np.sum(ref2_after_q1 * axis, axis=-1, keepdims=True)
    proj_b = tgt2 - axis * np.sum(tgt2 * axis, axis=-1, keepdims=True)
    proj_a = normalize_vectors(proj_a, fallback=np.array([0.0, 1.0, 0.0], dtype=np.float32))
    proj_b = normalize_vectors(proj_b, fallback=np.array([0.0, 1.0, 0.0], dtype=np.float32))

    cross = np.cross(proj_a, proj_b)
    sin_term = np.sum(cross * axis, axis=-1)
    cos_term = np.sum(proj_a * proj_b, axis=-1)
    twist_angle = np.arctan2(sin_term, cos_term).astype(np.float32)

    q_twist = quat_from_axis_angle_wxyz(axis, twist_angle)
    return qnormalize(qmul(q_twist, q1))


def quat_wxyz_to_mat(q: np.ndarray) -> np.ndarray:
    qn = qnormalize(q.astype(np.float64))
    w, x, y, z = np.split(qn, 4, axis=-1)

    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    m00 = 1.0 - 2.0 * (yy + zz)
    m01 = 2.0 * (xy - wz)
    m02 = 2.0 * (xz + wy)
    m10 = 2.0 * (xy + wz)
    m11 = 1.0 - 2.0 * (xx + zz)
    m12 = 2.0 * (yz - wx)
    m20 = 2.0 * (xz - wy)
    m21 = 2.0 * (yz + wx)
    m22 = 1.0 - 2.0 * (xx + yy)

    row0 = np.concatenate([m00, m01, m02], axis=-1)
    row1 = np.concatenate([m10, m11, m12], axis=-1)
    row2 = np.concatenate([m20, m21, m22], axis=-1)
    return np.stack([row0, row1, row2], axis=-2).astype(np.float32)


def mat_to_quat_xyzw(mats: np.ndarray) -> np.ndarray:
    m = mats.reshape(-1, 3, 3).astype(np.float64)
    qw = np.empty(m.shape[0], dtype=np.float64)
    qx = np.empty_like(qw)
    qy = np.empty_like(qw)
    qz = np.empty_like(qw)

    trace = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]
    c1 = trace > 0
    t = np.sqrt(np.maximum(trace[c1] + 1.0, 1e-12)) * 2
    qw[c1] = 0.25 * t
    qx[c1] = (m[c1, 2, 1] - m[c1, 1, 2]) / t
    qy[c1] = (m[c1, 0, 2] - m[c1, 2, 0]) / t
    qz[c1] = (m[c1, 1, 0] - m[c1, 0, 1]) / t

    c2 = (m[:, 0, 0] > m[:, 1, 1]) & (m[:, 0, 0] > m[:, 2, 2]) & (~c1)
    t = np.sqrt(np.maximum(1.0 + m[c2, 0, 0] - m[c2, 1, 1] - m[c2, 2, 2], 1e-12)) * 2
    qw[c2] = (m[c2, 2, 1] - m[c2, 1, 2]) / t
    qx[c2] = 0.25 * t
    qy[c2] = (m[c2, 0, 1] + m[c2, 1, 0]) / t
    qz[c2] = (m[c2, 0, 2] + m[c2, 2, 0]) / t

    c3 = (m[:, 1, 1] > m[:, 2, 2]) & (~c1) & (~c2)
    t = np.sqrt(np.maximum(1.0 + m[c3, 1, 1] - m[c3, 0, 0] - m[c3, 2, 2], 1e-12)) * 2
    qw[c3] = (m[c3, 0, 2] - m[c3, 2, 0]) / t
    qx[c3] = (m[c3, 0, 1] + m[c3, 1, 0]) / t
    qy[c3] = 0.25 * t
    qz[c3] = (m[c3, 1, 2] + m[c3, 2, 1]) / t

    c4 = (~c1) & (~c2) & (~c3)
    t = np.sqrt(np.maximum(1.0 + m[c4, 2, 2] - m[c4, 0, 0] - m[c4, 1, 1], 1e-12)) * 2
    qw[c4] = (m[c4, 1, 0] - m[c4, 0, 1]) / t
    qx[c4] = (m[c4, 0, 2] + m[c4, 2, 0]) / t
    qy[c4] = (m[c4, 1, 2] + m[c4, 2, 1]) / t
    qz[c4] = 0.25 * t

    quat = np.stack([qx, qy, qz, qw], axis=-1)
    quat /= np.clip(np.linalg.norm(quat, axis=-1, keepdims=True), 1e-8, None)
    return quat.reshape(mats.shape[:-2] + (4,)).astype(np.float32)


def build_traversal(parents: Sequence[int]) -> List[int]:
    children: List[List[int]] = [[] for _ in parents]
    root = 0
    for child, parent in enumerate(parents):
        if parent < 0:
            root = child
            continue
        children[parent].append(child)

    order = []
    queue = [root]
    while queue:
        node = queue.pop(0)
        order.append(node)
        queue.extend(children[node])
    return order


def infer_local_quat_wxyz(
    joints: np.ndarray,
    parents: np.ndarray,
    raw_offset_dirs: np.ndarray,
    right_hip: int,
    left_hip: int,
    right_shoulder: int,
    left_shoulder: int,
    up_axis: int,
    canonical_forward: np.ndarray,
) -> np.ndarray:
    t, j, _ = joints.shape
    if j != len(parents):
        raise ValueError(f"Joint count mismatch: joints={j}, parents={len(parents)}")

    children: List[List[int]] = [[] for _ in range(j)]
    for child_idx, parent_idx in enumerate(parents.tolist()):
        if parent_idx >= 0:
            children[parent_idx].append(child_idx)

    across = (joints[:, right_hip] - joints[:, left_hip]) + (
        joints[:, right_shoulder] - joints[:, left_shoulder]
    )
    across = normalize_vectors(across, fallback=np.array([1.0, 0.0, 0.0], dtype=np.float32))

    up = np.zeros(3, dtype=np.float32)
    up[up_axis] = 1.0
    forward = np.cross(np.broadcast_to(up, across.shape), across)
    forward = normalize_vectors(forward, fallback=canonical_forward.astype(np.float32))

    canon = np.broadcast_to(canonical_forward[None, :], forward.shape).astype(np.float32)
    root_global = qbetween(canon, forward)

    # Face flip: rotate an extra 180 degrees around vertical axis.
    flip_axis = np.zeros((t, 3), dtype=np.float32)
    flip_axis[:, up_axis] = 1.0
    root_flip = quat_from_axis_angle_wxyz(flip_axis, np.full((t,), np.pi, dtype=np.float32))
    root_global = qmul(root_flip, root_global)

    local = np.zeros((t, j, 4), dtype=np.float32)
    global_q = np.zeros_like(local)
    local[:, 0] = root_global
    global_q[:, 0] = root_global

    traversal = build_traversal(parents.tolist())
    for idx in traversal:
        if idx == 0:
            continue
        p = parents[idx]
        parent_global = global_q[:, p]
        parent_inv = qinv(parent_global)

        # Match this repo's FK semantics: parent rotation explains vectors from
        # current joint to its children.
        cands = children[idx]
        if len(cands) == 0:
            local[:, idx] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        elif len(cands) == 1:
            c = cands[0]
            ref = np.broadcast_to(raw_offset_dirs[c][None, :], (t, 3)).astype(np.float32)
            ref = normalize_vectors(ref, fallback=np.array([1.0, 0.0, 0.0], dtype=np.float32))

            tgt_world = joints[:, c] - joints[:, idx]
            tgt_world = normalize_vectors(tgt_world, fallback=ref[0])
            tgt_local = qrot(parent_inv, tgt_world)
            tgt_local = normalize_vectors(tgt_local, fallback=ref[0])
            # Hip joints only have one child (knee), so twist around the thigh
            # axis is under-constrained. Use an extra constraint (parent->hip)
            # to stabilize hip facing direction without changing leg positions.
            if idx == left_hip or idx == right_hip:
                ref_side = np.broadcast_to(raw_offset_dirs[idx][None, :], (t, 3)).astype(np.float32)
                ref_side = normalize_vectors(ref_side, fallback=np.array([0.0, 1.0, 0.0], dtype=np.float32))
                tgt_side_world = joints[:, idx] - joints[:, p]
                tgt_side_world = normalize_vectors(tgt_side_world, fallback=ref_side[0])
                tgt_side_local = qrot(parent_inv, tgt_side_world)
                tgt_side_local = normalize_vectors(tgt_side_local, fallback=ref_side[0])
                local[:, idx] = solve_two_direction_quat_wxyz(ref, tgt_local, ref_side, tgt_side_local)
            else:
                local[:, idx] = qbetween(ref, tgt_local)
        else:
            c1, c2 = cands[:2]
            ref1 = np.broadcast_to(raw_offset_dirs[c1][None, :], (t, 3)).astype(np.float32)
            ref2 = np.broadcast_to(raw_offset_dirs[c2][None, :], (t, 3)).astype(np.float32)
            ref1 = normalize_vectors(ref1, fallback=np.array([1.0, 0.0, 0.0], dtype=np.float32))
            ref2 = normalize_vectors(ref2, fallback=np.array([0.0, 1.0, 0.0], dtype=np.float32))

            tgt1_world = normalize_vectors(joints[:, c1] - joints[:, idx], fallback=ref1[0])
            tgt2_world = normalize_vectors(joints[:, c2] - joints[:, idx], fallback=ref2[0])
            tgt1_local = normalize_vectors(qrot(parent_inv, tgt1_world), fallback=ref1[0])
            tgt2_local = normalize_vectors(qrot(parent_inv, tgt2_world), fallback=ref2[0])
            local[:, idx] = solve_two_direction_quat_wxyz(ref1, tgt1_local, ref2, tgt2_local)

        global_q[:, idx] = qmul(parent_global, local[:, idx])

    return qfix(local)


def load_template(template_h5: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    with h5py.File(template_h5, "r") as f:
        parents = f["joint_parents_idx"][:].astype(np.int64)
        offsets = f["joint_offsets"][:].astype(np.float32)
        t_pose = f["T_pose"][:].astype(np.float32)
        end_sites = f["end_sites"][:].astype(np.int64)
        names = decode_names(f["joint_names"][:])
    return parents, offsets, t_pose, end_sites, names


def swap_axes(arr: np.ndarray) -> np.ndarray:
    return arr[..., AXIS_SWAP_ORDER] * AXIS_SWAP_SIGN


def map_22_to_20_with_collar(positions22: np.ndarray, collar_ratio_l: float, collar_ratio_r: float) -> np.ndarray:
    t = positions22.shape[0]
    out = np.zeros((t, 20, 3), dtype=np.float32)

    for src, dst in SRC_TO_DST_DIRECT.items():
        out[:, dst, :] = positions22[:, src, :]

    spine3 = positions22[:, SPINE3_SRC, :]
    l_sh = positions22[:, LEFT_SHOULDER_SRC, :]
    r_sh = positions22[:, RIGHT_SHOULDER_SRC, :]
    out[:, LEFT_COLLAR_DST, :] = spine3 + collar_ratio_l * (l_sh - spine3)
    out[:, RIGHT_COLLAR_DST, :] = spine3 + collar_ratio_r * (r_sh - spine3)
    return out


def estimate_offsets_from_motion(
    joints: np.ndarray,
    parents: np.ndarray,
    template_offsets: np.ndarray,
) -> np.ndarray:
    offsets = np.zeros_like(template_offsets, dtype=np.float32)
    dirs = normalize_vectors(template_offsets, fallback=np.array([0.0, 1.0, 0.0], dtype=np.float32))
    templ_len = np.linalg.norm(template_offsets, axis=-1)

    for idx, p in enumerate(parents):
        if p < 0:
            offsets[idx] = np.zeros(3, dtype=np.float32)
            continue
        vec = joints[:, idx] - joints[:, p]
        norm = np.linalg.norm(vec, axis=-1)
        valid = norm[np.isfinite(norm) & (norm > 1e-5)]
        if valid.size > 0:
            length = float(np.median(valid))
        else:
            length = float(templ_len[idx])
        offsets[idx] = dirs[idx] * max(length, 1e-4)
    return offsets


def build_t_pose(parents: np.ndarray, offsets: np.ndarray, root_pos: np.ndarray) -> np.ndarray:
    t_pose = np.zeros_like(offsets, dtype=np.float32)
    root = int(np.where(parents < 0)[0][0])
    t_pose[root] = root_pos.astype(np.float32)
    order = build_traversal(parents.tolist())
    for idx in order:
        p = parents[idx]
        if p < 0:
            continue
        t_pose[idx] = t_pose[p] + offsets[idx]
    return t_pose


def write_h5(
    out_path: str,
    rotat_xyzw: np.ndarray,
    parents: np.ndarray,
    offsets: np.ndarray,
    t_pose: np.ndarray,
    joint_names: Sequence[str],
    end_sites: np.ndarray,
    joint_translation: np.ndarray,
    fps: float,
) -> None:
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with h5py.File(out_path, "w") as f:
        f.attrs["fps"] = float(fps)
        f.create_dataset("rotat", data=rotat_xyzw.astype(np.float32))
        f.create_dataset("joint_parents_idx", data=parents.astype(np.int64))
        f.create_dataset("joint_offsets", data=offsets.astype(np.float32))
        f.create_dataset("end_sites", data=end_sites.astype(np.int64))
        f.create_dataset("T_pose", data=t_pose.astype(np.float32))
        f.create_dataset("joint_names", data=np.array([n.encode("utf-8") for n in joint_names]))
        f.create_dataset("joint_translation", data=joint_translation.astype(np.float32))


def collect_npy_files(input_path: str) -> List[str]:
    p = Path(input_path)
    if p.is_file():
        if p.suffix.lower() != ".npy":
            raise ValueError(f"Input file must be .npy: {input_path}")
        return [str(p)]
    if p.is_dir():
        files = sorted(glob.glob(str(p / "**" / "*.npy"), recursive=True))
        if not files:
            raise ValueError(f"No .npy files found under: {input_path}")
        return files
    raise ValueError(f"Input path not found: {input_path}")


def resolve_out_path(input_path: str, input_root: str, output_path: str, single_input: bool) -> str:
    out = Path(output_path)
    in_p = Path(input_path)

    if single_input and out.suffix.lower() == ".h5":
        return str(out)

    if single_input:
        out_dir = out
        out_dir.mkdir(parents=True, exist_ok=True)
        return str(out_dir / f"{in_p.stem}.h5")

    out_dir = out
    rel = os.path.relpath(input_path, input_root)
    out_file = out_dir / Path(rel).with_suffix(".h5")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    return str(out_file)


def process_file(
    npy_path: str,
    out_path: str,
    parents: np.ndarray,
    template_offsets: np.ndarray,
    template_t_pose: np.ndarray,
    end_sites: np.ndarray,
    joint_names: Sequence[str],
    collar_ratio_l: float,
    collar_ratio_r: float,
    fps: float,
) -> None:
    motion = np.load(npy_path)
    if motion.ndim != 3 or motion.shape[1:] != (22, 3):
        raise ValueError(f"{npy_path}: expected shape (T, 22, 3), got {motion.shape}")

    joints20 = map_22_to_20_with_collar(motion.astype(np.float32), collar_ratio_l, collar_ratio_r)
    joints20 = swap_axes(joints20)

    offsets = estimate_offsets_from_motion(joints20, parents, template_offsets)
    raw_offset_dirs = normalize_vectors(offsets, fallback=np.array([0.0, 1.0, 0.0], dtype=np.float32))

    name2idx = {n: i for i, n in enumerate(joint_names)}
    right_hip = name2idx["R_Hip"]
    left_hip = name2idx["L_Hip"]
    right_shoulder = name2idx["R_Shoulder"]
    left_shoulder = name2idx["L_Shoulder"]

    template_across = (template_t_pose[right_hip] - template_t_pose[left_hip]) + (
        template_t_pose[right_shoulder] - template_t_pose[left_shoulder]
    )
    template_across = normalize_vectors(
        template_across[None, :], fallback=np.array([1.0, 0.0, 0.0], dtype=np.float32)
    )[0]
    up = np.zeros(3, dtype=np.float32)
    up[UP_AXIS] = 1.0
    canonical_forward = normalize_vectors(
        np.cross(up[None, :], template_across[None, :]),
        fallback=np.array([1.0, 0.0, 0.0], dtype=np.float32),
    )[0]

    local_wxyz = infer_local_quat_wxyz(
        joints20,
        parents,
        raw_offset_dirs,
        right_hip=right_hip,
        left_hip=left_hip,
        right_shoulder=right_shoulder,
        left_shoulder=left_shoulder,
        up_axis=UP_AXIS,
        canonical_forward=canonical_forward,
    )
    rotat_xyzw = mat_to_quat_xyzw(quat_wxyz_to_mat(local_wxyz))

    joint_translation = np.zeros((joints20.shape[0], len(joint_names), 3), dtype=np.float32)
    root_trans = joints20[:, 0, :].copy()
    ground_offset = float(np.min(joints20[0, :, UP_AXIS]))
    root_trans[:, UP_AXIS] -= ground_offset
    joint_translation[:, 0, :] = root_trans

    t_pose = build_t_pose(parents, offsets, root_pos=np.zeros(3, dtype=np.float32))
    t_pose[:, UP_AXIS] -= t_pose[:, UP_AXIS].min()

    write_h5(
        out_path,
        rotat_xyzw,
        parents,
        offsets,
        t_pose,
        joint_names,
        end_sites,
        joint_translation,
        fps,
    )
    print(f"Saved {out_path} ({joints20.shape[0]} frames, {fps:g} FPS)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert MCM-LDM (T,22,3) npy motions to Pose2Motion h5.")
    ap.add_argument(
        "--input",
        required=True,
        help="Input .npy file or directory containing .npy files.",
    )
    ap.add_argument(
        "--output",
        required=True,
        help="Output .h5 path (single input) or output directory (directory input).",
    )
    ap.add_argument(
        "--template_h5",
        default="examples/human_motion.h5",
        help="Template h5 used for joint names / hierarchy / end_sites.",
    )
    ap.add_argument(
        "--fps",
        type=float,
        default=20.0,
        help="Frame rate stored in each output h5.",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing h5 files.",
    )
    args = ap.parse_args()

    if not os.path.exists(args.template_h5):
        raise SystemExit(f"template_h5 not found: {args.template_h5}")

    parents, template_offsets, template_t_pose, end_sites, joint_names = load_template(args.template_h5)
    if len(joint_names) != 20:
        raise SystemExit(
            f"template_h5 must have 20-joint human skeleton, got {len(joint_names)} joints: {args.template_h5}"
        )

    lc = np.linalg.norm(template_offsets[LEFT_COLLAR_DST])
    ls = np.linalg.norm(template_offsets[14])
    rc = np.linalg.norm(template_offsets[RIGHT_COLLAR_DST])
    rs = np.linalg.norm(template_offsets[15])
    collar_ratio_l = float(lc / max(lc + ls, 1e-8))
    collar_ratio_r = float(rc / max(rc + rs, 1e-8))

    npy_files = collect_npy_files(args.input)
    single_input = Path(args.input).is_file()

    for npy in npy_files:
        out_path = resolve_out_path(
            input_path=npy,
            input_root=args.input if not single_input else str(Path(npy).parent),
            output_path=args.output,
            single_input=single_input,
        )
        if os.path.exists(out_path) and not args.overwrite:
            print(f"[skip] {out_path} exists")
            continue
        process_file(
            npy,
            out_path,
            parents=parents,
            template_offsets=template_offsets,
            template_t_pose=template_t_pose,
            end_sites=end_sites,
            joint_names=joint_names,
            collar_ratio_l=collar_ratio_l,
            collar_ratio_r=collar_ratio_r,
            fps=args.fps,
        )


if __name__ == "__main__":
    main()
