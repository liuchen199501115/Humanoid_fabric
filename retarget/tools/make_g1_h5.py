#!/usr/bin/env python3
"""
Convert PHUMA g1 npy chunks into Pose-to-Motion B-domain h5 files.

Input (PHUMA g1):
    Each .npy is a dict with keys:
        root_trans: (T, 3)
        root_ori:   (T, 4) quaternion (x,y,z,w)
        dof_pos:    (T, 29) joint angles in radians (g1 order)
        fps:        int

Output (per .npy):
    h5 with datasets:
        rotat: (T, 30, 4)  # root + 29 joints, quaternions
        joint_parents_idx: (30,)
        joint_offsets: (30, 3)
        end_sites: (4,)
        T_pose: (30, 3)
        joint_names: (30,) bytes
        joint_translation: (T, 30, 3)  # only root filled

Usage:
    python tools/make_g1_h5.py --phuma_dir /path/to/PHUMA/data/g1/EgoBody --out_dir data/g1_h5
"""

import argparse
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
os.environ.setdefault("KMP_AFFINITY", "disabled")
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")

import glob
import xml.etree.ElementTree as ET
import numpy as np
import h5py


def rpy_to_matrix(rpy):
    roll, pitch, yaw = rpy
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def axis_angle_to_matrix_batch(axis, angles):
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / (np.linalg.norm(axis) + 1e-8)
    angles = np.asarray(angles, dtype=np.float64).reshape(-1)
    x, y, z = axis
    ca = np.cos(angles)
    sa = np.sin(angles)
    C = 1 - ca
    mats = np.empty((angles.shape[0], 3, 3), dtype=np.float64)
    mats[:, 0, 0] = ca + x * x * C
    mats[:, 0, 1] = x * y * C - z * sa
    mats[:, 0, 2] = x * z * C + y * sa
    mats[:, 1, 0] = y * x * C + z * sa
    mats[:, 1, 1] = ca + y * y * C
    mats[:, 1, 2] = y * z * C - x * sa
    mats[:, 2, 0] = z * x * C - y * sa
    mats[:, 2, 1] = z * y * C + x * sa
    mats[:, 2, 2] = ca + z * z * C
    return mats


def matmul3(A, B):
    C = np.empty((3, 3), dtype=np.float64)
    C[0, 0] = A[0, 0] * B[0, 0] + A[0, 1] * B[1, 0] + A[0, 2] * B[2, 0]
    C[0, 1] = A[0, 0] * B[0, 1] + A[0, 1] * B[1, 1] + A[0, 2] * B[2, 1]
    C[0, 2] = A[0, 0] * B[0, 2] + A[0, 1] * B[1, 2] + A[0, 2] * B[2, 2]

    C[1, 0] = A[1, 0] * B[0, 0] + A[1, 1] * B[1, 0] + A[1, 2] * B[2, 0]
    C[1, 1] = A[1, 0] * B[0, 1] + A[1, 1] * B[1, 1] + A[1, 2] * B[2, 1]
    C[1, 2] = A[1, 0] * B[0, 2] + A[1, 1] * B[1, 2] + A[1, 2] * B[2, 2]

    C[2, 0] = A[2, 0] * B[0, 0] + A[2, 1] * B[1, 0] + A[2, 2] * B[2, 0]
    C[2, 1] = A[2, 0] * B[0, 1] + A[2, 1] * B[1, 1] + A[2, 2] * B[2, 1]
    C[2, 2] = A[2, 0] * B[0, 2] + A[2, 1] * B[1, 2] + A[2, 2] * B[2, 2]
    return C


def matvec3(A, v):
    return np.array(
        [
            A[0, 0] * v[0] + A[0, 1] * v[1] + A[0, 2] * v[2],
            A[1, 0] * v[0] + A[1, 1] * v[1] + A[1, 2] * v[2],
            A[2, 0] * v[0] + A[2, 1] * v[1] + A[2, 2] * v[2],
        ],
        dtype=np.float64,
    )


def matrix_to_quat_batch(mats):
    m = np.asarray(mats, dtype=np.float64)
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

    return np.stack([qx, qy, qz, qw], axis=-1)


def parse_urdf(urdf_path):
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    rev_joints = [j for j in root.findall("joint") if j.get("type") == "revolute"]

    joint_names = ["pelvis"]
    joint_parents = [-1]
    joint_offsets = [np.zeros(3, dtype=np.float32)]
    base_rots = [np.eye(3, dtype=np.float64)]
    joint_axes = [np.array([0.0, 0.0, 1.0], dtype=np.float64)]
    link_to_idx = {"pelvis": 0}

    for j in rev_joints:
        name = j.get("name")
        parent_link = j.find("parent").get("link")
        child_link = j.find("child").get("link")
        origin = j.find("origin")
        xyz = (
            np.fromstring(origin.get("xyz"), sep=" ", dtype=np.float64)
            if origin is not None and origin.get("xyz") is not None
            else np.zeros(3, dtype=np.float64)
        )
        rpy = (
            np.fromstring(origin.get("rpy"), sep=" ", dtype=np.float64)
            if origin is not None and origin.get("rpy") is not None
            else np.zeros(3, dtype=np.float64)
        )
        axis = j.find("axis")
        axis_vec = (
            np.fromstring(axis.get("xyz"), sep=" ", dtype=np.float64)
            if axis is not None and axis.get("xyz") is not None
            else np.array([0.0, 0.0, 1.0], dtype=np.float64)
        )

        parent_idx = link_to_idx.get(parent_link, 0)
        joint_names.append(name)
        joint_parents.append(parent_idx)
        joint_offsets.append(xyz.astype(np.float32))
        base_rots.append(rpy_to_matrix(rpy))
        joint_axes.append(axis_vec)
        link_to_idx[child_link] = len(joint_names) - 1

    return joint_names, np.asarray(joint_parents, dtype=np.int64), np.stack(joint_offsets), base_rots, joint_axes


def compute_t_pose(parents, offsets, base_rots):
    joint_num = len(parents)
    positions = [np.zeros(3, dtype=np.float64) for _ in range(joint_num)]
    orientations = [np.eye(3, dtype=np.float64) for _ in range(joint_num)]

    for i in range(1, joint_num):
        p = parents[i]
        orientations[i] = matmul3(orientations[p], base_rots[i])
        positions[i] = positions[p] + matvec3(orientations[p], offsets[i])
    return np.stack(positions)


def compute_pose_with_angle_offsets(parents, offsets, base_rots, joint_axes, angle_offsets):
    """Static FK given per-joint angle offsets (len = J-1, rad)."""
    joint_num = len(parents)
    positions = [np.zeros(3, dtype=np.float64) for _ in range(joint_num)]
    orientations = [np.eye(3, dtype=np.float64) for _ in range(joint_num)]
    # root stays identity
    for i in range(1, joint_num):
        p = parents[i]
        R_axis = axis_angle_to_matrix_batch(joint_axes[i], angle_offsets[i - 1])[0]
        orientations[i] = matmul3(orientations[p], matmul3(base_rots[i], R_axis))
        positions[i] = positions[p] + matvec3(orientations[p], offsets[i])
    return np.stack(positions)


def write_h5(
    out_path,
    rotat,
    parents,
    offsets,
    t_pose,
    joint_names,
    end_sites,
    joint_translation,
    fps,
):
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with h5py.File(out_path, "w") as f:
        f.attrs["fps"] = float(fps)
        f.create_dataset("rotat", data=rotat.astype(np.float32))
        f.create_dataset("joint_parents_idx", data=parents.astype(np.int64))
        f.create_dataset("joint_offsets", data=offsets.astype(np.float32))
        f.create_dataset("end_sites", data=np.asarray(end_sites, dtype=np.int64))
        f.create_dataset("T_pose", data=t_pose.astype(np.float32))
        f.create_dataset("joint_names", data=np.array([n.encode("utf-8") for n in joint_names]))
        f.create_dataset("joint_translation", data=joint_translation.astype(np.float32))


def process_file(npy_path, out_dir, joint_names, parents, offsets, base_rots, joint_axes, phuma_base, angle_offsets=None):
    data = np.load(npy_path, allow_pickle=True).item()
    if "fps" not in data:
        raise ValueError(f"{npy_path}: missing fps")
    fps = float(data["fps"])
    angles = data["dof_pos"]  # (T,29)
    root_translation = data["root_trans"]  # (T,3)
    root_quat = data["root_ori"]  # (T,4)
    T = angles.shape[0]
    if angles.shape[1] != len(joint_names) - 1:
        raise ValueError(f"{npy_path}: expected angles shape [T,29], got {angles.shape}")

    rotat = np.zeros((T, len(joint_names), 4), dtype=np.float64)
    rotat[:, 0, :] = root_quat

    for j in range(angles.shape[1]):
        base_rot = base_rots[j + 1]
        axis = joint_axes[j + 1]
        angle_seq = angles[:, j]  # do not add offsets to motion
        R_axis = axis_angle_to_matrix_batch(axis, angle_seq)
        R_joint = np.empty_like(R_axis)
        for t in range(T):
            R_joint[t] = matmul3(base_rot, R_axis[t])
        rotat[:, j + 1, :] = matrix_to_quat_batch(R_joint)

    joint_translation = np.zeros((T, len(joint_names), 3), dtype=np.float64)
    joint_translation[:, 0, :] = root_translation

    if angle_offsets is not None:
        t_pose = compute_pose_with_angle_offsets(parents, offsets, base_rots, joint_axes, angle_offsets)
    else:
        t_pose = compute_t_pose(parents, offsets, base_rots)
    # align T-pose to ground: shift so lowest z is at 0
    min_z = t_pose[:, 2].min()
    t_pose[:, 2] -= min_z
    end_sites = [6, 12, 22, 29]  # feet, feet, left wrist yaw, right wrist yaw

    base_name = os.path.splitext(os.path.basename(npy_path))[0]
    # keep directory structure: recording_xxx/body_idx_y
    rel_dir = os.path.relpath(os.path.dirname(npy_path), phuma_base)
    out_dir_full = os.path.join(out_dir, rel_dir)
    os.makedirs(out_dir_full, exist_ok=True)
    out_path = os.path.join(out_dir_full, f"{base_name}.h5")
    write_h5(
        out_path,
        rotat,
        parents,
        offsets,
        t_pose,
        joint_names,
        end_sites,
        joint_translation,
        fps,
    )
    print(
        f"Saved {out_path} with {T} frames, "
        f"{len(joint_names)} joints, {fps:g} FPS."
    )


def main():
    parser = argparse.ArgumentParser(description="Convert PHUMA g1 npy chunks to Pose-to-Motion h5 files.")
    parser.add_argument("--urdf", default="assets/g1_description/g1_29dof.urdf", help="Path to G1 URDF")
    parser.add_argument("--phuma_dir", required=True, help="PHUMA g1 base directory containing recordings/body_idx_*/chunk npy")
    parser.add_argument("--out_dir", default="data/g1_h5", help="Directory to save h5 files")
    args = parser.parse_args()

    joint_names, parents, offsets, base_rots, joint_axes = parse_urdf(args.urdf)
    # Hardcoded angle offsets (radians) for specific joints
    offset_map = {
        "left_elbow_joint": 1.6,
        "left_shoulder_roll_joint": 1.6,
        "right_elbow_joint": 1.6,
        "right_shoulder_roll_joint": -1.6,
    }
    angle_offsets = np.zeros(len(joint_names) - 1, dtype=np.float64)
    for idx, name in enumerate(joint_names[1:]):  # skip root
        angle_offsets[idx] = float(offset_map.get(name, 0.0))

    npy_files = sorted(glob.glob(os.path.join(args.phuma_dir, "**", "*.npy"), recursive=True))
    if not npy_files:
        raise SystemExit(f"No npy files found under {args.phuma_dir}")

    for npy_path in npy_files:
        process_file(npy_path, args.out_dir, joint_names, parents, offsets, base_rots, joint_axes, args.phuma_dir, angle_offsets)


if __name__ == "__main__":
    main()
