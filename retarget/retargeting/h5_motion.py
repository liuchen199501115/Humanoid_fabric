from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch

from .rotation import matrix_to_rotation_6d, quaternion_xyzw_to_matrix


@dataclass
class MotionData:
    pose: torch.Tensor
    info: dict
    path: Path


def _decode_names(raw_names):
    return [
        name.decode("utf-8") if isinstance(name, (bytes, np.bytes_)) else str(name)
        for name in raw_names
    ]


def _get_height(end_list, parent_list, offsets):
    heights = []
    for idx in end_list:
        parent = int(parent_list[int(idx)])
        height = 0.0
        while parent != 0 and parent >= 0:
            height += float(np.linalg.norm(offsets[parent]))
            parent = int(parent_list[parent])
        heights.append(height)
    return heights


def _root_velocity(root_translation, use_global_y=False):
    velocity = np.zeros_like(root_translation, dtype=np.float32)
    velocity[:-1, :] = root_translation[1:, :] - root_translation[:-1, :]
    if use_global_y:
        velocity[:, 1] = root_translation[:, 1]
    return velocity


def _to_rotation_6d(raw_pose_xyzw: np.ndarray) -> torch.Tensor:
    pose = torch.from_numpy(raw_pose_xyzw.astype(np.float32))
    matrix = quaternion_xyzw_to_matrix(pose)
    return matrix_to_rotation_6d(matrix)


def load_motion_h5(
    path: str | Path,
    *,
    velocity_virtual_node: bool,
    use_global_y: bool,
    rotation_data: str = "rotation_6d",
    domain: str = "B",
    mode: str = "test",
) -> MotionData:
    if rotation_data != "rotation_6d":
        raise ValueError("This standalone release currently supports rotation_data=rotation_6d.")

    path = Path(path)
    with h5py.File(path, "r") as f:
        fps = f.attrs.get("fps")
        if fps is not None:
            fps = float(fps)
            if not np.isfinite(fps) or fps <= 0:
                raise ValueError(f"{path}: fps attribute must be positive, got {fps}")
        raw_pose = f["rotat"][:].astype(np.float32)
        joint_parents_idx = f["joint_parents_idx"][:].astype(np.int64)
        joint_offsets = f["joint_offsets"][:].astype(np.float32)
        end_sites = f["end_sites"][:].astype(np.int64)
        t_pose = torch.from_numpy(f["T_pose"][:].astype(np.float32))
        joint_names = _decode_names(f["joint_names"][:])
        joint_translation = f["joint_translation"][:].astype(np.float32)

    root_translation = joint_translation[:, 0, :]
    root_velocity = _root_velocity(root_translation, use_global_y=use_global_y)

    pose_for_virtual = raw_pose
    if velocity_virtual_node:
        root_velocity_node = np.zeros((raw_pose.shape[0], 1, raw_pose.shape[2]), dtype=np.float32)
        root_velocity_node[:, 0, :3] = root_velocity
        pose_for_virtual = np.concatenate([raw_pose, root_velocity_node], axis=1)
        joint_parents_idx = np.concatenate([joint_parents_idx, np.array([0], dtype=np.int64)])
        joint_offsets = np.concatenate([joint_offsets, np.zeros((1, 3), dtype=np.float32)], axis=0)
        t_pose = torch.cat([t_pose, torch.zeros((1, 3), dtype=torch.float32)], dim=0)
        joint_names = joint_names + ["root_velocity_node"]

    ref_idx = min(20, max(pose_for_virtual.shape[0] - 1, 0))
    pose_virtual_index = np.where(
        ((np.abs(pose_for_virtual[0, :] - pose_for_virtual[ref_idx, :])) < 1e-8).sum(axis=1) == 4
    )
    if velocity_virtual_node and domain.upper() == "A":
        pose_virtual_index = (pose_virtual_index[0][:-1],)

    if velocity_virtual_node:
        pose_rotation = _to_rotation_6d(raw_pose)
        root_node_6d = torch.zeros((raw_pose.shape[0], 1, 6), dtype=torch.float32)
        root_node_6d[:, 0, :3] = torch.from_numpy(root_velocity)
        pose = torch.cat([pose_rotation, root_node_6d], dim=1)
    else:
        pose = _to_rotation_6d(raw_pose)

    virtual_idx = np.asarray(pose_virtual_index[0], dtype=np.int64)
    pose_virtual_val = torch.zeros((pose.shape[1], pose.shape[2]), dtype=torch.float32)
    pose_virtual_val[virtual_idx] = pose[0, virtual_idx]
    virtual_mask = torch.ones((pose.shape[1], pose.shape[2]), dtype=torch.float32)
    virtual_mask[virtual_idx] = 0.0

    height_list = _get_height(end_sites, joint_parents_idx, joint_offsets)
    foot_height = float(t_pose[int(end_sites[3]), 2]) if len(end_sites) > 3 else 0.0
    info = {
        "joint_parents_idx": joint_parents_idx,
        "joint_parents_idx_full": joint_parents_idx,
        "joint_offsets": joint_offsets,
        "end_sites": end_sites,
        "T_pose": t_pose,
        "foot_height": foot_height,
        "root_translation": root_translation,
        "joint_names": joint_names,
        "pose_virtual_index": pose_virtual_index,
        "pose_virtual_val": pose_virtual_val,
        "virtual_mask": virtual_mask,
        "height_list": np.asarray(height_list, dtype=np.float32),
        "height": float(height_list[0] + height_list[2]) if len(height_list) > 2 else 0.0,
        "foot_index": [0, 1],
        "x_range": [float(t_pose[:, 0].min()), float(t_pose[:, 0].max())],
        "y_range": [float(t_pose[:, 1].min()), float(t_pose[:, 1].max())],
        "z_range": [float(t_pose[:, 2].min()), float(t_pose[:, 2].max())],
        "fps": fps,
    }
    return MotionData(pose=pose, info=info, path=path)


def load_stats(path: Optional[str | Path], device: torch.device):
    if path is None:
        return None
    data = np.load(path, allow_pickle=True)
    return {
        "mean_A": torch.from_numpy(data["mean_A"]).float().to(device),
        "var_A": torch.from_numpy(data["var_A"]).float().to(device),
        "mean_B": torch.from_numpy(data["mean_B"]).float().to(device),
        "var_B": torch.from_numpy(data["var_B"]).float().to(device),
    }
