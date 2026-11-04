from pathlib import Path

import numpy as np


def _matrix_to_euler_zxy_degrees(matrix):
    r = np.asarray(matrix)
    x = np.arcsin(np.clip(r[..., 2, 1], -1.0, 1.0))
    cx = np.cos(x)
    z = np.where(
        np.abs(cx) > 1e-8,
        np.arctan2(-r[..., 0, 1], r[..., 1, 1]),
        np.arctan2(r[..., 1, 0], r[..., 0, 0]),
    )
    y = np.where(np.abs(cx) > 1e-8, np.arctan2(-r[..., 2, 0], r[..., 2, 2]), 0.0)
    return np.rad2deg(np.stack([z, x, y], axis=-1))


def _write_bvh(parent, offset, rotation, position, names, frame_time, order, path):
    frame = rotation.shape[0]
    order = order.upper()
    file_string = "HIERARCHY\n"
    seq = []

    def write_static(idx, prefix):
        nonlocal file_string
        seq.append(idx)
        if idx == 0:
            name_label = "ROOT " + names[idx]
            channel_label = (
                "CHANNELS 6 Xposition Yposition Zposition "
                f"{order[0]}rotation {order[1]}rotation {order[2]}rotation"
            )
        else:
            name_label = "JOINT " + names[idx]
            channel_label = f"CHANNELS 3 {order[0]}rotation {order[1]}rotation {order[2]}rotation"
        offset_label = "OFFSET %.6f %.6f %.6f" % (offset[idx][0], offset[idx][1], offset[idx][2])

        file_string += prefix + name_label + "\n"
        file_string += prefix + "{\n"
        file_string += prefix + "\t" + offset_label + "\n"
        file_string += prefix + "\t" + channel_label + "\n"

        has_child = False
        for child in range(idx + 1, rotation.shape[1]):
            if parent[child] == idx:
                has_child = True
                write_static(child, prefix + "\t")
        if not has_child and "_end" not in names[idx]:
            file_string += prefix + "\tEnd Site\n"
            file_string += prefix + "\t{\n"
            file_string += prefix + "\t\tOFFSET 0 0 0\n"
            file_string += prefix + "\t}\n"
        file_string += prefix + "}\n"

    write_static(0, "")
    file_string += f"MOTION\nFrames: {frame}\nFrame Time: {frame_time:.8f}\n"
    for i in range(frame):
        file_string += "%.6f %.6f %.6f " % (position[i][0], position[i][1], position[i][2])
        for j in range(rotation.shape[1]):
            idx = seq[j]
            file_string += "%.6f %.6f %.6f " % (
                rotation[i][idx][0],
                rotation[i][idx][1],
                rotation[i][idx][2],
            )
        file_string += "\n"
    Path(path).write_text(file_string)


def save_bvh(rotation, info, filename, translation=None, normalize=True, fps=30.0):
    offset = np.asarray(info["joint_offsets"], dtype=np.float64)
    parent = np.asarray(info["joint_parents_idx_full"], dtype=np.int64)
    names = list(info["joint_names"])
    t_pose = info["T_pose"].detach().cpu().numpy() if hasattr(info["T_pose"], "detach") else info["T_pose"]

    if names and names[-1] == "root_velocity_node":
        rotation = rotation[:, :-1]
        offset = offset[:-1]
        parent = parent[:-1]
        names = names[:-1]

    position = np.zeros((rotation.shape[0], 3), dtype=np.float64) if translation is None else np.asarray(translation)
    if normalize:
        scale = (np.max(t_pose, axis=0) - np.min(t_pose, axis=0)).mean()
        offset = offset / scale
        position = position / scale

    offset = offset[:, [0, 2, 1]]
    position = position[:, [0, 2, 1]]

    rot_m = np.asarray(rotation, dtype=np.float64)
    if rot_m.ndim == 3 and rot_m.shape[-1] == 9:
        rot_m = rot_m.reshape(rot_m.shape[0], rot_m.shape[1], 3, 3)
    swap = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    rot_m = np.einsum("ab,njbc,cd->njad", swap, rot_m, swap.T)

    if position.shape[0] > 0:
        rot0 = rot_m[0]
        joint_num = rot0.shape[0]
        pos0 = np.zeros((joint_num, 3), dtype=np.float64)
        ori0 = np.zeros((joint_num, 3, 3), dtype=np.float64)
        for j in range(joint_num):
            p = parent[j]
            if p < 0:
                ori0[j] = rot0[j]
                pos0[j] = position[0] + offset[j]
            else:
                ori0[j] = ori0[p] @ rot0[j]
                pos0[j] = pos0[p] + ori0[p] @ offset[j]
        position[:, 1] -= pos0[:, 1].min()

    euler = _matrix_to_euler_zxy_degrees(rot_m)
    _write_bvh(parent, offset, euler, position, names, 1.0 / fps, "ZXY", filename)
