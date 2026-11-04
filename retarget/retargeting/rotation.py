import torch
import torch.nn.functional as F


def quaternion_xyzw_to_matrix(quat: torch.Tensor) -> torch.Tensor:
    quat = quat.to(dtype=torch.float32)
    quat = quat / quat.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    x, y, z, w = quat.unbind(-1)
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
    return torch.stack(
        (m00, m01, m02, m10, m11, m12, m20, m21, m22), dim=-1
    ).reshape(quat.shape[:-1] + (3, 3))


def matrix_to_quaternion_xyzw(matrix: torch.Tensor) -> torch.Tensor:
    m = matrix
    q_abs = torch.stack(
        [
            1.0 + m[..., 0, 0] - m[..., 1, 1] - m[..., 2, 2],
            1.0 - m[..., 0, 0] + m[..., 1, 1] - m[..., 2, 2],
            1.0 - m[..., 0, 0] - m[..., 1, 1] + m[..., 2, 2],
            1.0 + m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2],
        ],
        dim=-1,
    ).clamp_min(0.0).sqrt()
    qx, qy, qz, qw = q_abs.unbind(-1)
    candidates = torch.stack(
        [
            torch.stack(
                [
                    qx * qx,
                    m[..., 0, 1] + m[..., 1, 0],
                    m[..., 0, 2] + m[..., 2, 0],
                    m[..., 2, 1] - m[..., 1, 2],
                ],
                dim=-1,
            ),
            torch.stack(
                [
                    m[..., 0, 1] + m[..., 1, 0],
                    qy * qy,
                    m[..., 1, 2] + m[..., 2, 1],
                    m[..., 0, 2] - m[..., 2, 0],
                ],
                dim=-1,
            ),
            torch.stack(
                [
                    m[..., 0, 2] + m[..., 2, 0],
                    m[..., 1, 2] + m[..., 2, 1],
                    qz * qz,
                    m[..., 1, 0] - m[..., 0, 1],
                ],
                dim=-1,
            ),
            torch.stack(
                [
                    m[..., 2, 1] - m[..., 1, 2],
                    m[..., 0, 2] - m[..., 2, 0],
                    m[..., 1, 0] - m[..., 0, 1],
                    qw * qw,
                ],
                dim=-1,
            ),
        ],
        dim=-2,
    )
    denom = (2.0 * q_abs).clamp_min(1e-8).unsqueeze(-1)
    quat_by_case = candidates / denom
    best = q_abs.argmax(dim=-1)
    quat = quat_by_case.gather(
        -2, best[..., None, None].expand(best.shape + (1, 4))
    ).squeeze(-2)
    return quat / quat.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    return matrix[..., :2, :].clone().reshape(matrix.shape[:-2] + (6,))


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    a1 = d6[..., 0:3]
    a2 = d6[..., 3:6]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def format_rotation(input_tensor: torch.Tensor, rotation: str) -> torch.Tensor:
    if input_tensor.shape[-1] == 9:
        matrix = input_tensor.reshape(input_tensor.shape[:-1] + (3, 3))
        if rotation == "quaternion":
            return matrix_to_quaternion_xyzw(matrix)
        if rotation == "rotation_6d":
            return matrix_to_rotation_6d(matrix)
    elif input_tensor.shape[-1] == 6:
        if rotation == "matrix":
            matrix = rotation_6d_to_matrix(input_tensor)
            return matrix.reshape(input_tensor.shape[:-1] + (9,))
        if rotation == "quaternion":
            return matrix_to_quaternion_xyzw(rotation_6d_to_matrix(input_tensor))
    elif input_tensor.shape[-1] == 4:
        matrix = quaternion_xyzw_to_matrix(input_tensor)
        if rotation == "matrix":
            return matrix.reshape(input_tensor.shape[:-1] + (9,))
        if rotation == "rotation_6d":
            return matrix_to_rotation_6d(matrix)
    return input_tensor


def to_matrix(pose: torch.Tensor) -> torch.Tensor:
    if pose.shape[-1] == 9:
        return pose
    if pose.shape[-1] == 6:
        matrix = rotation_6d_to_matrix(pose)
    elif pose.shape[-1] == 4:
        matrix = quaternion_xyzw_to_matrix(pose)
    else:
        raise ValueError(f"Unsupported rotation dimension: {pose.shape[-1]}")
    return matrix.reshape(pose.shape[:-1] + (9,))
