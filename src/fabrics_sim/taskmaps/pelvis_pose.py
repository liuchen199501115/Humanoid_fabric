import torch

from fabrics_sim.taskmaps.maps_base import BaseMap


def normalize_quaternion_xyzw(quaternion):
    """Normalize quaternions stored in xyzw order."""
    return quaternion / torch.linalg.norm(
        quaternion,
        dim=-1,
        keepdim=True,
    ).clamp_min(1.0e-8)


def quaternion_xyzw_to_matrix(quaternion):
    """
    Convert quaternion in xyzw order to rotation matrix.

    quaternion: (..., 4)
    return: (..., 3, 3)
    """
    quaternion = normalize_quaternion_xyzw(quaternion)

    x = quaternion[..., 0]
    y = quaternion[..., 1]
    z = quaternion[..., 2]
    w = quaternion[..., 3]

    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    rotation = torch.stack(
        [
            1.0 - 2.0 * (yy + zz),
            2.0 * (xy - wz),
            2.0 * (xz + wy),
            2.0 * (xy + wz),
            1.0 - 2.0 * (xx + zz),
            2.0 * (yz - wx),
            2.0 * (xz - wy),
            2.0 * (yz + wx),
            1.0 - 2.0 * (xx + yy),
        ],
        dim=-1,
    )

    return rotation.reshape(*quaternion.shape[:-1], 3, 3)


def skew_symmetric(vector):
    """Return the skew-symmetric matrix [v]_x for (..., 3) vectors."""
    x, y, z = vector.unbind(dim=-1)
    zero = torch.zeros_like(x)
    return torch.stack(
        [
            zero, -z, y,
            z, zero, -x,
            -y, x, zero,
        ],
        dim=-1,
    ).reshape(*vector.shape[:-1], 3, 3)


def so3_log_map(rotation, eps=1.0e-7):
    """
    Calculate Log(R)^vee.

    rotation: (..., 3, 3)
    return: (..., 3)
    """
    trace = (
        rotation[..., 0, 0]
        + rotation[..., 1, 1]
        + rotation[..., 2, 2]
    )

    cos_theta = 0.5 * (trace - 1.0)
    cos_theta = torch.clamp(cos_theta, -1.0 + eps, 1.0 - eps)

    theta = torch.acos(cos_theta)

    skew_vector = torch.stack(
        [
            rotation[..., 2, 1] - rotation[..., 1, 2],
            rotation[..., 0, 2] - rotation[..., 2, 0],
            rotation[..., 1, 0] - rotation[..., 0, 1],
        ],
        dim=-1,
    )

    sin_theta = torch.sin(theta)

    small_angle = theta.abs() < 1.0e-4

    regular_scale = theta / (2.0 * sin_theta.clamp_min(eps))
    small_scale = 0.5 + theta.square() / 12.0

    scale = torch.where(
        small_angle,
        small_scale,
        regular_scale,
    )

    return scale.unsqueeze(-1) * skew_vector


def so3_left_jacobian_inverse(rotation_vector, eps=1.0e-7):
    """Return the inverse SO(3) left Jacobian for world-frame errors."""
    theta = torch.linalg.vector_norm(
        rotation_vector,
        dim=-1,
    )
    phi_hat = skew_symmetric(rotation_vector)
    phi_hat_squared = torch.matmul(phi_hat, phi_hat)

    identity = torch.eye(
        3,
        dtype=rotation_vector.dtype,
        device=rotation_vector.device,
    ).expand(*rotation_vector.shape[:-1], 3, 3)

    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)
    denominator = 2.0 * theta * sin_theta
    safe_denominator = torch.where(
        denominator.abs() < eps,
        torch.ones_like(denominator),
        denominator,
    )
    coefficient_regular = (
        1.0 / theta.square().clamp_min(eps)
        - (1.0 + cos_theta) / safe_denominator
    )
    coefficient_small = (
        torch.full_like(theta, 1.0 / 12.0)
        + theta.square() / 720.0
    )
    coefficient = torch.where(
        theta.abs() < 1.0e-4,
        coefficient_small,
        coefficient_regular,
    )

    return (
        identity
        - 0.5 * phi_hat
        + coefficient[..., None, None] * phi_hat_squared
    )


class PelvisPoseTaskMap(BaseMap):
    """
    Pelvis position and SO(3) orientation-error task map.

    Generalized configuration:
        q = [p_base(3), quaternion_xyzw(4), q_joint(29)]
        q.shape = (batch_size, 36)

    Generalized velocity:
        nu = [v_base(3), omega_base(3), qd_joint(29)]
        nu.shape = (batch_size, 35)

    Task:
        x = [position_error(3), orientation_error(3)]

    Jacobian:
        J.shape = (batch_size, 6, 35)
    """

    def __init__(
        self,
        batch_size,
        num_joints,
        device,
        track_position=(True, True, True),
        track_orientation=(True, True, True),
    ):
        super().__init__(device)

        self.batch_size = batch_size
        self.num_joints = num_joints
        self.velocity_dim = 6 + num_joints

        self.position_target = torch.zeros(
            batch_size,
            3,
            device=device,
        )

        self.orientation_target = torch.zeros(
            batch_size,
            4,
            device=device,
        )
        self.orientation_target[:, 3] = 1.0

        self.position_mask = torch.tensor(
            track_position,
            dtype=torch.float32,
            device=device,
        )

        self.orientation_mask = torch.tensor(
            track_orientation,
            dtype=torch.float32,
            device=device,
        )

    def set_target(self, position_target, orientation_target):
        """
        position_target:
            (batch_size, 3)

        orientation_target:
            quaternion xyzw, (batch_size, 4)
        """
        if position_target.shape != (self.batch_size, 3):
            raise ValueError(
                "Pelvis position target must have shape "
                f"({self.batch_size}, 3), got {position_target.shape}."
            )

        if orientation_target.shape != (self.batch_size, 4):
            raise ValueError(
                "Pelvis orientation target must have shape "
                f"({self.batch_size}, 4), got {orientation_target.shape}."
            )

        self.position_target.copy_(position_target)

        orientation_target = normalize_quaternion_xyzw(
            orientation_target
        )

        # Make target quaternion sign consistent with the previous target.
        same_hemisphere = (
            torch.sum(
                orientation_target * self.orientation_target,
                dim=-1,
                keepdim=True,
            )
            >= 0.0
        )

        orientation_target = torch.where(
            same_hemisphere,
            orientation_target,
            -orientation_target,
        )

        self.orientation_target.copy_(orientation_target)

    def forward_position(self, q, features=None):
        if q.shape[1] != 7 + self.num_joints:
            raise ValueError(
                "Expected q = [position(3), quaternion_xyzw(4), joints], "
                f"with dimension {7 + self.num_joints}; "
                f"got {q.shape[1]}."
            )

        pelvis_position = q[:, 0:3]
        pelvis_quaternion = normalize_quaternion_xyzw(q[:, 3:7])

        current_rotation = quaternion_xyzw_to_matrix(
            pelvis_quaternion
        )

        target_rotation = quaternion_xyzw_to_matrix(
            self.orientation_target
        )

        position_error = (
            pelvis_position - self.position_target
        ) * self.position_mask

        # World-frame orientation error:
        #
        #   R_error = R_current R_target^T
        #
        # so that e_R is expressed in the world frame and matches
        # a world-frame angular Jacobian.
        rotation_error = torch.bmm(
            current_rotation,
            target_rotation.transpose(1, 2),
        )

        unmasked_orientation_error = so3_log_map(rotation_error)
        orientation_error = (
            unmasked_orientation_error * self.orientation_mask
        )

        task_error = torch.cat(
            [position_error, orientation_error],
            dim=-1,
        )

        jacobian = torch.zeros(
            q.shape[0],
            6,
            self.velocity_dim,
            dtype=q.dtype,
            device=q.device,
        )

        # Valid when pelvis is the floating-base root and base twist
        # is [world linear velocity, world angular velocity].
        jacobian[:, 0:3, 0:3] = torch.diag(
            self.position_mask
        )

        # For e = Log(R_current R_target^T), expressed in the world frame,
        # e_dot = J_l(e)^(-1) omega. Apply the tracking mask to task rows,
        # after evaluating J_l at the complete (unmasked) rotation error.
        jacobian[:, 3:6, 3:6] = (
            self.orientation_mask.reshape(1, 3, 1)
            * so3_left_jacobian_inverse(unmasked_orientation_error)
        )

        return task_error, jacobian
