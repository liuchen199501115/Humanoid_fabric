"""World-expressed spatial shifts between rigid-body COMs and link origins."""
import torch


def skew(vector):
    x, y, z = vector.unbind(-1)
    zero = torch.zeros_like(x)
    return torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1).reshape(*vector.shape[:-1], 3, 3)


def origin_to_com_map(offset_world, angular_velocity, dimension=35):
    """v_COM = T v_origin; a_COM = T a_origin + beta.

    offset_world points from root link origin to its COM. Both velocities
    use world axes; beta is the centripetal acceleration of this offset.
    """
    batch = offset_world.shape[0]
    transform = torch.eye(dimension, device=offset_world.device, dtype=offset_world.dtype).expand(batch, -1, -1).clone()
    transform[:, :3, 3:6] = -skew(offset_world)
    bias = torch.zeros(batch, dimension, device=offset_world.device, dtype=offset_world.dtype)
    bias[:, :3] = torch.linalg.cross(angular_velocity, torch.linalg.cross(angular_velocity, offset_world))
    return transform, bias


def dynamics_at_origin(mass_com, bias_com, transform, acceleration_bias):
    transpose = transform.transpose(-1, -2)
    mass = transpose @ mass_com @ transform
    bias = (transpose @ (bias_com + (mass_com @ acceleration_bias[..., None]).squeeze(-1))[..., None]).squeeze(-1)
    return mass, bias


def jacobians_at_origins(jacobians_com, link_offsets_world, root_transform):
    """Shift both output link twist points and input root velocity coordinates."""
    shifted = jacobians_com.clone()
    shifted[..., :3, :] += skew(link_offsets_world) @ jacobians_com[..., 3:6, :]
    return shifted @ root_transform[:, None]
