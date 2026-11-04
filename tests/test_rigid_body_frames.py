import unittest

import torch

from fabrics_sim.solver.rigid_body_frames import (
    origin_to_com_map, dynamics_at_origin, jacobians_at_origins, skew,
)


class RigidBodyFrameTest(unittest.TestCase):
    def test_velocity_and_centripetal_acceleration_shift(self):
        offset = torch.tensor([[0., 0., -.076]], dtype=torch.float64)
        omega = torch.tensor([[0., 2., 0.]], dtype=torch.float64)
        T, beta = origin_to_com_map(offset, omega)
        velocity = torch.zeros(1, 35, dtype=torch.float64)
        velocity[:, 3:6] = omega
        com_velocity = (T @ velocity[..., None]).squeeze(-1)
        torch.testing.assert_close(com_velocity[:, :3], torch.tensor([[-.152, 0., 0.]], dtype=torch.float64))
        torch.testing.assert_close(beta[:, :3], torch.tensor([[0., 0., .304]], dtype=torch.float64))

    def test_dynamics_power_and_acceleration_equivalence(self):
        torch.manual_seed(12)
        offset, omega = torch.randn(2, 3, dtype=torch.float64), torch.randn(2, 3, dtype=torch.float64)
        T, beta = origin_to_com_map(offset, omega)
        matrix = torch.randn(2, 35, 35, dtype=torch.float64)
        mass = matrix.transpose(-1, -2) @ matrix + torch.eye(35, dtype=torch.float64)
        bias = torch.randn(2, 35, dtype=torch.float64)
        origin_mass, origin_bias = dynamics_at_origin(mass, bias, T, beta)
        acceleration = torch.randn(2, 35, dtype=torch.float64)
        com_acceleration = (T @ acceleration[..., None]).squeeze(-1) + beta
        force_com = (mass @ com_acceleration[..., None]).squeeze(-1) + bias
        force_origin = (origin_mass @ acceleration[..., None]).squeeze(-1) + origin_bias
        torch.testing.assert_close(force_origin, (T.transpose(-1, -2) @ force_com[..., None]).squeeze(-1))
        velocity = torch.randn(2, 35, dtype=torch.float64)
        torch.testing.assert_close((force_origin * velocity).sum(-1),
                                   (force_com * (T @ velocity[..., None]).squeeze(-1)).sum(-1))

    def test_link_and_root_shifts_restore_link_origin_jacobian(self):
        offset = torch.tensor([[[.02, 0., -.016], [0., .01, -.03]]], dtype=torch.float64)
        T, _ = origin_to_com_map(torch.tensor([[0., 0., -.076]], dtype=torch.float64), torch.zeros(1, 3, dtype=torch.float64))
        desired = torch.randn(1, 2, 6, 35, dtype=torch.float64)
        at_com = desired.clone()
        at_com[..., :3, :] -= skew(offset) @ desired[..., 3:, :]
        at_com = at_com @ torch.linalg.inv(T)[:, None]
        torch.testing.assert_close(jacobians_at_origins(at_com, offset, T), desired)
