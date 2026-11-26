import unittest

import torch

from fabrics_sim.solver.dynamic_consistent_gpu import (
    MAX_CONTACT_DIM,
    NV,
    ProjectionConfig,
    dynamically_consistent_projection_masked,
)


class ContactStabilizationTest(unittest.TestCase):
    def test_joint_correction_cost_preserves_contact_without_mutating_mass(self):
        mass = torch.eye(NV, dtype=torch.float64)[None]
        jacobian = torch.zeros(1, MAX_CONTACT_DIM, NV, dtype=mass.dtype)
        jacobian[0, 0, 0] = jacobian[0, 0, 6] = 1.
        mask = torch.zeros(1, MAX_CONTACT_DIM, dtype=mass.dtype)
        mask[0, 0] = 1.
        acceleration = torch.zeros(1, NV, dtype=mass.dtype)
        acceleration[0, 6] = 2.
        kwargs = dict(acc_fabric=acceleration, mass_matrix=mass,
                      stacked_contact_jacobian=jacobian, active_row_mask=mask,
                      stacked_jdot_v=torch.zeros_like(mask))
        ordinary, _, _ = dynamically_consistent_projection_masked(**kwargs,
            config=ProjectionConfig(regularization=1e-10))
        weighted, _, residual = dynamically_consistent_projection_masked(**kwargs,
            config=ProjectionConfig(regularization=1e-10, joint_correction_weight=10.))
        self.assertLess(float(residual.abs().max()), 1e-8)
        self.assertLess(float((weighted[0, 6]-2).abs()), float((ordinary[0, 6]-2).abs()))
        torch.testing.assert_close(mass, torch.eye(NV, dtype=mass.dtype)[None])
        for invalid in (-1., float('inf'), float('nan')):
            with self.assertRaises(ValueError):
                dynamically_consistent_projection_masked(**kwargs,
                    config=ProjectionConfig(joint_correction_weight=invalid))

    def test_mask_and_acceleration_limit_apply_per_foot(self):
        dtype = torch.float64
        mass = torch.eye(NV, dtype=dtype).unsqueeze(0)
        jacobian = torch.zeros((1, MAX_CONTACT_DIM, NV), dtype=dtype)
        jacobian[0, :, :MAX_CONTACT_DIM] = torch.eye(
            MAX_CONTACT_DIM, dtype=dtype
        )
        velocity = torch.ones((1, NV), dtype=dtype)
        pose_error = torch.full((1, MAX_CONTACT_DIM), 2.0, dtype=dtype)

        _, residual_before, _ = dynamically_consistent_projection_masked(
            acc_fabric=torch.zeros((1, NV), dtype=dtype),
            mass_matrix=mass,
            stacked_contact_jacobian=jacobian,
            stacked_jdot_v=torch.zeros((1, MAX_CONTACT_DIM), dtype=dtype),
            active_row_mask=torch.ones((1, MAX_CONTACT_DIM), dtype=dtype),
            generalized_velocity=velocity,
            stacked_contact_pose_error=pose_error,
            config=ProjectionConfig(
                contact_kp=3.0,
                contact_kd=2.0,
                contact_stabilization_mask=(1.0, 1.0, 0.0, 0.0, 0.0, 0.0),
                contact_stabilization_max_acceleration=5.0,
            ),
        )

        expected = torch.zeros((1, MAX_CONTACT_DIM), dtype=dtype)
        expected[:, (0, 1, 6, 7)] = 5.0
        torch.testing.assert_close(residual_before, expected)


if __name__ == "__main__":
    unittest.main()
