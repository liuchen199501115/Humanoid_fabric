import unittest

import torch

from fabrics_sim.solver.AccelerationQP import G1DynamicsAccelerationQP
from fabrics_sim.solver.joint_control import (
    joint_limit_avoidance_request,
    measured_joint_acceleration_bounds,
)


class JointPositionLimitTests(unittest.TestCase):
    def test_soft_task_pushes_inward_at_both_limits_and_is_idle_in_middle(self):
        q = torch.zeros(1, 29, dtype=torch.float64)
        q[0, :2] = torch.tensor([-.99, .99])
        lower, upper = torch.full_like(q, -1.), torch.ones_like(q)
        task = joint_limit_avoidance_request(q, torch.zeros_like(q), lower, upper, 20.)
        self.assertGreater(task.desired[0, 0].item(), 0.)
        self.assertLess(task.desired[0, 1].item(), 0.)
        self.assertTrue(task.active[0, :2].all())
        self.assertFalse(task.active[0, 2:].any())
        self.assertEqual(task.A[:, :, :6].count_nonzero().item(), 0)

    def test_qp_angle_bounds_override_outward_task_with_servo_feedback(self):
        for sign in (-1., 1.):
            with self.subTest(sign=sign):
                q = torch.zeros(1, 29, dtype=torch.float64)
                v = torch.zeros_like(q)
                q[0, 0], v[0, 0] = sign * .985, sign * .5
                lower, upper = torch.full_like(q, -1.), torch.ones_like(q)
                dt = .02
                lo, hi = measured_joint_acceleration_bounds(q, v, lower, upper, dt)
                physical_lo = q.new_full((1, 35), -float('inf'))
                physical_hi = q.new_full((1, 35), float('inf'))
                physical_lo[:, 6:], physical_hi[:, 6:] = lo, hi
                # A strong tracking request pushes outward despite the soft limit task.
                outward = joint_limit_avoidance_request(q, v, lower, upper, 20.)
                outward.desired[:] = sign * 100.
                outward.metric[:] = 20. * torch.eye(29, dtype=q.dtype)
                outward.active[:] = True
                result = G1DynamicsAccelerationQP(solver_backend='daqp').solve(
                    acc_reference=q.new_zeros(1, 35),
                    mass_matrix=torch.eye(35, dtype=q.dtype)[None],
                    bias_force=q.new_zeros(1, 35),
                    contact_jacobian=q.new_zeros(1, 12, 35),
                    jdot_qdot=q.new_zeros(1, 12), contact_mode=torch.tensor([0]),
                    acc_lower=-float('inf'), acc_upper=float('inf'), effort_limits=200.,
                    predicted_acc_lower=physical_lo, predicted_acc_upper=physical_hi,
                    servo_bias=torch.full_like(q, sign * 8.),
                    servo_acc_gain=torch.full_like(q, .5), task_requests=[outward])
                self.assertTrue(result.validated.all())
                next_q = q + dt * (v + dt * result.predicted_acc[:, 6:])
                self.assertTrue((next_q >= lower + .01 - 1e-6).all())
                self.assertTrue((next_q <= upper - .01 + 1e-6).all())
                self.assertLess(sign * result.predicted_acc[0, 6].item(), 0.)


if __name__ == '__main__':
    unittest.main()
