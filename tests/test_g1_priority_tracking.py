import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'examples'))
from g1_priority_tracking import compose_tracking_policy, scale_secondary_tasks
from fabrics_sim.fabrics.task_request import TaskRequest


class ContactPriorityTest(unittest.TestCase):
    def test_secondary_scaling_preserves_selected_rows_and_psd(self):
        A = torch.eye(35, dtype=torch.float64)[None, :2]
        W = torch.tensor([[[2., 1.], [1., 2.]]], dtype=A.dtype)
        r = TaskRequest('feet_pose', A, torch.zeros(1, 2, dtype=A.dtype),
            torch.zeros(1, 2, dtype=A.dtype), W, torch.ones(1, 2, dtype=torch.bool))
        low = scale_secondary_tasks([r], ['feet_pose'], {'feet_pose': (0,)}, .01)[0]
        torch.testing.assert_close(low.metric, torch.tensor([[[2., .1], [.1, .02]]], dtype=A.dtype))
        self.assertGreater(torch.linalg.eigvalsh(low.metric).min().item(), 0.)
        torch.testing.assert_close(r.metric, W)

    def test_physical_tasks_remain_consistent_with_affine_contact(self):
        D = torch.eye(35, dtype=torch.float64)[None]
        Jc = D[:, 6:7]
        A = Jc + D[:, 7:8]
        B = D[:, 7:8] + D[:, 8:9]
        def task(name, J, target):
            return TaskRequest(name, J, torch.zeros(1, 1, dtype=D.dtype),
                torch.tensor([[target]], dtype=D.dtype), torch.ones(1, 1, 1, dtype=D.dtype),
                torch.ones(1, 1, dtype=torch.bool))
        high, low = task('center_of_mass', A, 3.), task('joint_posture', B, 5.)
        a, H, projected, diagnostic = compose_tracking_policy([high, low], D, Jc,
            torch.tensor([[-1.]], dtype=D.dtype), torch.ones(1, 1, dtype=torch.bool),
            'contact-tracking', regularization=1e-8)
        torch.testing.assert_close(a[:, 6:9], torch.tensor([[1., 2., 3.]], dtype=D.dtype), atol=1e-6, rtol=0.)
        torch.testing.assert_close(a, projected, atol=1e-6, rtol=0.)
        self.assertLess(high.residual(a).abs().max().item(), 1e-6)
        self.assertLess(low.residual(a).abs().max().item(), 1e-6)
        self.assertGreater(torch.linalg.eigvalsh(H).min().item(), 0.)
        self.assertLess(diagnostic['fabric_priority_contact_error'].item(), 1e-12)

    def test_flight_and_inactive_contact_rows_do_not_pin_acceleration(self):
        D = torch.eye(35, dtype=torch.float64)[None]
        values = torch.ones(1, 35, dtype=D.dtype)
        task = TaskRequest('joint_posture', D, values*0, values, D, values.bool())
        a, H, projected, _ = compose_tracking_policy([task], D, D[:, :12],
            values[:, :12]*100., torch.zeros(1, 12, dtype=torch.bool), 'none')
        torch.testing.assert_close(a, projected)
        torch.testing.assert_close(a, values/1.0001)

    def test_composed_policy_uses_one_native_qp_with_hard_contact_and_effort(self):
        from unittest.mock import patch
        from test_g1 import solve_single_tracking_qp
        from unload_qp import G1DynamicsAccelerationQP
        D = torch.eye(35, dtype=torch.float64)[None]
        target = torch.zeros(1, 35, dtype=D.dtype)
        target[:, 6:9] = torch.tensor([1., 2., 8.])
        leaf = TaskRequest('joint_posture', D, target*0, target, D, torch.ones_like(target, dtype=torch.bool))
        Jc = torch.zeros(1, 12, 35, dtype=D.dtype)
        Jc[:, 0, 6] = 1.
        bias = torch.zeros(1, 12, dtype=D.dtype)
        bias[:, 0] = -1.
        mask = torch.zeros(1, 12, dtype=torch.bool)
        mask[:, 0] = True
        a, H, _, _ = compose_tracking_policy([leaf], D, Jc, bias, mask, 'none')
        solver = G1DynamicsAccelerationQP(solver_backend='daqp', acc_weight=H)
        solver.solver.single_solve = True
        solver.torque_weight = solver.wrench_weight = 0.
        with patch.object(solver.solver.daqp, 'solve', wraps=solver.solver.daqp.solve) as native:
            result = solve_single_tracking_qp(solver, acc_reference=a, mass_matrix=D,
                bias_force=torch.zeros_like(a), contact_jacobian=Jc, jdot_qdot=bias,
                contact_mode=torch.ones(1, dtype=torch.long), contact_row_mask=mask,
                effort_limits=3., acc_lower=-10., acc_upper=10.)
        self.assertTrue(result.validated.all())
        self.assertEqual(native.call_count, 1)
        self.assertAlmostEqual(result.acc[0, 6].item(), 1., places=6)
        self.assertAlmostEqual(result.acc[0, 8].item(), 3., places=5)
        self.assertLessEqual(result.joint_torque.abs().max().item(), 3.000001)
        self.assertIsNone(result.task_residuals)
        self.assertEqual(solver.last_priority_problems, [])


if __name__ == '__main__':
    unittest.main()
