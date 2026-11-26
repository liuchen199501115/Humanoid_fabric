"""Analytic slack optima and unchanged hard-constraint rejection."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'examples'))
from g1_soft_tracking_tube import SoftTrackingTubeBackend
from fabrics_sim.solver.admm_qp import QPData


class SoftTubeTests(unittest.TestCase):
    def problem(self, target=2.):
        H = torch.eye(35, dtype=torch.float64)[None]
        h = torch.zeros(1, 35, dtype=torch.float64)
        h[0, 6] = -target
        qp = QPData(H, h, H.new_zeros(1, 0, 35), h.new_zeros(1, 0),
                    torch.full_like(h, -10), torch.full_like(h, 10))
        C = H.new_zeros(1, 3, 35)
        C[0, 0, 6] = 1.
        return qp, (C, h.new_zeros(1, 3), h.new_ones(3))

    def test_soft_violation_matches_analytic_optimum_with_one_native_solve(self):
        qp, tube = self.problem()
        solver = SoftTrackingTubeBackend(3.)
        solver.tube = tube
        with patch.object(solver.daqp, 'solve', wraps=solver.daqp.solve) as native:
            result = solver.solve(qp)
        self.assertTrue(result.status_ok.all())
        self.assertEqual(native.call_count, 1)
        # min .5*(a-2)^2 + .5*3*s^2, a<=s, s>=0: a=s=.5.
        self.assertAlmostEqual(float(result.z[0, 6]), .5, places=6)
        self.assertAlmostEqual(float(solver.last_slack[0, 0]), .5, places=6)

    def test_hard_bound_still_wins_over_both_objectives(self):
        qp, tube = self.problem()
        qp.ub[0, 6] = .1
        solver = SoftTrackingTubeBackend(3.)
        solver.tube = tube
        result = solver.solve(qp)
        self.assertTrue(result.status_ok.all())
        self.assertAlmostEqual(float(result.z[0, 6]), .1, places=6)

    def test_joint_only_slack_keeps_root_tube_hard(self):
        qp, tube = self.problem()
        qp.h[0, 7] = -2.
        tube[0][0, 1, 7] = 1.
        solver = SoftTrackingTubeBackend(3., soft_rows=(True, False, False))
        solver.tube = tube
        result = solver.solve(qp)
        self.assertTrue(result.status_ok.all())
        self.assertAlmostEqual(result.z[0, 6].item(), .5, places=6)
        self.assertAlmostEqual(result.z[0, 7].item(), 0., places=6)
        torch.testing.assert_close(solver.last_slack[:, 1:], torch.zeros_like(solver.last_slack[:, 1:]), atol=1e-8, rtol=0.)
        self.assertEqual(solver.last_solve_counts, [1])

    def test_inactive_tube_does_not_pull_toward_boundary(self):
        qp, tube = self.problem(target=-2.)
        solver = SoftTrackingTubeBackend(3.)
        solver.tube = tube
        result = solver.solve(qp)
        self.assertTrue(result.status_ok.all())
        self.assertAlmostEqual(float(result.z[0, 6]), -2., places=6)
        torch.testing.assert_close(solver.last_slack, torch.zeros_like(solver.last_slack), atol=1e-7, rtol=0)

    def test_joint_and_orientation_slack_keeps_position_hard(self):
        qp, tube = self.problem()
        qp.h[0, 7:9] = -2.
        tube[0][0, 1, 7] = 1.
        tube[0][0, 2, 8] = 1.
        solver = SoftTrackingTubeBackend(3., soft_rows=(True, False, True))
        solver.tube = tube
        result = solver.solve(qp)
        self.assertTrue(result.status_ok.all())
        torch.testing.assert_close(result.z[0, 6:9],
            torch.tensor([.5, 0., .5], dtype=torch.float64), atol=1e-7, rtol=0.)
        self.assertAlmostEqual(float(solver.last_slack[0, 1]), 0., places=7)
        self.assertEqual(solver.last_solve_counts, [1])

    def test_infeasible_physics_cannot_use_tube_slack(self):
        qp, tube = self.problem()
        qp.A_eq = qp.H[:, 6:7].clone()
        qp.b_eq = qp.h.new_tensor([[2.]])
        qp.ub[0, 6] = .1
        solver = SoftTrackingTubeBackend(3.)
        solver.tube = tube
        result = solver.solve(qp)
        self.assertFalse(result.status_ok.any())
        self.assertEqual(solver.last_solve_counts, [1])
        torch.testing.assert_close(solver.last_augmented_problem.A_eq[:, :, :35], qp.A_eq)
        self.assertFalse(solver.last_augmented_problem.A_eq[:, :, 35:].any())

    def test_no_tube_is_exact_passthrough_and_clears_diagnostics(self):
        qp, tube = self.problem()
        solver = SoftTrackingTubeBackend(3.)
        solver.tube = tube
        solver.solve(qp)
        solver.tube = None
        result = solver.solve(qp)
        self.assertTrue(result.status_ok.all())
        self.assertAlmostEqual(float(result.z[0, 6]), 2., places=6)
        self.assertIsNone(solver.last_augmented_problem)
        self.assertIsNone(solver.last_slack)

    def test_full_dynamics_solver_keeps_torque_limits_and_one_solve(self):
        from test_g1 import solve_single_tracking_qp
        from unload_qp import G1DynamicsAccelerationQP
        _, tube = self.problem()
        solver = G1DynamicsAccelerationQP(solver_backend='daqp')
        solver.solver = SoftTrackingTubeBackend(3.)
        solver.solver.tube = tube
        result = solve_single_tracking_qp(solver,
            acc_reference=torch.full((1, 35), 8., dtype=torch.float64),
            mass_matrix=torch.eye(35, dtype=torch.float64)[None],
            bias_force=torch.zeros(1, 35, dtype=torch.float64),
            contact_jacobian=torch.zeros(1, 12, 35, dtype=torch.float64),
            jdot_qdot=torch.zeros(1, 12, dtype=torch.float64),
            contact_mode=torch.zeros(1, dtype=torch.long), effort_limits=.6)
        self.assertTrue(result.validated.all())
        self.assertLessEqual(float(result.joint_torque.abs().max()), .6 + 1e-6)
        self.assertLess(float(result.joint_dynamics_error.max()), 1e-6)
        self.assertEqual(solver.solver.last_solve_counts, [1])
        self.assertEqual(solver.last_priority_problems, [])
        torch.testing.assert_close(result.acc, result.predicted_acc)

    def test_defaults_retain_baseline_and_experiments_retain_source_auditing(self):
        from test_g1 import parse_args
        for extra in ([], ['--tracking-tube-soft-weight', '10', '--tracking-tube-reference', 'control']):
            with patch.object(sys, 'argv', ['test_g1.py', '--tracking-profile', 'amp-full', *extra]):
                args = parse_args()
            self.assertTrue(args.track_full_motion and args.stop_at_fidelity_loss)
            self.assertEqual(args.full_body_preview_start, float('inf'))
            self.assertEqual(args.centroidal_preview_start, float('inf'))
            if not extra:
                self.assertEqual(args.tracking_tube_soft_weight, 0.)
                self.assertEqual(args.tracking_tube_reference, 'source')


if __name__ == '__main__':
    unittest.main()
