"""Count real DAQP calls, including failed certificates and warm starts."""
import contextlib
import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'examples'))
from test_g1 import parse_args, solve_single_tracking_qp
from unload_qp import G1DynamicsAccelerationQP
from fabrics_sim.fabrics.task_request import TaskRequest


class SingleTrackingQPTests(unittest.TestCase):
    def test_cold_reset_preserves_single_solve_and_original_optimum(self):
        solver, kwargs = self.problem()
        solver.solver.single_proximal = 1e-4
        first = solve_single_tracking_qp(solver, **kwargs)
        self.assertTrue(first.validated.all())
        self.assertTrue(solver.solver._previous_solutions)
        solver.solver.reset()
        with patch.object(solver.solver.daqp, 'solve', wraps=solver.solver.daqp.solve) as native:
            second = solve_single_tracking_qp(solver, **kwargs)
        self.assertEqual(native.call_count, 1)
        self.assertNotIn('primal_start', native.call_args.kwargs)
        self.assertEqual(native.call_args.kwargs['eps_prox'], 1e-4)
        self.assertTrue(second.validated.all())
        torch.testing.assert_close(first.acc, second.acc, atol=1e-8, rtol=0.)
        with patch.object(sys, 'argv', ['test_g1.py', '--tracking-profile', 'amp-full']):
            self.assertEqual(parse_args().qp_cold_start_time, float('inf'))

    def problem(self):
        dtype = torch.float64
        A = torch.zeros(1, 1, 35, dtype=dtype)
        A[0, 0, 6] = 1.
        tasks = [TaskRequest(name, A, torch.zeros(1, 1, dtype=dtype),
                            torch.tensor([[target]], dtype=dtype),
                            torch.tensor([[[weight]]], dtype=dtype),
                            torch.ones(1, 1, dtype=torch.bool))
                 for name, target, weight in [('balance', 2., 1.), ('pose', -2., 3.)]]
        solver = G1DynamicsAccelerationQP(solver_backend='daqp')
        solver.solver.single_solve = True
        kwargs = dict(acc_reference=torch.full((1, 35), 8., dtype=dtype),
                      mass_matrix=torch.eye(35, dtype=dtype)[None],
                      bias_force=torch.zeros(1, 35, dtype=dtype),
                      contact_jacobian=torch.zeros(1, 12, 35, dtype=dtype),
                      jdot_qdot=torch.zeros(1, 12, dtype=dtype),
                      contact_mode=torch.zeros(1, dtype=torch.long),
                      task_requests=tasks, priority_tasks=('balance',),
                      priority_levels=(('balance',),), priority_rows={'balance': (0,)},
                      effort_limits=10., acc_lower=-10., acc_upper=10.)
        return solver, kwargs

    def test_acceleration_reference_overrides_leaves_and_each_update_solves_once(self):
        solver, kwargs = self.problem()
        with patch.object(solver.solver, 'solve', wraps=solver.solver.solve) as backend:
            with patch.object(solver.solver.daqp, 'solve', wraps=solver.solver.daqp.solve) as native:
                for tick in range(2):
                    target = (8., -3.)[tick]
                    kwargs['acc_reference'].fill_(target)
                    kwargs['command_acc_correction'] = torch.ones_like(kwargs['acc_reference'])
                    result = solve_single_tracking_qp(solver, **kwargs)
                    self.assertTrue(result.validated.all())
                    self.assertAlmostEqual(float(result.acc[0, 6]), target, delta=1e-3)
                    torch.testing.assert_close(result.predicted_acc, result.acc)
                    self.assertEqual(backend.call_count, tick + 1)
                    self.assertEqual(native.call_count, tick + 1)
                    self.assertEqual(solver.solver.last_solve_counts, [1])
                    self.assertEqual(solver.last_priority_problems, [])
                    self.assertIsNone(solver.last_priority_errors)
                    self.assertIsNone(result.task_residuals)

    def test_effort_constraint_survives_removal_of_priority_stage(self):
        solver, kwargs = self.problem()
        kwargs['effort_limits'] = .6
        result = solve_single_tracking_qp(solver, **kwargs)
        self.assertTrue(result.validated.all())
        self.assertAlmostEqual(float(result.acc[0, 6]), .6, delta=1e-6)
        self.assertLessEqual(float(result.joint_torque.abs().max()), .6 + 1e-6)
        self.assertEqual(solver.solver.last_solve_counts, [1])

    def test_preselected_proximal_still_calls_native_once_and_keeps_hard_limits(self):
        solver, kwargs = self.problem()
        solver.solver.single_proximal = 1e-4
        solver.solver.max_iter = 50000
        kwargs['effort_limits'] = .6
        with patch.object(solver.solver.daqp, 'solve', wraps=solver.solver.daqp.solve) as native:
            result = solve_single_tracking_qp(solver, **kwargs)
        self.assertTrue(result.validated.all())
        self.assertEqual(native.call_count, 1)
        self.assertEqual(native.call_args.kwargs['eps_prox'], 1e-4)
        self.assertEqual(native.call_args.kwargs['eta_prox'], 1e-12)
        self.assertEqual(native.call_args.kwargs['iter_limit'], 50000)
        self.assertLessEqual(result.joint_torque.abs().max().item(), .600001)
        self.assertEqual(solver.last_priority_problems, [])

    def test_failed_native_result_is_rejected_without_retry_or_repair(self):
        for flag in (-1, 1):
            with self.subTest(flag=flag):
                solver, kwargs = self.problem()
                original = solver.solver.daqp.solve

                def bad_certificate(*args, **options):
                    x, objective, _, info = original(*args, **options)
                    # Preserve dynamics and remain feasible, but break KKT.
                    x[6] += 1e-4
                    x[47] += 1e-4
                    return x, objective, flag, info

                with patch.object(solver.solver.daqp, 'solve', side_effect=bad_certificate) as native:
                    with patch('fabrics_sim.solver.active_set_predictor.polish_dual_active_set') as repair:
                        result = solve_single_tracking_qp(solver, **kwargs)
                self.assertFalse(result.validated.any())
                self.assertEqual(native.call_count, 1)
                self.assertEqual(solver.solver.last_solve_counts, [1])
                repair.assert_not_called()

    def test_profile_disables_resolve_paths(self):
        with patch.object(sys, 'argv', ['test_g1.py', '--tracking-profile', 'amp-full']):
            args = parse_args()
        self.assertFalse(args.relax_joint_tube or args.recover_orientation_tube or args.shoulder_roll_guard)
        self.assertEqual(args.qp_solver, 'daqp')
        for extra in (['--relax-joint-tube'], ['--recover-orientation-tube'],
                      ['--shoulder-roll-guard'], ['--qp-solver', 'osqp']):
            with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()):
                with patch.object(sys, 'argv', ['test_g1.py', '--tracking-profile', 'amp-full', *extra]):
                    with self.assertRaises(SystemExit):
                        parse_args()


if __name__ == '__main__':
    unittest.main()
