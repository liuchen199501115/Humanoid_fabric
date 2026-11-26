import sys
import unittest
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'examples'))
from g1_acceleration_cost import linear_acceleration_weight, preserve_com_acceleration, track_com_feedback, com_feedback_scale


class AccelerationCostTest(unittest.TestCase):
    def test_horizontal_feedback_ignores_vertical_demand_and_cross_weights(self):
        dtype = torch.float64
        reference = torch.tensor([[2., 4., 6.]], dtype=dtype)
        weights = torch.ones(3, dtype=dtype)
        state = dict(com_jac=torch.eye(3, dtype=dtype)[None],
                     com_jac_bias=torch.zeros(1, 3, dtype=dtype),
                     com_acc_desired=torch.tensor([[8., -2., -1000.]], dtype=dtype),
                     com_acc_metric=torch.tensor([[[2., 0., 1.], [0., 2., 1.], [1., 1., 2.]]], dtype=dtype))
        original = state['com_acc_metric'].clone()
        W, target = track_com_feedback(weights, reference, state, 1., axes='xy')
        torch.testing.assert_close(W, torch.diag(torch.tensor([3., 3., 1.], dtype=dtype))[None])
        torch.testing.assert_close(target, torch.tensor([[6., 0., 6.]], dtype=dtype))
        torch.testing.assert_close(state['com_acc_metric'], original)
        state['com_acc_desired'][:, 2] = 1000.
        W2, target2 = track_com_feedback(weights, reference, state, 1., axes='xy')
        torch.testing.assert_close(W, W2);torch.testing.assert_close(target, target2)
        with self.assertRaises(ValueError):
            track_com_feedback(weights, reference, state, 1., axes='z')

    def test_feedback_ramp_retains_prefix_and_reaches_requested_metric(self):
        scale = lambda t: com_feedback_scale(4., t, start=.7, ramp=.1)
        self.assertEqual(scale(.6), 0.)
        self.assertEqual(scale(.7), 0.)
        self.assertAlmostEqual(scale(.75), 2.)
        self.assertEqual(scale(.9), 4.)
        self.assertEqual(com_feedback_scale(4., 0.), 4.)
        for option in ('scale', 'time', 'start', 'ramp'):
            for invalid in (-1., float('nan'), float('inf')):
                args = dict(scale=1., time=0., start=0., ramp=0.)
                args[option] = invalid
                with self.assertRaises(ValueError):
                    com_feedback_scale(**args)

    def test_complete_fabric_metric_matches_task_objective_under_constraints(self):
        from test_g1 import solve_single_tracking_qp
        from unload_qp import G1DynamicsAccelerationQP
        from fabrics_sim.fabrics.task_request import TaskRequest
        dtype = torch.float64
        I = torch.eye(35, dtype=dtype)[None]
        J = torch.zeros(1, 1, 35, dtype=dtype)
        J[:, :, 6:8] = 1.
        tasks = [TaskRequest('posture', I, torch.zeros(1, 35, dtype=dtype),
                             torch.full((1, 35), 8., dtype=dtype), I,
                             torch.ones(1, 35, dtype=torch.bool)),
                 TaskRequest('com', J, torch.ones(1, 1, dtype=dtype),
                             torch.full((1, 1), -3., dtype=dtype),
                             torch.full((1, 1, 1), 100., dtype=dtype),
                             torch.ones(1, 1, dtype=torch.bool))]
        H = sum(r.natural_form()[0] for r in tasks)
        f = sum(r.natural_form()[1] for r in tasks)
        center = torch.linalg.solve(H, -f[..., None]).squeeze(-1)
        kwargs = dict(acc_reference=center, mass_matrix=I, bias_force=torch.zeros_like(center),
                      contact_jacobian=torch.zeros(1, 12, 35, dtype=dtype),
                      jdot_qdot=torch.zeros(1, 12, dtype=dtype), contact_mode=torch.zeros(1, dtype=torch.long),
                      effort_limits=.6, acc_lower=-10., acc_upper=10.)
        full = G1DynamicsAccelerationQP(solver_backend='daqp', acc_weight=H)
        leaves = G1DynamicsAccelerationQP(solver_backend='daqp')
        full.solver.single_solve = leaves.solver.single_solve = True
        a = solve_single_tracking_qp(full, **kwargs)
        b = leaves.solve(**kwargs, task_requests=tasks)
        self.assertTrue(a.validated.all() and b.validated.all())
        torch.testing.assert_close(a.acc, b.acc, atol=1e-7, rtol=0.)
        torch.testing.assert_close(a.joint_torque, b.joint_torque, atol=1e-7, rtol=0.)
        self.assertEqual(full.solver.last_solve_counts, [1])
        self.assertIsNone(a.task_residuals)

    def test_original_feedback_survives_opposite_projection_and_hard_effort_limit(self):
        from unittest.mock import patch
        from test_g1 import solve_single_tracking_qp
        from unload_qp import G1DynamicsAccelerationQP
        dtype = torch.float64
        J = torch.zeros(1, 3, 35, dtype=dtype)
        J[0, 1, 6:8] = 1.
        state = dict(com_jac=J, com_jac_bias=torch.tensor([[0., 1., 0.]], dtype=dtype),
                     com_acc_desired=torch.tensor([[0., -3., 0.]], dtype=dtype),
                     com_acc_metric=100*torch.eye(3, dtype=dtype)[None])
        projected = torch.full((1, 35), 8., dtype=dtype)
        weight = torch.ones(35, dtype=dtype)
        W, target = track_com_feedback(weight, projected, state, 1.)
        solver = G1DynamicsAccelerationQP(solver_backend='daqp', acc_weight=W)
        solver.solver.single_solve = True
        with patch.object(solver.solver.daqp, 'solve', wraps=solver.solver.daqp.solve) as native:
            result = solve_single_tracking_qp(solver, acc_reference=target,
                mass_matrix=torch.eye(35, dtype=dtype)[None], bias_force=torch.zeros_like(projected),
                contact_jacobian=torch.zeros(1, 12, 35, dtype=dtype),
                jdot_qdot=torch.zeros(1, 12, dtype=dtype), contact_mode=torch.zeros(1, dtype=torch.long),
                effort_limits=.6, acc_lower=-10., acc_upper=10.)
        self.assertTrue(result.validated.all())
        self.assertEqual(native.call_count, 1)
        # The original PD requests negative acceleration despite +8 projection.
        torch.testing.assert_close(result.acc[:, 6:8], torch.full((1, 2), -.6, dtype=dtype), atol=1e-6, rtol=0.)
        self.assertLessEqual(float(result.joint_torque.abs().max()), .600001)
        self.assertIsNone(result.task_residuals)
        # Disabling the objective is an exact identity, including dtype/storage.
        old_W, old_ref = track_com_feedback(weight, projected, None, 0.)
        self.assertIs(old_W, weight)
        self.assertIs(old_ref, projected)
        for scale in (-1., float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                track_com_feedback(weight, projected, state, scale)

    def test_delayed_linear_weight_keeps_prefix_and_has_smooth_endpoints(self):
        weight = lambda t: linear_acceleration_weight(100., 1000., t, start=.7, ramp=.2)
        self.assertEqual(weight(0.), 100.)
        self.assertEqual(weight(.7), 100.)
        self.assertAlmostEqual(weight(.8), 550.)
        self.assertEqual(weight(1.), 1000.)
        # A quintic transition has zero first and second derivatives at both ends.
        epsilon = 1e-4
        for endpoint, direction in ((.7, 1), (.9, -1)):
            coarse = abs(weight(endpoint+direction*epsilon)-weight(endpoint))
            fine = abs(weight(endpoint+direction*epsilon/2)-weight(endpoint))
            self.assertAlmostEqual(coarse/fine, 8., delta=.02)
        for time in (0., .5, 2.):
            self.assertEqual(linear_acceleration_weight(100., None, time), 100.)
            self.assertEqual(linear_acceleration_weight(100., 300., time), 300.)

    def test_invalid_linear_cost_is_rejected(self):
        for value in (-1., 0., float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                linear_acceleration_weight(100., value, 0.)
        for name in ('start', 'ramp'):
            for value in (-1., float('nan'), float('inf')):
                with self.assertRaises(ValueError):
                    linear_acceleration_weight(100., 300., 0., **{name: value})

    def test_cost_matches_com_acceleration_error_and_preserves_input(self):
        g = torch.Generator().manual_seed(7)
        J = torch.randn(2, 3, 35, generator=g, dtype=torch.float64)
        delta = torch.randn(2, 35, generator=g, dtype=torch.float64)
        weights = torch.linspace(1., 5., 35, dtype=torch.float64)
        W = preserve_com_acceleration(weights, J, 100.)
        measured = (delta[:, None] @ W @ delta[:, :, None]).flatten()
        expected = (weights*delta.square()).sum(-1) + 100.*(J@delta[..., None]).square().sum((1,2))
        torch.testing.assert_close(measured, expected)
        self.assertTrue(bool((torch.linalg.eigvalsh(W) > 0).all()))
        torch.testing.assert_close(weights, torch.linspace(1., 5., 35, dtype=torch.float64))
        self.assertIs(preserve_com_acceleration(weights, J, 0.), weights)
        for value in (-1., float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                preserve_com_acceleration(weights, J, value)
