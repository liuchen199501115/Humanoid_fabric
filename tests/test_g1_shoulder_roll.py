"""Analytic checks for interval alignment, hierarchy replay and optional weights."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'examples'))
from analyze_g1_shoulder_roll import forward_acceleration, metric_delta, solve_two_stage
from g1_shoulder_roll import apply_shoulder_roll_weights, shoulder_roll_multiplier


class ShoulderRollTest(unittest.TestCase):
    def test_runtime_options_are_opt_in_and_preserve_full_source_audit(self):
        from test_g1 import parse_args
        for extra in ([], ['--tracking-profile', 'amp-full']):
            with patch.object(sys, 'argv', ['test_g1.py', *extra]):
                args = parse_args()
            self.assertEqual(args.shoulder_roll_weight_scale, 1.)
            self.assertEqual(args.shoulder_roll_weight_start, float('inf'))
        with patch.object(sys, 'argv', ['test_g1.py', '--tracking-profile', 'amp-full',
                         '--shoulder-roll-weight-scale', '4', '--shoulder-roll-weight-start', '3.3']):
            args = parse_args()
        self.assertEqual(args.shoulder_roll_weight_scale, 4.)
        self.assertTrue(args.track_full_motion and args.stop_at_fidelity_loss and args.direct_wbc)
        self.assertEqual(args.tracking_steps, 0)

    def test_invalid_experiment_rejected_before_simulation(self):
        from test_g1 import main
        for extra in (['--tracking-profile', 'amp-full', '--shoulder-roll-weight-scale', '4'],
                      ['--shoulder-roll-weight-scale', '4', '--shoulder-roll-weight-start', '3.3']):
            with patch.object(sys, 'argv', ['test_g1.py', *extra]), self.assertRaises(ValueError):
                main()

    def test_forward_interval_alignment_with_unequal_timesteps(self):
        t = np.array([0., .1, .3])
        v = np.array([[1., 2.], [1.2, 1.9], [2., 2.9]])
        np.testing.assert_allclose(forward_acceleration(t, v), [[2., -1.], [4., 5.]])
        with self.assertRaises(ValueError):
            forward_acceleration(np.array([0., 0., .3]), v)

    def test_default_and_smooth_ramp(self):
        self.assertEqual(shoulder_roll_multiplier(100., float('inf'), .1, 1.), 1.)
        self.assertEqual(shoulder_roll_multiplier(3.3, 3.3, .1, 4.), 1.)
        self.assertAlmostEqual(shoulder_roll_multiplier(3.35, 3.3, .1, 4.), 2.5)
        self.assertEqual(shoulder_roll_multiplier(3.5, 3.3, .1, 4.), 4.)
        for start, ramp, maximum in [(-1., .1, 4.), (float('nan'), .1, 4.),
                                     (3., 0., 4.), (3., .1, .5), (3., .1, float('inf'))]:
            with self.assertRaises(ValueError):
                shoulder_roll_multiplier(0., start, ramp, maximum)

    def test_metric_preserves_psd_and_matches_offline_counterfactual(self):
        names = ['other', 'right_shoulder_roll_joint', 'left_shoulder_roll_joint']
        basis = torch.tensor([[1., 2., -1.], [0., 3., 1.]], dtype=torch.float64)
        metric = basis.T @ basis
        request = SimpleNamespace(name='joint_posture', metric=metric[None].clone(),
                                  desired=torch.tensor([1., 2., 3.]))
        old_desired = request.desired.clone()
        original = request.metric
        apply_shoulder_roll_weights([request], names, 1.)
        self.assertIs(request.metric, original)
        apply_shoulder_roll_weights([request], names, 4., 'right')
        expected = np.diag([1., 2., 1.]) @ metric.numpy() @ np.diag([1., 2., 1.])
        np.testing.assert_allclose(request.metric[0], expected)
        self.assertGreaterEqual(float(torch.linalg.eigvalsh(request.metric)[0].min()), -1e-10)
        torch.testing.assert_close(request.desired, old_desired)
        np.testing.assert_allclose(metric.numpy() + metric_delta(metric.numpy(),
                                   np.ones(3, dtype=bool), [1], 4.), expected)

    def test_inactive_metric_rows_stay_inactive(self):
        M = np.array([[2., 1.], [1., 2.]])
        delta = metric_delta(M, np.array([True, False]), [1], 8.)
        np.testing.assert_array_equal(delta, np.zeros((2, 2)))

    def test_rebuilds_priority_locks_instead_of_reusing_old_targets(self):
        # Stage one: minimize .5*x² + .5*(x-2)² => x=1.
        # Add a 4x weight on the second task => x=1.6.
        # Stage two prefers x=0 and must reach the new lower lock 1.59.
        E = np.empty((0, 1)); e = np.empty(0)
        G = np.empty((0, 1)); b = np.empty(0)
        lo, hi = np.array([-10.]), np.array([10.])
        first = [np.array([[2.]]), np.array([-2.]), E, e, G, b, lo, hi]
        final = [np.array([[10.]]), np.array([0.]), E, e,
                 np.array([[1.], [-1.]]), np.array([1.01, -.99]), lo, hi]
        x, status = solve_two_stage(first, final, np.array([[3.]]), np.array([-6.]))
        self.assertTrue(status['accepted'])
        self.assertAlmostEqual(x[0], 1.59, places=6)
        # Input objectives, original bounds and original lock targets remain intact.
        np.testing.assert_array_equal(final[5], [1.01, -.99])
        np.testing.assert_array_equal(first[0], [[2.]])


if __name__ == '__main__':
    unittest.main()
