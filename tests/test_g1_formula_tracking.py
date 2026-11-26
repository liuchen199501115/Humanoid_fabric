"""State-law invariants and matched-execution experiment guards (no Isaac launch)."""
import contextlib
import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'examples'))
from test_g1_2 import PhaseTrackingLaw, parse_args


class FormulaProfileTest(unittest.TestCase):
    def parse(self, *args):
        with patch.object(sys, 'argv', ['test_g1_2.py', *args]):
            return parse_args()

    def test_weight_ablation_changes_only_formula_and_output_options(self):
        baseline = vars(self.parse('--tracking-profile', 'twelve-wbc'))
        formula = vars(self.parse('--tracking-profile', 'formula-wbc', '--formula-scope', 'weights'))
        changed = {key for key in baseline if baseline[key] != formula[key]}
        self.assertEqual(changed, {'tracking_profile', 'formula_scope', 'output_path', 'tracking_margin_start'})
        self.assertTrue(formula['direct_wbc'] and formula['tracking_servo'])

    def test_phase_replaces_late_gates_and_keeps_full_acceptance(self):
        args = self.parse('--tracking-profile', 'formula-wbc')
        self.assertTrue(args.track_full_motion and args.stop_at_fidelity_loss)
        self.assertEqual(args.tracking_steps, 0)
        for key in ('physical_braking_start', 'position_tube_start', 'tracking_tube_start',
                    'landing_gated_unload_start', 'anticipate_unload_start', 'model_effort_start', 'clock_transition_start'):
            self.assertEqual(getattr(args, key), 0.)
        for key in ('tracking_margin_start', 'horizontal_feedback_start',
                    'global_posture_feedback_start', 'late_body_angular_metric_start',
                    'late_foot_metric_start'):
            self.assertEqual(getattr(args, key), float('inf'))
        self.assertEqual(args.clock_final_speed, 1.)
        from smooth_clock import smooth_clock
        for t in (0., .05, 1., 5., 18.):
            self.assertEqual(smooth_clock(t, args.clock_transition_start,
                args.clock_transition_ramp, args.clock_final_speed), (t, 1., 0.))

    def test_source_reference_excludes_prepared_pose_and_clearance_edits(self):
        args = self.parse('--tracking-profile', 'formula-wbc', '--formula-source-reference')
        self.assertIsNone(args.reference_pose_candidate)
        self.assertEqual((args.swing_midpoint_clearance, args.smooth_stride, args.tube_smooth_stride), (0., 0, 0))

    def test_incompatible_execution_or_relaxed_acceptance_rejected(self):
        for extra in [('--g1-md',), ('--tracking-steps', '12'),
                      ('--diagnostic-continue-after-fidelity-loss',), ('--mit-consistent-sync',)]:
            with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    self.parse('--tracking-profile', 'formula-wbc', *extra)


class FormulaLawTest(unittest.TestCase):
    def test_zero_error_neutral_and_outward_velocity_increases_risk(self):
        zero = torch.zeros(1, 3, dtype=torch.float64)
        law = PhaseTrackingLaw()
        weights, activation, risk = law.update([(zero, zero)], [1.], .002)
        torch.testing.assert_close(weights, torch.ones_like(weights))
        self.assertEqual(activation.item(), 0.)
        self.assertEqual(risk.item(), 0.)
        error = zero.clone(); error[:, 0] = .4
        outward = zero.clone(); outward[:, 0] = 10.
        moving = PhaseTrackingLaw().update([(error, outward)], [1.], .002)
        still = PhaseTrackingLaw().update([(error, zero)], [1.], .002)
        self.assertGreater(moving[0].item(), still[0].item())
        self.assertGreater(moving[2].item(), still[2].item())

    def test_filter_period_invariance_and_bounded_response(self):
        error = torch.tensor([[1e3, 0., 0.]], dtype=torch.float64)
        zero = torch.zeros_like(error)
        a, b = PhaseTrackingLaw(maximum=1.2), PhaseTrackingLaw(maximum=1.2)
        result_a = a.update([(error, zero)], [1.], .02)
        for _ in range(10):
            result_b = b.update([(error, zero)], [1.], .002)
        for x, y in zip(result_a, result_b):
            torch.testing.assert_close(x, y)
        self.assertTrue(bool(((result_a[0] >= 1) & (result_a[0] <= 1.2)).all()))
        self.assertTrue(bool(((result_a[1] >= 0) & (result_a[1] <= 1)).all()))

    def test_task_allocation_preserves_feedforward_and_physical_rows(self):
        # Exercise the actual wiring with coupled positive-definite foot metrics.
        from fabrics_sim.fabrics.task_request import TaskRequest
        q = torch.zeros(1, 36, dtype=torch.float64); q[:, 6] = 1.
        q[:, 7:] = 1.
        v = torch.zeros(1, 35, dtype=q.dtype)
        audit = q.clone(); audit[:, 7:] = 0.
        def request(name, size, feedforward):
            basis = torch.eye(size, dtype=q.dtype) + .1
            return TaskRequest(name, torch.zeros(1, size, 35, dtype=q.dtype),
                torch.zeros(1, size, dtype=q.dtype), feedforward.clone(),
                (basis.T @ basis)[None], torch.ones(1, size, dtype=torch.bool))
        joint_a = torch.full((1, 29), 200., dtype=q.dtype)
        foot_a = torch.full((1, 12), 200., dtype=q.dtype)
        joint = request('joint_posture', 29, joint_a)
        feet = request('feet_pose', 12, foot_a)
        A, bias = joint.A.clone(), joint.bias.clone()
        transforms = torch.zeros(1, 2, 7, dtype=q.dtype)
        transforms[..., 0] = 1.
        foot_target = torch.zeros(1, 2, 3, dtype=q.dtype)
        fabric = SimpleNamespace(
            dynamics_tasks=SimpleNamespace(last_snapshot={'transforms': transforms},
                model=SimpleNamespace(kinematics=SimpleNamespace(get_link_index=lambda name: {'left': 0, 'right': 1}[name])),
                last_requests=[joint, feet], references=lambda feature, desired: feature),
            feet_pose_taskmap=SimpleNamespace(frame_groups={'left': ['left'], 'right': ['right']},
                end_effector_order=['left', 'right'], position_target=foot_target),
            joint_posture_taskmap=SimpleNamespace(position_target=torch.zeros(1, 29, dtype=q.dtype)),
            fabrics_features={
                'joint_posture': {'joint_posture_attractor': (torch.zeros_like(joint_a), joint_a)},
                'feet_pose': {'feet_pose_attractor': (torch.zeros_like(foot_a), foot_a)}},
            fabric_params={'joint_posture_attractor': {'stiffness': 1000., 'damping': 20.},
                'feet_pose_attractor': {'stiffness': [1000.] * 12, 'damping': [20.] * 12}})
        law = PhaseTrackingLaw()
        law.apply(fabric, q, v, audit, v, 10.)
        # Large feedback is clipped; feedforward 200 is never clipped to 150/80.
        torch.testing.assert_close(joint.desired, torch.full_like(joint_a, 50.))
        expected = foot_a.reshape(1, 2, 6).clone(); expected[..., 0] = 120.
        torch.testing.assert_close(feet.desired, expected.flatten(1))
        torch.testing.assert_close(joint.A, A); torch.testing.assert_close(joint.bias, bias)
        self.assertTrue(bool((torch.linalg.eigvalsh(feet.metric) > 0).all()))
        torch.testing.assert_close(fabric.joint_posture_taskmap.position_target, torch.zeros_like(joint_a))

    def test_missing_tasks_and_nonfinite_states_rejected(self):
        zero = torch.zeros(1, 3)
        for pairs, radii in [([], []), ([(zero, zero)], []),
                             ([(zero, zero)], [0.]), ([(zero * float('nan'), zero)], [1.])]:
            with self.subTest(radii=radii), self.assertRaises(ValueError):
                PhaseTrackingLaw().update(pairs, radii, .002)


if __name__ == '__main__':
    unittest.main()
