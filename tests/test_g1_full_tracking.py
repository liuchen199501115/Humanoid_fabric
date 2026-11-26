"""Runtime profile guards and independently differentiated rolling kinematics."""
import contextlib
import io
import math
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'examples'))
from test_g1 import (control_timesteps, parse_args, rolling_sphere_bias_correction, blend_root_position_weight,
                     scale_stance_posture_metric, root_position_request, blend_secondary_task_metrics, priority_error_diagnostics, step_physics_interval)


class FullTrackingProfileTest(unittest.TestCase):
    def parse(self, *args):
        with patch.object(sys, 'argv', ['test_g1.py', *args]):
            return parse_args()

    def test_full_source_and_validation_are_required(self):
        args = self.parse('--tracking-profile', 'amp-full')
        self.assertTrue(args.track_full_motion and args.stop_at_fidelity_loss and args.direct_wbc)
        self.assertEqual(args.tracking_steps, 0)
        self.assertIsNone(args.support_knee_min_angle)
        self.assertEqual(args.joint_limit_priority_scale, 1.)
        self.assertEqual(args.support_knee_guard_start, float('inf'))
        for extra in [('--tracking-steps', '1'), ('--diagnostic-continue-after-fidelity-loss',),
                      ('--full-body-preview-start', '0'), ('--g1-md',)]:
            with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    self.parse('--tracking-profile', 'amp-full', *extra)

    def test_ordinary_entry_does_not_enable_experimental_gates(self):
        args = self.parse()
        self.assertIsNone(args.tracking_profile)
        self.assertIsNone(args.maximum_support_acceleration)
        self.assertIsNone(args.com_upward_speed_limit)
        self.assertEqual(args.qp_late_iteration_start, float('inf'))
        self.assertEqual(args.qp_late_iterations, 5000)
        self.assertEqual(args.clock_final_speed, 1.)
        for name in ('tracking_tube_start', 'vertical_feedback_start', 'model_effort_start',
                     'flight_feedback_start', 'global_posture_feedback_start',
                     'stance_posture_metric_start', 'root_position_task_start',
                     'support_flatten_start', 'weighted_tasks_start',
                     'orientation_tube_start', 'late_physics_substeps_start', 'stance_heading_start', 'torque_continuity_start'):
            self.assertEqual(getattr(args, name), float('inf'))

    def test_periods_match_and_invalid_rates_fail_before_simulation(self):
        self.assertEqual(control_timesteps(500), (.002, .002))
        self.assertEqual(control_timesteps(1000, 2000), (.001, .0005))
        self.assertEqual(control_timesteps(1000, g1_md=True), (.001, .0005))
        for control, physics in [(1000, None), (500, 750), (0, 500)]:
            with self.assertRaises(ValueError):
                control_timesteps(control, physics)



class PriorityHistoryTest(unittest.TestCase):
    def test_stage_disappearance_remains_serializable_and_is_marked_inactive(self):
        errors = (torch.tensor([1e-5]), torch.tensor([2e-5]))
        history = [priority_error_diagnostics(errors), priority_error_diagnostics(None),
                   priority_error_diagnostics(errors)]
        arrays = {key: np.asarray([row[key] for row in history]) for key in history[0]}
        np.testing.assert_array_equal(arrays['priority_active'], [True, False, True])
        self.assertEqual(arrays['priority_lock_error'].shape, (3, 1))
        self.assertTrue(np.isnan(arrays['priority_lock_error'][1, 0]))
        np.testing.assert_allclose(arrays['priority_stage_constraint_error'][[0, 2], 0], 1e-5)


class WeightedTaskTransitionTest(unittest.TestCase):
    def make_request(self, name):
        from fabrics_sim.fabrics.task_request import TaskRequest
        A = torch.eye(35, dtype=torch.float64)[None, :3]
        zero = torch.zeros(1, 3, dtype=torch.float64)
        basis = torch.tensor([[1., 2., 0.], [0., 1., 1.], [1., 0., 3.]], dtype=torch.float64)
        return TaskRequest(name, A, zero, zero, (basis.T@basis)[None],
                           torch.ones(1, 3, dtype=torch.bool))

    def test_partial_primary_rows_preserve_cost_at_transition_endpoints(self):
        for blend in [0., .25, 1.]:
            primary = self.make_request('primary');secondary = self.make_request('secondary')
            original = primary.metric.clone()
            blend_secondary_task_metrics([primary, secondary], ('primary',), {'primary': (0, 2)}, blend)
            factors = torch.tensor([1., math.sqrt(blend), 1.], dtype=torch.float64)
            residual = torch.tensor([2., -1., 3.], dtype=torch.float64)
            expected = (residual*factors)@original[0]@(residual*factors)
            torch.testing.assert_close(residual@primary.metric[0]@residual, expected)
            torch.testing.assert_close(secondary.metric, original*blend)
            self.assertTrue(bool((torch.linalg.eigvalsh(primary.metric) >= -1e-10).all()))

    def test_full_primary_unchanged_and_invalid_blend_rejected(self):
        request = self.make_request('primary');original = request.metric.clone()
        blend_secondary_task_metrics([request], ('primary',), {}, .1)
        torch.testing.assert_close(request.metric, original)
        for blend in [-.1, 1.1, float('nan')]:
            with self.assertRaises(ValueError):
                blend_secondary_task_metrics([request], (), {}, blend)


class RootPositionTaskTest(unittest.TestCase):
    def test_late_root_cost_is_a_convex_combination_of_axis_objectives(self):
        q=torch.zeros(1,36,dtype=torch.float64);v=torch.zeros(1,35,dtype=torch.float64)
        target=q.clone();target[:,:3]=torch.tensor([.1,-.2,.3])
        initial=root_position_request(q,v,target,v,v,100.,axes='y')
        final=root_position_request(q,v,target,v,v,1000.,axes='xyz')
        initial_form=initial.natural_form();final_form=final.natural_form()
        blend_root_position_weight(initial,1000.,'xyz',.3)
        for actual,left,right in zip(initial.natural_form(),initial_form,final_form):
            torch.testing.assert_close(actual,.7*left+.3*right)
        self.assertTrue(bool((torch.linalg.eigvalsh(initial.metric)>=0).all()))

    def test_lateral_task_ignores_forward_and_vertical_targets(self):
        q=torch.zeros(1,36,dtype=torch.float64);v=torch.zeros(1,35,dtype=torch.float64)
        target=q.clone();target[:,1]=.1
        first=root_position_request(q,v,target,v,v,300.,axes='y')
        q[:,[0,2]]=10.;target[:,[0,2]]=-20.;v[:,[0,2]]=100.
        second=root_position_request(q,v,target,v,v,300.,axes='y')
        for before,after in zip(first.natural_form(),second.natural_form()):
            torch.testing.assert_close(before,after,rtol=0,atol=0)
        H,g=second.natural_form()
        self.assertEqual(H[0,0,0].item(),0.)
        self.assertEqual(H[0,2,2].item(),0.)
        self.assertGreater(H[0,1,1].item(),0.)

    def test_horizontal_task_has_no_vertical_cost(self):
        q=torch.zeros(1,36,dtype=torch.float64);v=torch.zeros(1,35,dtype=torch.float64)
        target=q.clone();target[:,:2]=torch.tensor([.1,-.2])
        first=root_position_request(q,v,target,v,v,300.,axes='xy')
        q[:,2]=10.;target[:,2]=-20.;changed_v=v.clone();changed_v[:,2]=100.
        changed_a=v.clone();changed_a[:,2]=-300.
        second=root_position_request(q,changed_v,target,v,changed_a,300.,axes='xy')
        for before,after in zip(first.natural_form(),second.natural_form()):
            torch.testing.assert_close(before,after,rtol=0,atol=0)
        H,g=second.natural_form()
        self.assertEqual(H[0,2,2].item(),0.)
        self.assertEqual(g[0,2].item(),0.)
        self.assertGreater(H[0,0,0].item(),0.)
        with self.assertRaises(ValueError):root_position_request(q,v,target,v,v,1.,axes='invalid')

    def test_moving_target_feedforward_and_world_translation_invariance(self):
        q = torch.zeros(2, 36, dtype=torch.float64)
        v = torch.zeros(2, 35, dtype=torch.float64)
        target = q.clone(); target[:, :3] = torch.tensor([.1, -.2, .3])
        target_v = v.clone(); target_v[:, :3] = torch.tensor([.2, .3, -.1])
        target_a = v.clone(); target_a[:, :3] = torch.tensor([40., -50., 60.])
        request = root_position_request(q, v, target, target_v, target_a, 1000.)
        expected = target_a[:, :3]+100*target[:, :3]+20*target_v[:, :3]
        torch.testing.assert_close(request.desired, expected)
        # The task observes only root translation, including in a batch.
        acceleration = torch.arange(70, dtype=q.dtype).reshape(2, 35)
        torch.testing.assert_close(request.residual(acceleration), acceleration[:, :3]-expected)
        shifted_q = q.clone(); shifted_target = target.clone()
        shifted_q[:, :3] += 10.; shifted_target[:, :3] += 10.
        shifted = root_position_request(shifted_q, v, shifted_target, target_v, target_a, 1000.)
        torch.testing.assert_close(shifted.desired, request.desired)

    def test_feedback_clipping_preserves_feedforward_and_rejects_invalid_metric(self):
        q = torch.zeros(1, 36);v = torch.zeros(1, 35)
        target = torch.ones_like(q);a = torch.full_like(v, 80.)
        request = root_position_request(q, v, target, v, a, 1.)
        torch.testing.assert_close(request.desired, torch.full((1, 3), 110.))
        for weight in [0., -1., float('inf'), float('nan')]:
            with self.assertRaises(ValueError):
                root_position_request(q, v, target, v, a, weight)


class StanceMetricTest(unittest.TestCase):
    def test_dense_metric_preserves_residual_energy_and_other_coordinates(self):
        generator = torch.Generator().manual_seed(8)
        basis = torch.randn(2, 29, 29, generator=generator, dtype=torch.float64)
        metric = basis.transpose(-1, -2) @ basis
        original = metric.clone()
        contact = torch.tensor([[True, False], [False, True]])
        adjusted = scale_stance_posture_metric(metric, contact, .25)
        residual = torch.randn(2, 29, generator=generator, dtype=torch.float64)
        scaled = residual.clone()
        scaled[0, :6] *= .5
        scaled[1, 6:12] *= .5
        expected = torch.einsum('bi,bij,bj->b', scaled, metric, scaled)
        actual = torch.einsum('bi,bij,bj->b', residual, adjusted, residual)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(adjusted[:, 12:, 12:], metric[:, 12:, 12:])
        torch.testing.assert_close(adjusted[0, 6:, 6:], metric[0, 6:, 6:])
        torch.testing.assert_close(metric, original)
        self.assertTrue(bool((torch.linalg.eigvalsh(adjusted) >= -1e-10).all()))

    def test_flight_is_unchanged_and_invalid_weights_fail(self):
        metric = torch.eye(29)[None]
        contact = torch.zeros(1, 2, dtype=torch.bool)
        torch.testing.assert_close(scale_stance_posture_metric(metric, contact, .1), metric)
        for scale in (0., -1., 1.1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                scale_stance_posture_metric(metric, contact, scale)


class RollingSphereKinematicsTest(unittest.TestCase):
    def test_bias_matches_derivative_of_migrating_contact_velocity(self):
        # Differentiate the actual lowest sphere point's velocity as the body
        # rotates. This includes a nonzero collider-center offset and mixed axes.
        dtype = torch.float64
        R = Rotation.from_rotvec([.2, -.3, .1]).as_matrix()
        frame = Rotation.from_euler('z', .7).as_matrix()
        omega = np.array([3., -4., 2.])
        center = np.array([.12, -.025, -.03]); radius = .005
        world_offset = R @ center - np.array([0., 0., radius])
        local_point = R.T @ world_offset
        correction, matched = rolling_sphere_bias_correction(
            torch.tensor(R[None], dtype=dtype), torch.tensor(omega[None], dtype=dtype),
            torch.tensor(frame[None], dtype=dtype), torch.tensor(local_point[None, None], dtype=dtype),
            torch.tensor(center[None], dtype=dtype), torch.tensor([radius], dtype=dtype))
        def surface_velocity(t):
            rotation = Rotation.from_rotvec(t * omega).as_matrix() @ R
            return np.cross(omega, rotation @ center - [0., 0., radius])
        dt = 1e-6
        derivative = (surface_velocity(dt) - surface_velocity(-dt)) / (2 * dt)
        rigid_bias = np.cross(omega, np.cross(omega, world_offset))
        self.assertTrue(matched.item())
        np.testing.assert_allclose(frame.T @ rigid_bias + correction[0].numpy(),
                                   frame.T @ derivative, atol=1e-9)

    def test_unmatched_or_nonspherical_contact_is_unchanged(self):
        identity = torch.eye(3, dtype=torch.float64)[None]
        omega = torch.tensor([[3., 4., 0.]], dtype=torch.float64)
        centers = torch.zeros(1, 3, dtype=torch.float64)
        for point, radius in [([.1, 0., -.005], .005), ([0., 0., 0.], 0.)]:
            correction, matched = rolling_sphere_bias_correction(
                identity, omega, identity, torch.tensor([[point]], dtype=torch.float64),
                centers, torch.tensor([radius], dtype=torch.float64))
            self.assertFalse(matched.item())
            torch.testing.assert_close(correction, torch.zeros_like(correction))


if __name__ == '__main__':
    unittest.main()


class PhysicsIntervalTest(unittest.TestCase):
    class World:
        current_time = 0.
        def __init__(self):
            self.dt = .002
            self.steps = []
            self.changes = []
        def get_physics_dt(self): return self.dt
        def get_rendering_dt(self): return 1/60
        def set_simulation_dt(self, physics_dt, rendering_dt):
            self.changes.append((physics_dt, rendering_dt))
            self.dt = physics_dt
        def step(self, render):
            self.steps.append((self.dt, render))
            self.current_time += self.dt

    def test_subdivision_preserves_torque_clock_and_reports_impulse_interval(self):
        world = self.World()
        self.assertEqual(step_physics_interval(world, .002, 1), .002)
        self.assertEqual(world.changes, [])
        self.assertEqual(step_physics_interval(world, .002, 4), .0005)
        self.assertAlmostEqual(world.current_time, .004)
        self.assertEqual(world.steps, [(.002, False)]+[(.0005, False)]*4)
        step_physics_interval(world, .002, 4)
        self.assertEqual(len(world.changes), 1)
        self.assertAlmostEqual(world.current_time, .006)

    def test_invalid_subdivision_cannot_step_physics(self):
        for count in [0, -1, 1.5, True]:
            world = self.World()
            with self.assertRaises(ValueError): step_physics_interval(world, .002, count)
            self.assertEqual(world.steps, [])

    def test_held_effort_is_reissued_before_every_native_substep(self):
        world = self.World(); issued = []
        def apply(): issued.append(len(world.steps))
        step_physics_interval(world, .002, 4, apply_effort=apply)
        self.assertEqual(issued, [0, 1, 2, 3])
