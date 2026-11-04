import unittest
from types import SimpleNamespace

import torch
import warp as wp

from fabrics_sim.reference.whole_body import (
    WholeBodyReference, validate_reference_options, support_orientation_request,
)


class WholeBodyReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.config.kernel_cache_dir = '/tmp/g1_warp_cache'
        wp.init()
        from fabrics_sim.fabrics.g1_pose_fabric import G1PoseFabric
        cls.fabric = G1PoseFabric(1, 'cpu', robot_name='g1_29dof_rev_1_0',
                                  task_profile='dynamics-aware')

    def setUp(self):
        self.q = torch.zeros(1, 36)
        self.q[:, 2], self.q[:, 6] = .8, 1.
        self.v = torch.linspace(-.01, .01, 35)[None]
        self.a = torch.linspace(.02, -.02, 35)[None]
        self.fabric.dynamics_tasks.source_pelvis_metric_scale = 1.
        self.fabric.set_contact_state(torch.tensor([[True, False]]))

    def test_all_tracking_tasks_agree_at_moving_reference(self):
        # Any leftover COM offset, crouch, torso hold or hand shift produces
        # feedback error here even when actual q/v equal the reference.
        reference = WholeBodyReference(self.q, self.v, self.a)
        reference.apply(self.fabric)
        requests = {r.name: r for r in self.fabric.compute_task_requests(self.q, self.v)}
        names = ('center_of_mass', 'centroidal_momentum', 'body_orientation',
                 'source_pelvis_orientation', 'feet_pose', 'hand_pose', 'joint_posture')
        for name in names:
            with self.subTest(task=name):
                residual = requests[name].residual(self.a)
                torch.testing.assert_close(residual, torch.zeros_like(residual), atol=2e-4, rtol=0)
        self.assertEqual(requests['feet_pose'].active[0, :6].count_nonzero(), 0)
        self.assertEqual(requests['feet_pose'].active[0, 6:].count_nonzero(), 6)

    def test_stationary_reference_clears_previous_motion_and_preserves_source_joints(self):
        WholeBodyReference(self.q, self.v, self.a).apply(self.fabric)
        z = torch.zeros_like(self.v)
        WholeBodyReference(self.q, z, z).apply(self.fabric)
        torch.testing.assert_close(self.fabric.joint_posture_taskmap.position_target, self.q[:, 7:])
        for request in self.fabric.compute_task_requests(self.q, z):
            if request.name != 'collision':
                torch.testing.assert_close(request.desired, torch.zeros_like(request.desired), atol=2e-4, rtol=0)

    def test_source_is_not_aliased_by_callers_or_live_kinematics(self):
        reference = WholeBodyReference(self.q, self.v, self.a)
        expected = reference.q.clone()
        self.q[:, 7:] += .2
        self.v.zero_()
        reference.apply(self.fabric)
        self.fabric.compute_task_requests(self.q, self.v)
        self.fabric.joint_posture_taskmap.position_target.add_(.1)
        torch.testing.assert_close(reference.q, expected, rtol=0, atol=0)

    def test_shared_retiming_reaches_every_task(self):
        from fabrics_sim.reference.interpolation import MotionInterpolator
        poses = self.q.repeat(4, 1)
        poses[:, 0] = torch.arange(4) * .003
        poses[:, 7:] += torch.arange(4)[:, None] * .002
        sampler = MotionInterpolator(poses)
        q, v, a = sampler.kinematics(1.3, 2., .5)
        WholeBodyReference(q, v, a).apply(self.fabric)
        for request in self.fabric.compute_task_requests(q, v):
            if request.name != 'collision':
                residual = request.residual(a)
                torch.testing.assert_close(residual, torch.zeros_like(residual), atol=2e-4, rtol=0)

    def test_invalid_reference_is_rejected_before_targets_change(self):
        for q, v, a in ((self.q[:, :-1], self.v, self.a),
                        (self.q, self.v.double(), self.a),
                        (self.q, self.v, self.a * float('nan'))):
            with self.assertRaises(ValueError):
                WholeBodyReference(q, v, a)

    def test_observed_contact_uses_source_orientation_in_its_free_coordinates(self):
        WholeBodyReference(self.q, self.v, self.a).apply(self.fabric)
        feet = next(r for r in self.fabric.compute_task_requests(self.q, self.v) if r.name == 'feet_pose')
        # A rotated contact edge must release its own y axis, not world y.
        frame = torch.tensor([[[.8, -.6, 0.], [.6, .8, 0.], [0., 0., 1.]]])
        for point, polygon, count in ((False, False, 1), (True, False, 3), (False, True, 0)):
            request = support_orientation_request(feet, 0, frame, point, polygon)
            self.assertEqual(int(request.active.sum()), count)
            torch.testing.assert_close(request.residual(self.a), torch.zeros(1, 3), atol=2e-4, rtol=0)

    def test_rotated_support_metric_stays_symmetric_at_larger_weights(self):
        from dataclasses import replace
        WholeBodyReference(self.q, self.v, self.a).apply(self.fabric)
        feet = next(r for r in self.fabric.compute_task_requests(self.q,self.v) if r.name=='feet_pose')
        feet = replace(feet, metric=feet.metric*100.)
        torch.manual_seed(17)
        for _ in range(20):
            frame = torch.linalg.qr(torch.randn(1,3,3)).Q
            request = support_orientation_request(feet,0,frame,True,False)
            self.assertTrue(torch.equal(request.metric,request.metric.transpose(-1,-2)))
            expected = frame.transpose(-1,-2)@feet.metric[:,3:6,3:6]@frame
            torch.testing.assert_close(request.metric,expected,atol=2e-4,rtol=1e-5)


class ReferenceModeTests(unittest.TestCase):
    def options(self, **kwargs):
        return SimpleNamespace(**(dict(motion_pkl='walk1_g1.pkl', reference_mode='unified',
            task_profile='dynamics-aware', motion_interpolation='pchip-analytic',
            reference_manager=True, reference_footsteps='source') | kwargs))

    def test_conflicting_generators_fail_explicitly(self):
        self.assertTrue(validate_reference_options(self.options()))
        for flag in ('dynamic_reference', 'source_swing_reference', 'align_balance_pelvis',
                     'weight_transfer_test', 'contact_cycle_test'):
            with self.subTest(flag=flag), self.assertRaisesRegex(ValueError, 'legacy-managed'):
                validate_reference_options(self.options(**{flag: True}))
        self.assertFalse(validate_reference_options(self.options(reference_mode='legacy-managed', dynamic_reference=True)))

    def test_unified_requires_one_analytic_source(self):
        self.assertTrue(validate_reference_options(self.options(motion_interpolation="bspline-analytic")))
        for option in (dict(reference_footsteps='compact'), dict(task_profile='legacy'),
                       dict(motion_interpolation='linear')):
            with self.assertRaises(ValueError):
                validate_reference_options(self.options(**option))


if __name__ == '__main__':
    unittest.main()
