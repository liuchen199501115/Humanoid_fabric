import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import warp as wp

from fabrics_sim.reference.inverse_dynamics import required_single_support
from fabrics_sim.reference.source_step import smooth_body_samples
from fabrics_sim.reference.source_step_optimization import constraint_score, correct_candidate, constraint_family_penalty
from fabrics_sim.reference.whole_body import WholeBodyReference


class SourceAdaptationTests(unittest.TestCase):
    def test_smoothing_preserves_pose_endpoints_and_unmodified_coordinates(self):
        q=torch.zeros(6,36);q[:,6]=1.;q[2,20]=.6;q[3,3:7]*=-1
        original=q.clone();result=smooth_body_samples(q,4)
        torch.testing.assert_close(q,original)
        torch.testing.assert_close(result[[0,-1]],q[[0,-1]])
        torch.testing.assert_close(result[:,:3],q[:,:3])
        torch.testing.assert_close(result[:,7:19],q[:,7:19])
        torch.testing.assert_close(result[:,3:7].norm(dim=-1),torch.ones(6))
        self.assertLess(float(result[2,20]),.6)

    def row(self, cop=(.02,0.), force=(0.,0.,100.), velocity=.5):
        return dict(cop=np.array([cop]),force=np.array([force]),velocity_ratio=velocity,
                    wrench_passed=True,angular_momentum_rate=np.zeros((1,3)))

    def test_score_keeps_friction_and_velocity_even_when_cop_passes(self):
        self.assertEqual(constraint_score([self.row()]),0.)
        self.assertGreater(constraint_score([self.row(force=(70.,0.,100.))]),0.)
        self.assertGreater(constraint_score([self.row(velocity=1.1)]),0.)
        self.assertGreater(constraint_score([self.row(force=(0.,0.,0.))]),0.)
        with self.assertRaises(ValueError):
            constraint_score([self.row(cop=(float('nan'),0.))])
        self.assertGreater(constraint_score([self.row(cop=(.02,.03))]),0.)

    def test_exact_dynamics_bounds_and_nonfinite_values_are_rejected(self):
        self.assertGreater(constraint_score([dict(self.row(), exact_wrench_score=.1)]), 0.)
        self.assertGreater(constraint_score([dict(self.row(), torque_ratio=1.01)]), 0.)
        self.assertGreater(constraint_score([dict(self.row(), acceleration_ratio=1.01)]), 0.)
        self.assertGreater(constraint_score([dict(self.row(), joint_bound_score=.01)]), 0.)
        for key in ('exact_wrench_score', 'torque_ratio', 'acceleration_ratio', 'joint_bound_score'):
            with self.assertRaises(ValueError):
                constraint_score([dict(self.row(), **{key: float('nan')})])

    def test_ground_penetration_cannot_pass_other_valid_bounds(self):
        self.assertEqual(constraint_score([dict(self.row(),swing_sole_min_height=-1e-6)]),0.)
        self.assertGreater(constraint_score([dict(self.row(),swing_sole_min_height=-.01)]),.9)
        with self.assertRaises(ValueError):
            constraint_score([dict(self.row(),swing_sole_min_height=float('nan'))])

    def test_penalty_exposes_braking_when_wrench_is_worse(self):
        a=[dict(self.row(),exact_wrench_score=.3,joint_bound_score=.1)]
        b=[dict(self.row(),exact_wrench_score=.3,joint_bound_score=.2)]
        self.assertEqual(constraint_score(a),constraint_score(b))
        pa,pb=constraint_family_penalty(a),constraint_family_penalty(b)
        self.assertEqual(pa['exact_wrench'],pb['exact_wrench'])
        self.assertGreater(pb['joint_braking'],pa['joint_braking'])
        self.assertGreater(sum(pb.values()),sum(pa.values()))
        self.assertEqual(sum(constraint_family_penalty([self.row()]).values()),0.)
        with self.assertRaises(ValueError):constraint_score([])

    def test_full_body_audit_does_not_replace_reference(self):
        old = SimpleNamespace(duration=.8)
        plan = SimpleNamespace(preview=old, initial=SimpleNamespace(q=torch.zeros(1,36)))
        rows = [self.row(cop=(.02,.04)), self.row(cop=(.02,.04))]
        with patch('fabrics_sim.reference.source_step_optimization.evaluate_candidate', return_value=rows) as evaluate:
            returned, report = correct_candidate(plan, np.array([0.,.8]), torch.ones(29), 2)
        evaluate.assert_called_once()
        self.assertIs(plan.preview, old)
        self.assertIs(returned, rows)
        self.assertFalse(report['sampled_necessary_constraints_passed'])
        self.assertEqual(report['rejected'], [])


class InverseDynamicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.config.kernel_cache_dir='/tmp/g1_warp_cache';wp.init()
        from fabrics_sim.fabrics.g1_pose_fabric import G1PoseFabric
        cls.fabric=G1PoseFabric(1,'cpu',robot_name='g1_29dof_rev_1_0',task_profile='dynamics-aware')
        cls.model=cls.fabric.dynamics_tasks.model
        cls.foot=cls.model.kinematics.get_link_index('left_ankle_roll_link')

    def reference(self):
        q=torch.zeros(1,36);q[:,2]=.8;q[:,6]=1.
        return WholeBodyReference(q,torch.zeros(1,35),torch.zeros(1,35))

    def test_static_wrench_matches_gravity_and_com_moment(self):
        ref=self.reference();r=required_single_support(self.model,ref,self.fabric.integrate_root_position,self.foot)
        expected=torch.tensor([[0.,0.,float(self.model.com_task.link_masses.sum())*9.81]])
        torch.testing.assert_close(r['wrench'][:,:3],expected,atol=1e-4,rtol=1e-5)
        state=r['state'];moment=torch.linalg.cross(state['com']-state['transforms'][:,self.foot,:3],expected)
        torch.testing.assert_close(r['wrench'][:,3:],moment,atol=1e-4,rtol=1e-5)
        self.assertLess(float(r['base_residual'].abs().max()),1e-4)

    def test_rotor_inertia_adds_joint_torque_and_does_not_fake_support(self):
        base=self.reference();a=torch.linspace(-.1,.1,35)[None];ref=WholeBodyReference(base.q,base.v,a)
        without=required_single_support(self.model,ref,self.fabric.integrate_root_position,self.foot)
        armature=torch.ones(1,29)*.02
        with_rotor=required_single_support(self.model,ref,self.fabric.integrate_root_position,self.foot,armature)
        torch.testing.assert_close(with_rotor['wrench'],without['wrench'])
        torch.testing.assert_close(with_rotor['torque']-without['torque'],armature*a[:,6:],atol=2e-6,rtol=1e-4)

    def test_checked_reference_feedforward_survives_soft_task_limits(self):
        base = self.reference()
        acceleration = torch.linspace(-80., 100., 35)[None]
        ref = WholeBodyReference(base.q, base.v, acceleration)
        tasks = self.fabric.dynamics_tasks
        tasks.source_pelvis_metric_scale = 1.
        ref.apply(self.fabric)
        try:
            tasks.preserve_reference_acceleration = True
            for request in tasks.compute(ref.q, ref.v):
                if request.name == 'collision':
                    continue
                self.assertLess(float(request.residual(ref.a).abs().max()), 2e-4, request.name)
            # A far-away reference still has a bounded feedback correction.
            reference = torch.tensor([[100.]])
            torch.testing.assert_close(tasks._limit_tracking(reference+200., reference, 30.), reference+30.)
            tasks.preserve_reference_acceleration = False
            requests = tasks.compute(ref.q, ref.v)
            posture = next(r for r in requests if r.name == 'joint_posture')
            self.assertGreater(float(posture.residual(ref.a).abs().max()), 1.)
        finally:
            tasks.preserve_reference_acceleration = False
            tasks.source_pelvis_metric_scale = 0.
