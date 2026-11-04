import unittest

import torch

from fabrics_sim.solver.AccelerationQP import (
    G1DynamicsAccelerationQP, build_single_support_wrench_constraints,
)


class ContactWrenchFrameTest(unittest.TestCase):
    def test_sole_offset_preserves_point_force_wrench(self):
        rotation = torch.eye(3, dtype=torch.float64).expand(1, 2, 3, 3)
        kwargs = dict(contact_mode=torch.tensor([3]), foot_rotation_world=rotation,
                      half_width=.025, heel_length=.05)
        G, b = build_single_support_wrench_constraints(**kwargs, sole_depth=.035)
        G0, b0 = build_single_support_wrench_constraints(**kwargs)
        force = torch.tensor([12., -8., 200.], dtype=torch.float64)
        point = torch.tensor([.03, .02, -.035], dtype=torch.float64)
        ankle = torch.cat((force, torch.linalg.cross(point, force))).repeat(2)[None]
        point[2] = 0.
        sole = torch.cat((force, torch.linalg.cross(point, force))).repeat(2)[None]
        torch.testing.assert_close((G @ ankle[..., None]).squeeze(-1)-b,
                                   (G0 @ sole[..., None]).squeeze(-1)-b0)
        self.assertTrue(((G @ ankle[..., None]).squeeze(-1) <= b+1e-10).all())

    def test_mixed_batch_matches_individual_modes(self):
        modes = torch.tensor([0, 1, 2, 3])
        rotations = torch.eye(3, dtype=torch.float64).expand(4, 2, 3, 3)
        G, b = build_single_support_wrench_constraints(
            contact_mode=modes, foot_rotation_world=rotations)
        for index in range(4):
            single_G, single_b = build_single_support_wrench_constraints(
                contact_mode=modes[index:index+1],
                foot_rotation_world=rotations[index:index+1])
            torch.testing.assert_close(G[index:index+1], single_G)
            torch.testing.assert_close(b[index:index+1], single_b)
        self.assertEqual(G[0].count_nonzero().item(), 0)
        self.assertGreater(G[3, :12, :6].count_nonzero().item(), 0)
        self.assertGreater(G[3, 12:, 6:].count_nonzero().item(), 0)
        self.assertEqual(b[3, 4].item(), 0.)
        self.assertEqual(b[3, 16].item(), 0.)
        self.assertEqual(b[1, 4].item(), -20.)
        self.assertEqual(b[2, 16].item(), -20.)

    def test_double_support_rotation_preserves_local_wrench_residual(self):
        mode = torch.tensor([3])
        identity = torch.eye(3, dtype=torch.float64).expand(1, 2, 3, 3)
        rotation = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]],
                                dtype=torch.float64).expand(1, 2, 3, 3)
        local = torch.tensor([[[10., 20., 100., 2., -4., 1.],
                               [-20., 5., 120., -1., 3., -2.]]], dtype=torch.float64)
        world = torch.cat(((rotation @ local[..., :3, None]).squeeze(-1),
                           (rotation @ local[..., 3:, None]).squeeze(-1)), dim=-1)
        local_G, local_b = build_single_support_wrench_constraints(
            contact_mode=mode, foot_rotation_world=identity)
        world_G, world_b = build_single_support_wrench_constraints(
            contact_mode=mode, foot_rotation_world=rotation)
        torch.testing.assert_close(
            (local_G @ local.reshape(1, 12, 1)).squeeze(-1) - local_b,
            (world_G @ world.reshape(1, 12, 1)).squeeze(-1) - world_b)


class AccelerationQPEffortLimitTest(unittest.TestCase):
    def test_unbounded_acceleration_retains_effort_limit(self):
        dtype=torch.float64
        target=torch.zeros(1,35,dtype=dtype);target[:,6]=300.
        solver=G1DynamicsAccelerationQP(solver_backend='daqp')
        args=dict(acc_reference=target,mass_matrix=torch.eye(35,dtype=dtype)[None],
            bias_force=torch.zeros_like(target),contact_jacobian=torch.zeros(1,12,35,dtype=dtype),
            jdot_qdot=torch.zeros(1,12,dtype=dtype),contact_mode=torch.tensor([0]),effort_limits=200.)
        bounded=solver.solve(**args,acc_lower=-150.,acc_upper=150.)
        unbounded=solver.solve(**args,acc_lower=-float('inf'),acc_upper=float('inf'),
            predicted_acc_lower=None,predicted_acc_upper=None)
        self.assertTrue(unbounded.validated.all())
        self.assertLessEqual(float(bounded.acc[0,6]),150.01)
        self.assertGreater(float(unbounded.acc[0,6]),190.)
        self.assertLessEqual(float(unbounded.total_joint_torque.abs().max()),200.01)
        self.assertEqual(float(unbounded.contact_wrench.abs().max()),0.)

    def test_wrench_objective_tracks_total_feedback_inclusive_force(self):
        dtype = torch.float64
        jac = torch.zeros((1, 12, 35), dtype=dtype)
        jac[:, :6, 6:12] = torch.eye(6, dtype=dtype)
        target = torch.zeros((1, 12), dtype=dtype)
        target[:, 2] = 10.
        result = G1DynamicsAccelerationQP(max_iter=4000, wrench_weight=1.).solve(
            acc_reference=torch.zeros((1, 35), dtype=dtype),
            mass_matrix=torch.eye(35, dtype=dtype).unsqueeze(0),
            bias_force=torch.zeros((1, 35), dtype=dtype), contact_jacobian=jac,
            jdot_qdot=torch.zeros((1, 12), dtype=dtype),
            contact_mode=torch.ones(1, dtype=torch.long),
            servo_bias=torch.full((1, 29), 4., dtype=dtype),
            servo_acc_gain=torch.full((1, 29), .1, dtype=dtype),
            contact_wrench_reference=target,
            friction_G=torch.zeros((1, 12), dtype=dtype), friction_b=torch.zeros(1, dtype=dtype))
        self.assertTrue(result.validated.all())
        torch.testing.assert_close(result.contact_wrench, target, atol=.002, rtol=.001)
        self.assertGreater((result.nominal_contact_wrench - target).abs().max().item(), 3.9)

    def test_untracked_contact_force_target_does_not_change_solution(self):
        dtype = torch.float64
        jac = torch.zeros((1,12,35),dtype=dtype)
        jac[:,:,6:18] = torch.eye(12,dtype=dtype)
        weights = torch.zeros((1,12),dtype=dtype)
        weights[:,:6] = 1.
        args = dict(acc_reference=torch.zeros((1,35),dtype=dtype),
            mass_matrix=torch.eye(35,dtype=dtype)[None],
            bias_force=torch.zeros((1,35),dtype=dtype),contact_jacobian=jac,
            jdot_qdot=torch.zeros((1,12),dtype=dtype),contact_mode=torch.tensor([3]),
            friction_G=torch.zeros((1,12),dtype=dtype),friction_b=torch.zeros(1,dtype=dtype),
            contact_wrench_weights=weights,
            servo_bias=torch.full((1,29),4.,dtype=dtype),
            servo_acc_gain=torch.full((1,29),.1,dtype=dtype))
        solver = G1DynamicsAccelerationQP(max_iter=4000,wrench_weight=1.)
        target = torch.zeros((1,12),dtype=dtype)
        target[:,2] = 10.
        first = solver.solve(**args,contact_wrench_reference=target)
        target[:,8] = 1000.
        second = G1DynamicsAccelerationQP(max_iter=4000,wrench_weight=1.).solve(
            **args,contact_wrench_reference=target)
        self.assertTrue(first.validated.all() and second.validated.all())
        torch.testing.assert_close(first.contact_wrench,second.contact_wrench,atol=1e-5,rtol=0.)
        self.assertAlmostEqual(float(second.contact_wrench[0,2]),10.,places=2)
        with self.assertRaises(ValueError):
            solver.solve(**(args | {'contact_wrench_weights': -torch.ones_like(weights)}),
                         contact_wrench_reference=target)

    def test_command_correction_validation_and_physical_acceleration_limit(self):
        dtype = torch.float64
        args = dict(acc_reference=torch.zeros((1, 35), dtype=dtype),
                    mass_matrix=torch.eye(35, dtype=dtype).unsqueeze(0),
                    bias_force=torch.zeros((1, 35), dtype=dtype),
                    contact_jacobian=torch.zeros((1, 12, 35), dtype=dtype),
                    jdot_qdot=torch.zeros((1, 12), dtype=dtype),
                    contact_mode=torch.zeros(1, dtype=torch.long))
        solver = G1DynamicsAccelerationQP(max_iter=4000)
        bad = torch.zeros((1, 35), dtype=dtype)
        bad[:, 0] = 1.
        with self.assertRaisesRegex(ValueError, "zero base rows"):
            solver.solve(**args, command_acc_correction=bad)
        bad.zero_()
        bad[:, 6] = float("nan")
        with self.assertRaises(ValueError):
            solver.solve(**args, command_acc_correction=bad)
        correction = torch.zeros((1, 35), dtype=dtype)
        correction[:, 6] = 2.
        result = solver.solve(**args, command_acc_correction=correction,
                              predicted_acc_limit=.1, effort_limits=.1)
        self.assertTrue(result.validated.all())
        self.assertGreater(result.acc[0, 6].item(), 1.899)
        self.assertLessEqual(result.predicted_acc.abs().max().item(), .101)
        torch.testing.assert_close(result.nominal_acc[:, 6:], result.total_joint_torque,
                                   atol=1e-5, rtol=0.)

    def test_asymmetric_physical_bounds_include_servo_and_correction(self):
        dtype = torch.float64
        zeros = torch.zeros((1, 35), dtype=dtype)
        correction = zeros.clone()
        correction[:, 6:] = 2.
        lo, hi = zeros.clone() - 100., zeros.clone() + 100.
        lo[:, 6:], hi[:, 6:] = .2, .3
        args = dict(acc_reference=zeros, mass_matrix=torch.eye(35, dtype=dtype)[None],
                    bias_force=zeros, contact_jacobian=torch.zeros((1, 12, 35), dtype=dtype),
                    jdot_qdot=torch.zeros((1, 12), dtype=dtype),
                    contact_mode=torch.zeros(1, dtype=torch.long),
                    command_acc_correction=correction,
                    servo_bias=torch.ones((1, 29), dtype=dtype),
                    servo_acc_gain=torch.full((1, 29), .2, dtype=dtype))
        solver = G1DynamicsAccelerationQP(max_iter=4000)
        result = solver.solve(**args, predicted_acc_lower=lo, predicted_acc_upper=hi)
        self.assertTrue(result.validated.all())
        self.assertTrue((result.predicted_acc[:, 6:] >= .2-1e-3).all())
        self.assertTrue((result.predicted_acc[:, 6:] <= .3+1e-3).all())
        self.assertGreater(result.acc[:, 6:].min().item(), .9)
        with self.assertRaises(ValueError):
            solver.solve(**args, predicted_acc_lower=lo)
        with self.assertRaises(ValueError):
            solver.solve(**args, predicted_acc_lower=hi, predicted_acc_upper=lo)

    def test_joint_only_physical_bounds_leave_base_unbounded(self):
        dtype = torch.float64
        zeros = torch.zeros((2,35), dtype=dtype)
        lower, upper = torch.full_like(zeros,-float('inf')), torch.full_like(zeros,float('inf'))
        lower[:,6:], upper[:,6:] = .2, .3
        # Different missing sides per batch must not disable another member's bound.
        lower[0,6] = -float('inf')
        upper[1,6] = float('inf')
        bias = zeros.clone()
        bias[:,2] = 9.81
        correction = zeros.clone()
        correction[:,6:] = 2.
        solver = G1DynamicsAccelerationQP(max_iter=4000)
        result = solver.solve(acc_reference=zeros,
            mass_matrix=torch.eye(35,dtype=dtype)[None].expand(2,-1,-1),
            bias_force=bias, contact_jacobian=torch.zeros((2,12,35),dtype=dtype),
            jdot_qdot=torch.zeros((2,12),dtype=dtype), contact_mode=torch.zeros(2,dtype=torch.long),
            command_acc_correction=correction, servo_bias=torch.ones((2,29),dtype=dtype),
            servo_acc_gain=torch.full((2,29),.2,dtype=dtype),
            predicted_acc_lower=lower, predicted_acc_upper=upper)
        self.assertTrue(result.validated.all())
        torch.testing.assert_close(result.predicted_acc[:,:6],-bias[:,:6],atol=1e-3,rtol=0.)
        self.assertTrue((result.predicted_acc[:,7:]>=.2-1e-3).all())
        self.assertTrue((result.predicted_acc[:,7:]<=.3+1e-3).all())
        self.assertLessEqual(result.predicted_acc[0,6],.3+1e-3)
        self.assertGreaterEqual(result.predicted_acc[1,6],.2-1e-3)
        self.assertTrue(torch.isfinite(solver.last_problem.G).all())
        self.assertTrue(torch.isfinite(solver.last_problem.b).all())

    def test_physical_bounds_reject_invalid_infinities_and_nan(self):
        dtype=torch.float64
        zeros=torch.zeros((1,35),dtype=dtype)
        args=dict(acc_reference=zeros,mass_matrix=torch.eye(35,dtype=dtype)[None],
            bias_force=zeros,contact_jacobian=torch.zeros((1,12,35),dtype=dtype),
            jdot_qdot=torch.zeros((1,12),dtype=dtype),contact_mode=torch.zeros(1,dtype=torch.long))
        solver=G1DynamicsAccelerationQP()
        for lo,hi in [(float('nan'),1.),(-1.,float('nan')),
                      (float('inf'),float('inf')),(-float('inf'),-float('inf'))]:
            with self.subTest(lower=lo,upper=hi),self.assertRaises(ValueError):
                solver.solve(**args,predicted_acc_lower=lo,predicted_acc_upper=hi)

    def test_command_sync_survives_contact_without_unphysical_feedforward(self):
        dtype = torch.float64
        correction = torch.zeros((1, 35), dtype=dtype)
        correction[:, 6] = 2.
        jac = torch.zeros((1, 12, 35), dtype=dtype)
        jac[:, :6, 6:12] = torch.eye(6, dtype=dtype)
        result = G1DynamicsAccelerationQP(max_iter=4000).solve(
            acc_reference=correction, command_acc_correction=correction,
            mass_matrix=torch.eye(35, dtype=dtype).unsqueeze(0),
            bias_force=torch.zeros_like(correction), contact_jacobian=jac,
            jdot_qdot=torch.zeros((1, 12), dtype=dtype),
            contact_mode=torch.ones(1, dtype=torch.long),
            servo_bias=torch.zeros((1, 29), dtype=dtype),
            servo_acc_gain=torch.full((1, 29), .1, dtype=dtype),
            friction_G=torch.zeros((1, 12), dtype=dtype), friction_b=torch.zeros(1, dtype=dtype),
            acc_lower=-3., acc_upper=3., effort_limits=1., predicted_acc_limit=.1)
        self.assertTrue(result.validated.all())
        self.assertAlmostEqual(result.acc[0, 6].item(), 2., places=3)
        torch.testing.assert_close(result.nominal_acc, result.acc - correction)
        self.assertLess(result.predicted_acc.abs().max().item(), .001)
        torch.testing.assert_close(
            result.nominal_acc[:, 6:12],
            result.joint_torque[:, :6] + result.nominal_contact_wrench[:, :6],
            atol=1e-5, rtol=0.)
        torch.testing.assert_close(jac @ result.predicted_acc.unsqueeze(-1),
                                   torch.zeros((1, 12, 1), dtype=dtype), atol=1e-5, rtol=0.)

    def test_predicted_base_zero_feedback_matches_nominal(self):
        dtype = torch.float64
        inputs = dict(
            acc_reference=torch.ones((1, 35), dtype=dtype),
            mass_matrix=torch.eye(35, dtype=dtype).unsqueeze(0),
            bias_force=torch.zeros((1, 35), dtype=dtype),
            contact_jacobian=torch.zeros((1, 12, 35), dtype=dtype),
            jdot_qdot=torch.zeros((1, 12), dtype=dtype),
            contact_mode=torch.zeros(1, dtype=torch.long))
        solver = G1DynamicsAccelerationQP(max_iter=4000)
        with self.assertRaisesRegex(ValueError, "requires a servo model"):
            solver.solve(**inputs, track_predicted_base=True)
        inputs.update(servo_bias=torch.zeros((1, 29), dtype=dtype),
                      servo_acc_gain=torch.zeros((1, 29), dtype=dtype))
        nominal = solver.solve(**inputs)
        # Compare from identical cold starts, not the previous solve's ADMM state.
        predicted = G1DynamicsAccelerationQP(max_iter=4000).solve(
            **inputs, track_predicted_base=True)
        self.assertTrue(nominal.validated.all() and predicted.validated.all())
        torch.testing.assert_close(nominal.acc, predicted.acc)
        torch.testing.assert_close(nominal.total_joint_torque, predicted.total_joint_torque)

    def test_predicted_base_objective_accounts_for_servo_coupling(self):
        dtype = torch.float64
        mass = torch.eye(35, dtype=dtype).unsqueeze(0)
        mass[:, 0, 6] = mass[:, 6, 0] = .3
        reference = torch.zeros((1, 35), dtype=dtype)
        servo_bias = torch.full((1, 29), 4., dtype=dtype)
        weights = torch.ones(35, dtype=dtype)
        weights[:6] = 100.
        solver = G1DynamicsAccelerationQP(acc_weight=weights, max_iter=4000)
        def solve(predicted):
            return solver.solve(
                acc_reference=reference, mass_matrix=mass,
                bias_force=torch.zeros_like(reference),
                contact_jacobian=torch.zeros((1, 12, 35), dtype=dtype),
                jdot_qdot=torch.zeros((1, 12), dtype=dtype),
                contact_mode=torch.zeros(1, dtype=torch.long),
                servo_bias=servo_bias, servo_acc_gain=torch.full((1, 29), .5, dtype=dtype),
                track_predicted_base=predicted)
        nominal, predicted = solve(False), solve(True)
        self.assertTrue(nominal.validated.all() and predicted.validated.all())
        self.assertLess(predicted.predicted_acc[:, :6].norm().item(),
                        .1 * nominal.predicted_acc[:, :6].norm().item())
        # Uncoupled joints retain persistent PD; the base objective must not
        # turn into all-joint feedback cancellation.
        torch.testing.assert_close(predicted.predicted_acc[:, 7:],
                                   torch.full((1, 28), 4., dtype=dtype), atol=.002, rtol=.001)
        torch.testing.assert_close(mass @ predicted.predicted_acc.unsqueeze(-1),
                                   torch.cat((torch.zeros((1, 6), dtype=dtype),
                                              predicted.total_joint_torque), dim=1).unsqueeze(-1),
                                   atol=1e-5, rtol=1e-5)

    def test_servo_feedback_is_not_canceled_and_total_effort_is_bounded(self):
        dtype = torch.float64
        mass = torch.eye(35, dtype=dtype).unsqueeze(0)
        bias = torch.zeros((1, 35), dtype=dtype)
        bias[:, 6:] = 10.
        servo_bias = torch.full((1, 29), 4., dtype=dtype)
        servo_gain = torch.full((1, 29), .5, dtype=dtype)
        def solve(limit, acceleration_limit=None):
            return G1DynamicsAccelerationQP(max_iter=4000).solve(
                acc_reference=torch.zeros((1, 35), dtype=dtype),
                mass_matrix=mass, bias_force=bias,
                contact_jacobian=torch.zeros((1, 12, 35), dtype=dtype),
                jdot_qdot=torch.zeros((1, 12), dtype=dtype),
                contact_mode=torch.zeros(1, dtype=torch.int64),
                effort_limits=limit, servo_bias=servo_bias, servo_acc_gain=servo_gain,
                predicted_acc_limit=acceleration_limit,
            )
        free = solve(100.)
        self.assertTrue(free.validated.all())
        # Nominal inverse dynamics retains ~10 Nm. PD adds ~4 Nm rather
        # than being canceled by reducing feedforward to ~6 Nm.
        torch.testing.assert_close(free.joint_torque, torch.full((1, 29), 10., dtype=dtype), atol=.002, rtol=.001)
        torch.testing.assert_close(free.predicted_acc[:, 6:], torch.full((1, 29), 4., dtype=dtype), atol=.002, rtol=.001)
        bounded = solve(5.)
        self.assertTrue(bounded.validated.all())
        self.assertLessEqual(bounded.total_joint_torque.abs().max().item(), 5.002)
        torch.testing.assert_close(bounded.total_joint_torque,
                                   bounded.joint_torque + servo_bias + servo_gain * bounded.acc[:, 6:])
        torch.testing.assert_close(bounded.predicted_acc[:, 6:] + bias[:, 6:],
                                   bounded.total_joint_torque, atol=1e-3, rtol=1e-3)
        limited_acc = solve(100., 2.)
        self.assertTrue(limited_acc.validated.all())
        self.assertLessEqual(limited_acc.predicted_acc.abs().max().item(), 2.002)

    def test_servo_response_preserves_active_contact(self):
        dtype = torch.float64
        mass = torch.eye(35, dtype=dtype).unsqueeze(0)
        jac = torch.zeros((1, 12, 35), dtype=dtype)
        jac[:, :6, :6] = torch.eye(6, dtype=dtype)
        jac[:, :6, 6:12] = .2 * torch.eye(6, dtype=dtype)
        result = G1DynamicsAccelerationQP(max_iter=2000).solve(
            acc_reference=torch.zeros((1, 35), dtype=dtype), mass_matrix=mass,
            bias_force=torch.zeros((1, 35), dtype=dtype), contact_jacobian=jac,
            jdot_qdot=torch.zeros((1, 12), dtype=dtype), contact_mode=torch.ones(1, dtype=torch.int64),
            servo_bias=torch.ones((1, 29), dtype=dtype), servo_acc_gain=torch.full((1, 29), .1, dtype=dtype),
            # Bound one TOTAL contact-force component, not its nominal value.
            friction_G=torch.cat((torch.eye(12, dtype=dtype)[:1], -torch.eye(12, dtype=dtype)[:1])),
            friction_b=torch.full((2,), .05, dtype=dtype),
        )
        self.assertTrue(result.validated.all())
        torch.testing.assert_close(jac @ result.predicted_acc.unsqueeze(-1), torch.zeros((1, 12, 1), dtype=dtype), atol=1e-6, rtol=0.)
        self.assertGreater(result.predicted_acc.abs().max().item(), .5)
        self.assertLessEqual(result.contact_wrench[:,0].abs().max().item(), .051)
        self.assertGreater((result.contact_wrench - result.nominal_contact_wrench).abs().max().item(), .1)

    def test_effort_limits_are_imposed_inside_qp(self):
        dtype = torch.float64
        mass = torch.eye(35, dtype=dtype).unsqueeze(0)
        bias = torch.zeros((1, 35), dtype=dtype)
        bias[:, 6:35] = 10.0
        jacobian = torch.zeros((1, 12, 35), dtype=dtype)
        jdot_qdot = torch.zeros((1, 12), dtype=dtype)
        reference = torch.zeros((1, 35), dtype=dtype)
        mode = torch.zeros(1, dtype=torch.int64)

        result = G1DynamicsAccelerationQP(
            max_iter=2000,
            validation_tolerance=1.0e-3,
        ).solve(
            acc_reference=reference,
            mass_matrix=mass,
            bias_force=bias,
            contact_jacobian=jacobian,
            jdot_qdot=jdot_qdot,
            contact_mode=mode,
            acc_lower=-20.0,
            acc_upper=20.0,
            effort_limits=5.0,
        )

        torque = result.acc[:, 6:35] + bias[:, 6:35]
        self.assertTrue(result.validated.all())
        self.assertLessEqual(torque.abs().max().item(), 5.001)
        self.assertLessEqual(result.effort_violation.max().item(), 1.0e-3)

    def test_effort_limits_must_be_positive(self):
        solver = G1DynamicsAccelerationQP()
        with self.assertRaisesRegex(ValueError, "must be positive"):
            solver.solve(
                acc_reference=torch.zeros((1, 35)),
                mass_matrix=torch.eye(35).unsqueeze(0),
                bias_force=torch.zeros((1, 35)),
                contact_jacobian=torch.zeros((1, 12, 35)),
                jdot_qdot=torch.zeros((1, 12)),
                contact_mode=torch.zeros(1, dtype=torch.int64),
                effort_limits=0.0,
            )

    def test_known_feedback_is_in_dynamics_and_total_effort_bounds(self):
        dtype = torch.float64
        mass = torch.eye(35, dtype=dtype).unsqueeze(0)
        bias = torch.zeros((1, 35), dtype=dtype)
        bias[:, 6:35] = 10.0
        known = torch.full((1, 29), 4.0, dtype=dtype)

        result = G1DynamicsAccelerationQP(
            max_iter=2000,
            validation_tolerance=1.0e-3,
        ).solve(
            acc_reference=torch.zeros((1, 35), dtype=dtype),
            mass_matrix=mass,
            bias_force=bias,
            contact_jacobian=torch.zeros((1, 12, 35), dtype=dtype),
            jdot_qdot=torch.zeros((1, 12), dtype=dtype),
            contact_mode=torch.zeros(1, dtype=torch.int64),
            acc_lower=-20.0,
            acc_upper=20.0,
            effort_limits=5.0,
            known_joint_torque=known,
        )

        expected_total = result.acc[:, 6:35] + bias[:, 6:35]
        torch.testing.assert_close(
            result.total_joint_torque,
            result.joint_torque + known,
        )
        torch.testing.assert_close(
            result.total_joint_torque,
            expected_total,
            atol=1.0e-3,
            rtol=1.0e-3,
        )
        self.assertLessEqual(
            result.total_joint_torque.abs().max().item(), 5.001
        )
        self.assertTrue(result.validated.all())


if __name__ == "__main__":
    unittest.main()
