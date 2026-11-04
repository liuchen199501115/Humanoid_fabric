import unittest
import importlib.util
from unittest.mock import patch

import torch

from fabrics_sim.fabrics.task_request import TaskRequest
from fabrics_sim.solver.AccelerationQP import G1DynamicsAccelerationQP


class TaskPriorityTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('daqp'), 'Optional daqp dependency')
    def test_separate_priority_leaf_overrides_conflicting_root_without_duplicate_objective(self):
        dtype=torch.float64
        A=torch.zeros(1,1,35,dtype=dtype);A[0,0,6]=1.
        def task(name,desired,weight):
            return TaskRequest(name,A,torch.zeros(1,1,dtype=dtype),
                torch.tensor([[desired]],dtype=dtype),torch.tensor([[[weight]]],dtype=dtype),
                torch.ones(1,1,dtype=torch.bool))
        solver=G1DynamicsAccelerationQP(solver_backend='daqp')
        result=solver.solve(acc_reference=torch.zeros(1,35,dtype=dtype),
            mass_matrix=torch.eye(35,dtype=dtype)[None],bias_force=torch.zeros(1,35,dtype=dtype),
            contact_jacobian=torch.zeros(1,12,35,dtype=dtype),jdot_qdot=torch.zeros(1,12,dtype=dtype),
            contact_mode=torch.zeros(1,dtype=torch.long),
            task_requests=[task('root',-4.,10000.)],
            priority_requests=[task('center_of_mass',2.,250.)],
            priority_tasks=('center_of_mass',),effort_limits=1.5,
            acc_lower=-10.,acc_upper=10.,priority_tolerance=.001)
        self.assertTrue(result.validated.all())
        self.assertAlmostEqual(float(result.predicted_acc[0,6]),1.5,delta=.00102)
        self.assertEqual(set(result.task_residuals),{'root'})
        self.assertLessEqual(float(result.total_joint_torque.abs().max()),1.5+1e-6)

    @unittest.skipUnless(importlib.util.find_spec('daqp'), 'Optional daqp dependency')
    def test_body_balance_stage_preserves_attitude_against_source_momentum(self):
        dtype=torch.float64
        A=torch.zeros(1,2,35,dtype=dtype);A[0,0,6]=1.;A[0,1,7]=1.
        tasks=[TaskRequest(name,rows,torch.zeros(1,rows.shape[1],dtype=dtype),
            torch.tensor([target],dtype=dtype),weight*torch.eye(rows.shape[1],dtype=dtype)[None],
            torch.ones(1,rows.shape[1],dtype=torch.bool)) for name,rows,target,weight in (
                ('center_of_mass',A[:,:1],[1.],250.),
                ('body_orientation',A[:,1:],[-2.],40.),
                ('centroidal_momentum',A,[4.,4.],20.),
                ('joint_posture',A,[-4.,4.],10000.))]
        solver=G1DynamicsAccelerationQP(solver_backend='daqp')
        result=solver.solve(acc_reference=torch.zeros(1,35,dtype=dtype),
            mass_matrix=torch.eye(35,dtype=dtype)[None],bias_force=torch.zeros(1,35,dtype=dtype),
            contact_jacobian=torch.zeros(1,12,35,dtype=dtype),jdot_qdot=torch.zeros(1,12,dtype=dtype),
            contact_mode=torch.zeros(1,dtype=torch.long),task_requests=tasks,
            priority_tasks=tuple(task.name for task in tasks),
            priority_levels=(('center_of_mass','body_orientation'),('centroidal_momentum','joint_posture')),
            effort_limits=1.5,acc_lower=-10.,acc_upper=10.,priority_tolerance=.001)
        self.assertTrue(result.validated.all())
        torch.testing.assert_close(result.predicted_acc[0,6:8],
            torch.tensor([1.,-1.5],dtype=dtype),atol=.00102,rtol=0.)
        self.assertLessEqual(float(result.total_joint_torque.abs().max()),1.5+1e-6)

    @unittest.skipUnless(importlib.util.find_spec('daqp'), 'Optional daqp dependency')
    def test_centroidal_stage_preserves_linear_and_angular_tasks_before_pose(self):
        dtype = torch.float64
        rows = torch.zeros(1, 2, 35, dtype=dtype)
        rows[0, 0, 6] = rows[0, 1, 7] = 1.
        tasks = [TaskRequest(name, A, torch.zeros(1, A.shape[1], dtype=dtype),
            torch.tensor([target], dtype=dtype), weight * torch.eye(A.shape[1], dtype=dtype)[None],
            torch.ones(1, A.shape[1], dtype=torch.bool)) for name, A, target, weight in (
                ('center_of_mass', rows[:, :1], [1.], 250.),
                ('centroidal_momentum', rows[:, 1:], [-2.], 20.),
                ('pose', rows, [-3., 3.], 10000.))]
        solver = G1DynamicsAccelerationQP(solver_backend='daqp')
        result = solver.solve(acc_reference=torch.zeros(1,35,dtype=dtype),
            mass_matrix=torch.eye(35,dtype=dtype)[None], bias_force=torch.zeros(1,35,dtype=dtype),
            contact_jacobian=torch.zeros(1,12,35,dtype=dtype), jdot_qdot=torch.zeros(1,12,dtype=dtype),
            contact_mode=torch.zeros(1,dtype=torch.long), task_requests=tasks,
            priority_tasks=('center_of_mass','centroidal_momentum','pose'),
            priority_levels=(('center_of_mass','centroidal_momentum'),('pose',)),
            priority_tolerance=.001, effort_limits=1.5, acc_lower=-10., acc_upper=10.)
        self.assertTrue(result.validated.all())
        # Momentum's unreachable -2 request yields to the physical effort
        # bound; subsequent pose tracking cannot erase the achieved optimum.
        torch.testing.assert_close(result.predicted_acc[0,6:8],
            torch.tensor([1.,-1.5],dtype=dtype),atol=.00102,rtol=0.)
        self.assertLessEqual(float(result.total_joint_torque.abs().max()),1.5+1e-6)

    @unittest.skipUnless(importlib.util.find_spec('daqp'), 'Optional daqp dependency')
    def test_float32_rank_deficient_task_is_accumulated_in_double(self):
        generator=torch.Generator().manual_seed(19)
        A=10*torch.randn(1,3,35,generator=generator)
        task=TaskRequest('low_rank',A,torch.zeros(1,3),torch.zeros(1,3),
                         torch.eye(3)[None],torch.ones(1,3,dtype=torch.bool))
        solver=G1DynamicsAccelerationQP(solver_backend='daqp')
        result=solver.solve(acc_reference=torch.zeros(1,35),
            mass_matrix=torch.eye(35)[None],bias_force=torch.zeros(1,35),
            contact_jacobian=torch.zeros(1,12,35),jdot_qdot=torch.zeros(1,12),
            contact_mode=torch.zeros(1,dtype=torch.long),task_requests=[task],
            priority_tasks=('low_rank',),effort_limits=10.,acc_lower=-10.,acc_upper=10.)
        self.assertTrue(result.validated.all())
        self.assertEqual(result.acc.dtype,torch.float32)
        H=solver.last_priority_problem.H
        self.assertEqual(H.dtype,torch.float64)
        self.assertGreater(float(torch.linalg.eigvalsh(H).min()),.9*solver.hessian_regularization)
        expected=A.double().transpose(1,2)@A.double()
        expected+=solver.hessian_regularization*torch.eye(35,dtype=torch.float64)[None]
        torch.testing.assert_close(H[:,:35,:35],expected,atol=1e-11,rtol=1e-12)

    @unittest.skipUnless(importlib.util.find_spec('daqp'), 'Optional daqp dependency')
    def test_ordered_levels_preserve_balance_and_then_pose_with_servo(self):
        dtype = torch.float64
        rows = torch.zeros(1, 2, 35, dtype=dtype)
        rows[0, 0, 6] = rows[0, 1, 7] = 1.
        tasks = [TaskRequest(name, A, torch.zeros(1, A.shape[1], dtype=dtype),
            torch.tensor([target], dtype=dtype), weight * torch.eye(A.shape[1], dtype=dtype)[None],
            torch.ones(1, A.shape[1], dtype=torch.bool)) for name, A, target, weight in (
                ('balance', rows[:, :1], [1.], 1.),
                ('pose', rows, [-2., 2.], 100.),
                ('tracking', rows, [-3., -3.], 10000.))]
        solver = G1DynamicsAccelerationQP(solver_backend='daqp')
        kwargs = dict(acc_reference=torch.zeros(1,35,dtype=dtype),
            mass_matrix=torch.eye(35,dtype=dtype)[None], bias_force=torch.zeros(1,35,dtype=dtype),
            contact_jacobian=torch.zeros(1,12,35,dtype=dtype), jdot_qdot=torch.zeros(1,12,dtype=dtype),
            contact_mode=torch.zeros(1,dtype=torch.long), task_requests=tasks,
            priority_tasks=('balance','pose'), priority_levels=(('balance',),('pose',)),
            priority_tolerance=.001, effort_limits=10., acc_lower=-10., acc_upper=10.,
            servo_bias=torch.full((1,29),.1,dtype=dtype), servo_acc_gain=torch.full((1,29),.2,dtype=dtype))
        result = solver.solve(**kwargs)
        self.assertTrue(result.validated.all())
        # Later pose/tracking requests cannot accumulate drift in the first lock.
        self.assertLessEqual(abs(result.predicted_acc[0,6].item()-1.),.00102)
        self.assertLessEqual(abs(result.predicted_acc[0,7].item()-2.),.00102)
        self.assertEqual(len(solver.last_priority_problems),2)
        kwargs['priority_tasks'] += ('tracking',)
        kwargs['priority_levels'] = (('balance',),('pose',),('tracking',))
        result = solver.solve(**kwargs)
        self.assertTrue(result.validated.all())
        self.assertLessEqual(abs(result.predicted_acc[0,6].item()-1.),.00102)
        self.assertLessEqual(abs(result.predicted_acc[0,7].item()-2.),.00102)
        self.assertEqual(len(solver.last_priority_problems),3)
        # A valid final solve cannot hide a failed earlier certificate.
        native_solve = solver.solver.solve
        def failed_first(*args, **options):
            stage = native_solve(*args, **options)
            if options.get('workspace') == 'priority':
                stage.status_ok.fill_(False)
                stage.converged.fill_(False)
            return stage
        with patch.object(solver.solver, 'solve', side_effect=failed_first):
            result = solver.solve(**kwargs)
        self.assertFalse(result.validated.any())
        self.assertFalse(result.converged.any())
        kwargs['priority_levels']=(('balance',),('balance','pose'))
        with self.assertRaisesRegex(ValueError, 'partition'):
            solver.solve(**kwargs)

    @unittest.skipUnless(importlib.util.find_spec('daqp'), 'Optional daqp dependency')
    def test_daqp_keeps_task_locks_servo_coupling_and_effort_limits(self):
        for servo in (False,True):
            for effort in (.5,10.):
                result=self.solve(servo=servo,effort=effort,backend='daqp')
                self.assertTrue(result.validated.all())
                self.assertLessEqual(result.total_joint_torque.abs().max().item(),effort+1e-6)
                self.assertLessEqual(abs(result.predicted_acc[0,6].item()-min(1.,effort)),.00102)

    def test_active_set_fast_path_keeps_two_stage_dynamics_and_task_locks(self):
        dtype=torch.float64
        A=torch.zeros(1,1,35,dtype=dtype);A[:,:,6]=1.
        tasks=[TaskRequest(name,A,torch.zeros(1,1,dtype=dtype),
                torch.full((1,1),target,dtype=dtype),torch.full((1,1,1),weight,dtype=dtype),
                torch.ones(1,1,dtype=torch.bool))
               for name,target,weight in [('balance',1.,1.),('tracking',-2.,100.)]]
        solver=G1DynamicsAccelerationQP(solver_backend='osqp')
        solver.solver.active_set_predictor=True
        args=dict(acc_reference=torch.zeros(1,35,dtype=dtype),
            mass_matrix=torch.eye(35,dtype=dtype)[None],bias_force=torch.zeros(1,35,dtype=dtype),
            contact_jacobian=torch.zeros(1,12,35,dtype=dtype),jdot_qdot=torch.zeros(1,12,dtype=dtype),
            contact_mode=torch.zeros(1,dtype=torch.long),task_requests=tasks,priority_tasks=('balance',),
            priority_tolerance=.001,effort_limits=10.,acc_lower=-10.,acc_upper=10.,
            servo_bias=torch.full((1,29),.1,dtype=dtype),servo_acc_gain=torch.full((1,29),.2,dtype=dtype))
        solver.solve(**args)
        tasks[1].desired.fill_(-2.1)
        result=solver.solve(**args)
        self.assertTrue(result.validated.all())
        self.assertLessEqual(abs(result.predicted_acc[0,6].item()-1.),.00102)
        self.assertIn(('priority',0),solver.solver._previous_solutions)
        self.assertIn(('tracking',0),solver.solver._previous_solutions)
        self.assertEqual(solver.solver.last_status,['active-set KKT certified'])

    def solve(self, priority=True, effort=10., servo=False, active=True, backend='admm'):
        dtype = torch.float64
        A = torch.zeros(1, 1, 35, dtype=dtype)
        A[:, :, 6] = 1
        tasks = [TaskRequest(name, A, torch.zeros(1, 1, dtype=dtype),
            torch.full((1, 1), target, dtype=dtype),
            torch.full((1, 1, 1), weight, dtype=dtype), torch.ones(1, 1, dtype=torch.bool))
            for name, target, weight in [('balance', 1., 1.), ('tracking', -2., 100.)]]
        tasks[0].active.fill_(active)
        extra = dict(servo_bias=torch.full((1,29), .1, dtype=dtype),
                     servo_acc_gain=torch.full((1,29), .2, dtype=dtype)) if servo else {}
        return G1DynamicsAccelerationQP(max_iter=10000, rho=1., eps_abs=1e-8, eps_rel=1e-8,
                                       solver_backend=backend).solve(
            acc_reference=torch.zeros(1,35,dtype=dtype), mass_matrix=torch.eye(35,dtype=dtype)[None],
            bias_force=torch.zeros(1,35,dtype=dtype), contact_jacobian=torch.zeros(1,12,35,dtype=dtype),
            jdot_qdot=torch.zeros(1,12,dtype=dtype), contact_mode=torch.zeros(1,dtype=torch.long),
            task_requests=tasks, priority_tasks=('balance',) if priority else (),
            priority_tolerance=.001, effort_limits=effort, acc_lower=-10., acc_upper=10., **extra)

    def test_conflicting_tracking_cannot_displace_balance_optimum(self):
        weighted, prioritized = self.solve(False), self.solve()
        self.assertTrue(prioritized.validated.all())
        self.assertLess(weighted.predicted_acc[0,6], -1.9)
        self.assertAlmostEqual(prioritized.predicted_acc[0,6].item(), 1., places=4)

    def test_unreachable_balance_remains_soft_and_respects_motor_limit(self):
        result = self.solve(effort=.5)
        self.assertTrue(result.validated.all())
        self.assertAlmostEqual(result.predicted_acc[0,6].item(), .5, places=4)
        self.assertLessEqual(result.total_joint_torque.abs().max().item(), .50001)

    def test_priority_applies_to_physical_prediction_with_servo_feedback(self):
        result = self.solve(servo=True)
        self.assertTrue(result.validated.all())
        self.assertAlmostEqual(result.predicted_acc[0,6].item(), 1., places=4)
        self.assertGreater(abs(result.acc[0,6]-result.predicted_acc[0,6]).item(), .1)

    def test_disabled_priority_rows_do_not_lock_lower_priority_motion(self):
        result = self.solve(active=False)
        self.assertTrue(result.validated.all())
        self.assertAlmostEqual(result.predicted_acc[0,6].item(), -2., places=4)

    def test_osqp_preserves_priority_with_and_without_servo_feedback(self):
        for servo in (False, True):
            with self.subTest(servo=servo):
                result=self.solve(servo=servo, backend='osqp')
                self.assertTrue(result.validated.all())
                self.assertLessEqual(abs(result.predicted_acc[0,6].item()-1.),.00102)

    def test_height_only_leaves_horizontal_coordinates_available(self):
        dtype = torch.float64
        A = torch.zeros(1, 3, 35, dtype=dtype)
        A[:, :, 6:9] = torch.eye(3, dtype=dtype)
        tasks = [TaskRequest(name, A, torch.zeros(1, 3, dtype=dtype),
            torch.full((1, 3), target, dtype=dtype),
            weight * torch.eye(3, dtype=dtype)[None], torch.ones(1, 3, dtype=torch.bool))
            for name, target, weight in [('com', 1., 1.), ('foot', -2., 100.)]]
        for backend in ('admm', 'osqp'):
            with self.subTest(backend=backend):
                result = G1DynamicsAccelerationQP(max_iter=10000, rho=1., eps_abs=1e-8,
                    eps_rel=1e-8, solver_backend=backend).solve(
                    acc_reference=torch.zeros(1,35,dtype=dtype),
                    mass_matrix=torch.eye(35,dtype=dtype)[None],
                    bias_force=torch.zeros(1,35,dtype=dtype),
                    contact_jacobian=torch.zeros(1,12,35,dtype=dtype),
                    jdot_qdot=torch.zeros(1,12,dtype=dtype),
                    contact_mode=torch.zeros(1,dtype=torch.long),
                    task_requests=tasks, priority_tasks=('com',), priority_rows={'com': (2,)},
                    priority_tolerance=.001, effort_limits=10., acc_lower=-10., acc_upper=10.)
                self.assertTrue(result.validated.all())
                self.assertLessEqual(abs(result.predicted_acc[0,8].item()-1.), .00102)
                torch.testing.assert_close(result.predicted_acc[0,6:8],
                    torch.full((2,), -199/101, dtype=dtype), atol=1e-4, rtol=0.)
                self.assertTrue(tasks[0].active.all())

    def test_heading_projection_preserves_lateral_and_frees_forward(self):
        dtype = torch.float64
        A = torch.zeros(1, 3, 35, dtype=dtype)
        A[:, :, 6:9] = torch.eye(3, dtype=dtype)
        lateral = torch.tensor([-.8, .6, 0.], dtype=dtype)
        forward = torch.tensor([.6, .8, 0.], dtype=dtype)
        projection = torch.stack((lateral, torch.tensor([0., 0., 1.], dtype=dtype)))[None]
        tasks = [TaskRequest(name, A, torch.zeros(1, 3, dtype=dtype),
            torch.full((1, 3), target, dtype=dtype),
            weight * torch.eye(3, dtype=dtype)[None], torch.ones(1, 3, dtype=torch.bool))
            for name, target, weight in [('com', 1., 1.), ('foot', -2., 100.)]]
        for backend in ('admm', 'osqp'):
            result = G1DynamicsAccelerationQP(max_iter=10000, rho=1., eps_abs=1e-8,
                eps_rel=1e-8, solver_backend=backend).solve(
                acc_reference=torch.zeros(1,35,dtype=dtype),
                mass_matrix=torch.eye(35,dtype=dtype)[None],
                bias_force=torch.zeros(1,35,dtype=dtype),
                contact_jacobian=torch.zeros(1,12,35,dtype=dtype),
                jdot_qdot=torch.zeros(1,12,dtype=dtype),
                contact_mode=torch.zeros(1,dtype=torch.long),
                task_requests=tasks, priority_tasks=('com',),
                priority_projections={'com': projection}, priority_tolerance=.001,
                effort_limits=10., acc_lower=-10., acc_upper=10.)
            self.assertTrue(result.validated.all())
            acc = result.predicted_acc[0,6:9]
            self.assertLessEqual(abs((lateral @ acc).item() - lateral.sum().item()), .00102)
            self.assertLessEqual(abs(acc[2].item() - 1.), .00102)
            self.assertAlmostEqual((forward @ acc).item(), -199/101 * 1.4, places=4)
        tasks[0].active[:, 0] = False
        projected = tasks[0].project(projection)
        self.assertEqual(projected.A[0,0,6].item(), 0.)
        self.assertAlmostEqual(projected.desired[0,0].item(), .6)
        with self.assertRaises(ValueError):
            tasks[0].project(projection * 2)


if __name__ == '__main__':
    unittest.main()
