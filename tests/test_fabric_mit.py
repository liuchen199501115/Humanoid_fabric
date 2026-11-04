import unittest
from types import SimpleNamespace
import torch
from fabrics_sim.solver.fabric_mit import configure_fabric_mit
from fabrics_sim.solver.joint_control import integrated_servo_model


class FabricMITTests(unittest.TestCase):
    def test_mode_retains_integrator_and_physical_velocity_limits(self):
        args = SimpleNamespace(direct_wbc=True, tracking_servo=True,
                               joint_velocity_limits=True, acceleration_and_braking_limits=False)
        configure_fabric_mit(args)
        self.assertFalse(args.direct_wbc)
        self.assertFalse(args.tracking_servo)
        self.assertTrue(args.joint_velocity_limits)
        self.assertFalse(args.acceleration_and_braking_limits)
        self.assertEqual(args.control_frequency, 1000)

    def test_composition_includes_bias_and_late_leaves(self):
        from fabrics_sim.solver.fabric_mit import compose_task_acceleration
        from fabrics_sim.fabrics.task_request import TaskRequest
        v = torch.zeros(1,35,dtype=torch.float64)
        A = torch.zeros(1,1,35,dtype=v.dtype)
        A[:,:,6] = 2.
        leaf = TaskRequest('late', A, torch.tensor([[1.]],dtype=v.dtype),
                           torch.tensor([[5.]],dtype=v.dtype),
                           torch.tensor([[[3.]]],dtype=v.dtype),
                           torch.ones(1,1,dtype=torch.bool))
        result = compose_task_acceleration([leaf],v,.1)
        self.assertAlmostEqual(result[0,6].item(),24/12.1)
        self.assertEqual(torch.count_nonzero(result).item(),1)

    def test_root_preserves_leaf_cost_differences_and_metric(self):
        from fabrics_sim.solver.fabric_mit import compose_task_policy
        from fabrics_sim.fabrics.task_request import TaskRequest
        torch.manual_seed(12)
        v=torch.zeros(1,35,dtype=torch.float64)
        requests=[]
        for rows, weight in ((3,250.),(6,20.),(29,.5)):
            A=torch.randn(1,rows,35,dtype=v.dtype)
            requests.append(TaskRequest(str(rows),A,torch.randn(1,rows,dtype=v.dtype),
                torch.randn(1,rows,dtype=v.dtype),weight*torch.eye(rows,dtype=v.dtype)[None],
                torch.ones(1,rows,dtype=torch.bool)))
        acc,root=compose_task_policy(requests,v,1e-4)
        expected=1e-4*torch.eye(35,dtype=v.dtype)[None]
        for leaf in requests:
            expected=expected+leaf.natural_form()[0]
        torch.testing.assert_close(root.metric,expected)
        a,b=torch.randn_like(v),torch.randn_like(v)
        def cost(task,x):
            r=task.residual(x)
            return (r[:,None]@task.metric@r[:,:,None]).sum()
        old=sum(cost(t,a)-cost(t,b) for t in requests)+1e-4*(a.square().sum()-b.square().sum())
        torch.testing.assert_close(cost(root,a)-cost(root,b),old)
        torch.testing.assert_close(root.residual(acc),torch.zeros_like(v))

    def test_float32_composition_remains_exactly_symmetric(self):
        from fabrics_sim.solver.fabric_mit import compose_task_policy
        from fabrics_sim.fabrics.task_request import TaskRequest
        torch.manual_seed(43)
        v=torch.zeros(1,35)
        A=torch.randn(1,29,35)*3
        leaf=TaskRequest('posture',A,torch.zeros(1,29),torch.randn(1,29),
                         torch.diag_embed(torch.logspace(-2,2,29)[None]),
                         torch.ones(1,29,dtype=torch.bool))
        _,root=compose_task_policy([leaf],v,1e-4)
        self.assertTrue(torch.equal(root.metric,root.metric.transpose(-1,-2)))
        self.assertTrue(torch.isfinite(root.desired).all())

    def test_double_root_with_float_physics_and_mit_feedback(self):
        from fabrics_sim.solver.fabric_mit import compose_task_policy
        from fabrics_sim.fabrics.task_request import TaskRequest
        from fabrics_sim.solver.AccelerationQP import G1DynamicsAccelerationQP
        v=torch.zeros(1,35)
        target=v.clone();target[:,6:]=1.
        eye=torch.eye(35)[None]
        leaf=TaskRequest('all',eye,v,target,eye,torch.ones_like(v,dtype=torch.bool))
        acc,root=compose_task_policy([leaf],v,1e-4)
        correction=torch.zeros_like(v);correction[:,6:]=.3
        result=G1DynamicsAccelerationQP(solver_backend='daqp').solve(
            acc_reference=acc+correction,command_acc_correction=correction,mass_matrix=eye,bias_force=v,
            contact_jacobian=torch.zeros(1,12,35),jdot_qdot=torch.zeros(1,12),
            contact_mode=torch.zeros(1,dtype=torch.long),effort_limits=10.,
            servo_bias=torch.full((1,29),.2),servo_acc_gain=torch.full((1,29),.1),
            task_requests=[root],task_tracks_predicted=True)
        self.assertTrue(result.validated.all())
        torch.testing.assert_close(result.nominal_acc,result.acc-correction)
        torch.testing.assert_close(result.predicted_acc[:,6:],target[:,6:],atol=.002,rtol=0.)
        self.assertLess(result.task_residuals['fabric_composed'].abs().max().item(),.002)

    def test_rejects_direct_wbc_probe(self):
        with self.assertRaises(ValueError):
            configure_fabric_mit(SimpleNamespace(mocap_unload_test=True))

    def test_persistent_integrated_torque_matches_solver_budget(self):
        q = torch.zeros(1,29,dtype=torch.float64)
        v = q + .2
        measured_q, measured_v = q + .01, v - .03
        kp, kd, dt = q+40, q+2, .001
        # Multiple ticks preserve history, including nonzero servo error.
        for i in range(5):
            acc = q + (i+1)*.1
            bias, gain = integrated_servo_model(q,v,measured_q,measured_v,kp,kd,dt)
            v = v + dt*acc
            q = q + dt*v
            actual_pd = kp*(q-measured_q)+kd*(v-measured_v)
            torch.testing.assert_close(actual_pd,bias+gain*acc)


if __name__ == '__main__':
    unittest.main()
