import unittest
import torch
from fabrics_sim.solver.joint_control import joint_velocity_constraints
from fabrics_sim.solver.AccelerationQP import G1DynamicsAccelerationQP


class JointVelocityConstraintTests(unittest.TestCase):
    def test_rows_are_exact_next_step_velocity_bounds_without_base_bounds(self):
        v=torch.linspace(-2.,2.,29,dtype=torch.float64)[None]
        limits=torch.linspace(1.,3.,29,dtype=torch.float64)
        C,d=joint_velocity_constraints(v,limits,.002)
        a=torch.linspace(-100.,100.,35,dtype=v.dtype)[None]
        residual=(C@a[...,None]).squeeze(-1)-d
        next_v=v+.002*a[:,6:]
        torch.testing.assert_close(residual,torch.cat((next_v-limits,-next_v-limits),dim=1))
        self.assertEqual(C[:,:,:6].count_nonzero(),0)

    def test_invalid_input_rejected(self):
        v=torch.zeros(1,29)
        for limits,dt in ((torch.zeros(29),.002),(torch.ones(28),.002),(torch.ones(29),0.),(torch.ones(29),float('nan'))):
            with self.assertRaises(ValueError):joint_velocity_constraints(v,limits,dt)
        v[0,0]=float('nan')
        with self.assertRaises(ValueError):joint_velocity_constraints(v,torch.ones(29),.002)

    def test_qp_limits_predicted_speed_in_both_directions_and_recovers_overspeed(self):
        for sign,initial in ((1.,.99),(-1.,-.99),(1.,1.01)):
            dtype=torch.float64
            v=torch.zeros(1,29,dtype=dtype);v[0,0]=initial
            target=torch.zeros(1,35,dtype=dtype);target[0,6]=sign*300.
            correction=torch.zeros_like(target);correction[:,6]=.5
            solver=G1DynamicsAccelerationQP(solver_backend='daqp')
            result=solver.solve(acc_reference=target,mass_matrix=torch.eye(35,dtype=dtype)[None],
                bias_force=torch.zeros_like(target),contact_jacobian=torch.zeros(1,12,35,dtype=dtype),
                jdot_qdot=torch.zeros(1,12,dtype=dtype),contact_mode=torch.tensor([0]),
                acc_lower=-float('inf'),acc_upper=float('inf'),effort_limits=200.,
                command_acc_correction=correction,
                acceleration_constraints=joint_velocity_constraints(v,torch.ones(29,dtype=dtype),.01))
            self.assertTrue(result.validated.all())
            self.assertLessEqual(float((v+.01*result.predicted_acc[:,6:]).abs().max()),1.+1e-5)
            self.assertLessEqual(float(result.total_joint_torque.abs().max()),200.+1e-5)

    def test_speed_bound_includes_predicted_servo_response(self):
        dtype=torch.float64
        v=torch.zeros(1,29,dtype=dtype);v[0,0]=.99
        target=torch.zeros(1,35,dtype=dtype);target[:,6:]=30.
        solver=G1DynamicsAccelerationQP(solver_backend='daqp')
        result=solver.solve(acc_reference=target,mass_matrix=torch.eye(35,dtype=dtype)[None],
            bias_force=torch.zeros_like(target),contact_jacobian=torch.zeros(1,12,35,dtype=dtype),
            jdot_qdot=torch.zeros(1,12,dtype=dtype),contact_mode=torch.tensor([0]),
            acc_lower=-float('inf'),acc_upper=float('inf'),effort_limits=50.,
            servo_bias=torch.full((1,29),8.,dtype=dtype),
            servo_acc_gain=torch.full((1,29),.5,dtype=dtype),
            acceleration_constraints=joint_velocity_constraints(v,torch.ones(29,dtype=dtype),.01))
        self.assertTrue(result.validated.all())
        self.assertLessEqual(float((v+.01*result.predicted_acc[:,6:]).abs().max()),1.+1e-5)
        self.assertGreater(float((result.predicted_acc-result.acc).abs().max()),1.)
