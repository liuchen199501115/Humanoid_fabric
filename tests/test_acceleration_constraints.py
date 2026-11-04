import unittest
import torch

from fabrics_sim.solver.AccelerationQP import G1DynamicsAccelerationQP


class PhysicalConstraintTests(unittest.TestCase):
    def solve(self, servo=False, effort=5., backend='admm'):
        dtype=torch.float64
        C=torch.zeros(1,1,35,dtype=dtype);C[:,:,6]=-1.
        target=torch.zeros(1,35,dtype=dtype);target[:,6]=-2.
        extra = dict(servo_bias=torch.full((1,29),.1,dtype=dtype),
                     servo_acc_gain=torch.full((1,29),.2,dtype=dtype)) if servo else {}
        return G1DynamicsAccelerationQP(max_iter=10000,rho=1.,eps_abs=1e-8,eps_rel=1e-8,
            solver_backend=backend).solve(
                acc_reference=target,mass_matrix=torch.eye(35,dtype=dtype)[None],
                bias_force=torch.zeros(1,35,dtype=dtype),contact_jacobian=torch.zeros(1,12,35,dtype=dtype),
                jdot_qdot=torch.zeros(1,12,dtype=dtype),contact_mode=torch.zeros(1,dtype=torch.long),
                acceleration_constraints=(C,torch.full((1,1),-.7,dtype=dtype)),
                effort_limits=effort,acc_lower=-10.,acc_upper=10.,**extra)

    def test_physical_lower_bound_overrides_conflicting_tracking(self):
        for backend in ('admm','osqp'):
            for servo in (False,True):
                with self.subTest(backend=backend,servo=servo):
                    result=self.solve(servo=servo,backend=backend)
                    self.assertTrue(result.validated.all())
                    self.assertAlmostEqual(result.predicted_acc[0,6].item(),.7,places=4)
                    if servo:
                        self.assertGreater(abs(result.acc[0,6]-result.predicted_acc[0,6]).item(),.1)

    def test_infeasible_constraint_is_not_silently_discarded(self):
        self.assertFalse(self.solve(effort=.1).validated.all())


if __name__=='__main__':unittest.main()
