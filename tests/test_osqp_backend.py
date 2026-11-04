import unittest

import torch

from fabrics_sim.solver.admm_qp import QPData
from fabrics_sim.solver.osqp_backend import OSQPBackend
from fabrics_sim.solver.AccelerationQP import G1DynamicsAccelerationQP


class OSQPBackendTests(unittest.TestCase):
    def test_reset_clears_native_and_predicted_warm_starts_per_environment(self):
        solver=OSQPBackend(reuse_workspace=True,active_set_predictor=True)
        for name in ('priority','tracking'):
            for index in (0,1):
                solver._workspaces[(name,index)]=object()
                solver._previous_solutions[(name,index)]=object()
        solver.reset(torch.tensor([0]))
        self.assertEqual(set(solver._workspaces),{('priority',1),('tracking',1)})
        self.assertEqual(set(solver._previous_solutions),set(solver._workspaces))
        solver.reset()
        self.assertFalse(solver._workspaces)
        self.assertFalse(solver._previous_solutions)

    def test_active_set_prediction_and_fallback_keep_original_problem(self):
        qp=QPData(torch.eye(2,dtype=torch.float64)[None],
            torch.tensor([[-2.,-1.]],dtype=torch.float64),
            torch.tensor([[[1.,0.]]],dtype=torch.float64),torch.tensor([[.5]],dtype=torch.float64),
            torch.full((1,2),-10.,dtype=torch.float64),torch.full((1,2),10.,dtype=torch.float64))
        solver=OSQPBackend(active_set_predictor=True)
        solver.solve(qp)
        qp.h[0,1]=-1.2
        warm=solver.solve(qp)
        self.assertEqual(solver.last_status,['active-set KKT certified'])
        torch.testing.assert_close(warm.z,OSQPBackend().solve(qp).z)
        qp.h[0,0]=0.
        released=solver.solve(qp)
        torch.testing.assert_close(released.z,OSQPBackend().solve(qp).z,atol=1e-6,rtol=1e-6)
        qp.b.fill_(-11.)
        impossible=solver.solve(qp)
        self.assertNotIn('active-set KKT certified',solver.last_status)
        self.assertFalse(impossible.status_ok.any())

    def test_workspace_updates_preserve_constraints_and_separate_priorities(self):
        qp = QPData(torch.eye(2, dtype=torch.float64)[None],
                    torch.zeros(1, 2, dtype=torch.float64),
                    torch.tensor([[[1., 0.]]], dtype=torch.float64),
                    torch.tensor([[.25]], dtype=torch.float64),
                    torch.full((1, 2), -10., dtype=torch.float64),
                    torch.full((1, 2), 10., dtype=torch.float64),
                    torch.ones(1, 1, 2, dtype=torch.float64),
                    torch.ones(1, 1, dtype=torch.float64))
        solver = OSQPBackend(reuse_workspace=True)
        solver.solve(qp, workspace='priority')
        first = solver._workspaces[('priority', 0)][0]
        solver.solve(qp, workspace='tracking')
        self.assertIsNot(first, solver._workspaces[('tracking', 0)][0])
        qp.b.fill_(.1)
        qp.h[0, 1] = -.2
        warm = solver.solve(qp, workspace='priority')
        cold = OSQPBackend().solve(qp)
        self.assertIs(first, solver._workspaces[('priority', 0)][0])
        self.assertTrue(warm.status_ok.all())
        torch.testing.assert_close(warm.z, cold.z, atol=1e-6, rtol=1e-6)
        qp.G[0, 0, 1] = 1.  # Sparsity changed: old workspace must be replaced.
        qp.b.fill_(1.5)
        solver.solve(qp, workspace='priority')
        self.assertIsNot(first, solver._workspaces[('priority', 0)][0])
        grown = solver._workspaces[('priority', 0)][0]
        qp.G[0, 0, 1] = 0.
        qp.b.fill_(.1)
        reused = solver.solve(qp, workspace='priority')
        self.assertIs(grown, solver._workspaces[('priority', 0)][0])
        self.assertTrue(solver.last_workspace_reused[0])
        torch.testing.assert_close(reused.z, OSQPBackend().solve(qp).z, atol=1e-6, rtol=1e-6)
        qp.G[0, 0, 1] = 1.
        qp.b.fill_(.5)  # Contradicts the unchanged equality x+y=1.
        impossible = solver.solve(qp, workspace='priority')
        self.assertFalse(impossible.status_ok.any())
        self.assertIn('cold restart', solver.last_status[0])

    def test_nullspace_refinement_preserves_fixed_rows_and_optimizes_free_joint(self):
        import numpy as np
        from fabrics_sim.solver.qp_refinement import _nullspace_refinement
        p=dict(H=np.eye(3), h=np.array([0.,-2.,0.]),
               G=np.array([[1.,0.,0.],[-1.,0.,0.],[0.,1.,0.]]),
               b=np.array([.3,-.3,1.]), lb=np.array([-10.,-10.,0.]),
               ub=np.array([10.,10.,0.]),
               A_eq=np.array([[1.,0.,0.],[0.,0.,1.],[2.,0.,0.]]),
               b_eq=np.array([.3,0.,.6]))
        x,_=_nullspace_refinement(p,1e-6)
        np.testing.assert_allclose(x,[.3,1.,0.],atol=1e-8)
        # An incompatible fixed-coordinate inequality must remain rejected.
        p['b'][0]=.2
        self.assertIsNone(_nullspace_refinement(p,1e-6))

    def test_iteration_limit_refinement_keeps_original_constraints(self):
        qp=QPData(torch.eye(2,dtype=torch.float64)[None],torch.zeros(1,2,dtype=torch.float64),
            torch.tensor([[[1.,0.]]],dtype=torch.float64),torch.tensor([[.25]],dtype=torch.float64),
            torch.full((1,2),-10.,dtype=torch.float64),torch.full((1,2),10.,dtype=torch.float64),
            torch.ones(1,1,2,dtype=torch.float64),torch.ones(1,1,dtype=torch.float64))
        solver=OSQPBackend(max_iter=1)
        result=solver.solve(qp)
        self.assertIn('SLSQP refined',solver.last_status[0])
        self.assertTrue(result.status_ok.all())
        torch.testing.assert_close(result.z,torch.tensor([[.25,.75]],dtype=torch.float64))

    def test_refinement_checks_discarded_dependent_equalities(self):
        import numpy as np
        from fabrics_sim.solver.qp_refinement import refine_qp
        p=dict(H=np.eye(2),h=np.zeros(2),G=np.empty((0,2)),b=np.empty(0),
               lb=np.full(2,-10.),ub=np.full(2,10.),
               A_eq=np.array([[1.,1.],[2.,2.]]),b_eq=np.array([1.,3.]))
        self.assertIsNone(refine_qp(p,np.zeros(2)))
        p['b_eq'][1]=2.
        x,_=refine_qp(p,np.zeros(2))
        np.testing.assert_allclose(x,[.5,.5],atol=1e-8)

    def test_batch_equality_and_active_inequality(self):
        # Minimize ||x||^2 with x0+x1=1, x0<=0.25 in one environment;
        # the second environment is unconstrained at its optimum (0.5,0.5).
        qp = QPData(torch.eye(2, dtype=torch.float64)[None].repeat(2, 1, 1),
                    torch.zeros(2, 2, dtype=torch.float64),
                    torch.tensor([[[1., 0.]], [[1., 0.]]], dtype=torch.float64),
                    torch.tensor([[.25], [1.]], dtype=torch.float64),
                    torch.full((2, 2), -10., dtype=torch.float64),
                    torch.full((2, 2), 10., dtype=torch.float64),
                    torch.ones(2, 1, 2, dtype=torch.float64),
                    torch.ones(2, 1, dtype=torch.float64))
        result = OSQPBackend().solve(qp)
        self.assertTrue(result.status_ok.all())
        torch.testing.assert_close(result.z, torch.tensor([[.25, .75], [.5, .5]], dtype=torch.float64))
        self.assertEqual(result.lam_eq.shape, (2, 1))

    def test_infeasible_dynamics_cannot_pass_even_with_loose_validation(self):
        # Unactuated base acceleration fixed at zero cannot balance this load
        # without contacts; loose caller tolerances must not accept a fallback.
        result = G1DynamicsAccelerationQP(solver_backend='osqp', validation_tolerance=100.).solve(
            acc_reference=torch.zeros(1, 35), mass_matrix=torch.eye(35)[None],
            bias_force=torch.ones(1, 35), contact_jacobian=torch.zeros(1, 12, 35),
            jdot_qdot=torch.zeros(1, 12), contact_mode=torch.zeros(1, dtype=torch.long),
            acc_lower=0., acc_upper=0., effort_limits=10.)
        self.assertFalse(result.validated.any())
        self.assertFalse(result.converged.any())


if __name__ == '__main__':
    unittest.main()
