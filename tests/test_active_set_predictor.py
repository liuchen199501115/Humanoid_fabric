import unittest
import numpy as np
from fabrics_sim.solver.active_set_predictor import predict_active_set,polish_dual_active_set


class ActiveSetPredictorTests(unittest.TestCase):
    def test_dual_polish_handles_large_multipliers_and_dependent_equalities(self):
        p=dict(H=np.diag([1e8,1.,1.]),h=np.array([0.,-1e6,0.]),
            G=np.array([[-1.,0.,0.]]),b=np.array([-.25]),
            lb=np.array([-10.,-10.,0.]),ub=np.array([10.,10.,0.]),
            A_eq=np.array([[1.,1.,0.],[2.,2.,0.],[0.,0.,1.]]),b_eq=np.array([1.,2.,0.]))
        lam=np.array([0.,0.,0.,0.,0.,0.,.26])
        result=polish_dual_active_set(p,lam)
        self.assertIsNotNone(result)
        x,dual=result
        np.testing.assert_allclose(x,[.25,.75,0.],atol=1e-10)
        np.testing.assert_allclose(p['A_eq']@x,p['b_eq'],atol=1e-10)
        # Full original stationarity, with the positive lower-bound multiplier.
        gradient=p['H']@x+p['h']+p['A_eq'].T@dual
        self.assertGreater(gradient[0],0.)
        np.testing.assert_allclose(gradient[1:],0.,atol=1e-7)
        p['b_eq'][1]=2.1
        self.assertIsNone(polish_dual_active_set(p,lam))

    def test_dual_polish_rejects_wrong_active_set_and_nonconvex_problem(self):
        p=self.problem()
        # Claim the LOWER variable bound x=-10 is active, contrary to its cost.
        self.assertIsNone(polish_dual_active_set(p,np.array([-1.,0.,0.])))
        p['H'][0,0]=-1.
        self.assertIsNone(polish_dual_active_set(p,np.array([0.,0.,1.])))

    def problem(self):
        return dict(H=np.eye(2), h=np.array([-2.,-1.]),
                    G=np.array([[1.,0.]]), b=np.array([.5]),
                    lb=np.full(2,-10.), ub=np.full(2,10.),
                    A_eq=np.empty((0,2)), b_eq=np.empty(0))

    def test_adjacent_problem_preserves_optimum_with_same_active_set(self):
        p=self.problem(); p['h'][1]=-1.2
        x,_=predict_active_set(p,np.array([.5,1.]))
        np.testing.assert_allclose(x,[.5,1.2],atol=1e-10)

    def test_newly_active_or_released_constraint_requires_fallback(self):
        p=self.problem()
        self.assertIsNone(predict_active_set(p,np.array([0.,1.])))
        p['h'][0]=0.
        self.assertIsNone(predict_active_set(p,np.array([.5,1.])))

    def test_bounded_corrections_enter_and_release_constraints(self):
        p=self.problem()
        x,_=predict_active_set(p,np.array([0.,1.]),max_updates=2)
        np.testing.assert_allclose(x,[.5,1.],atol=1e-10)
        p['h'][0]=0.
        x,_=predict_active_set(p,np.array([.5,1.]),max_updates=2)
        np.testing.assert_allclose(x,[0.,1.],atol=1e-10)
        p['b'][0]=-11.  # Incompatible with the unchanged lower bound -10.
        self.assertIsNone(predict_active_set(p,np.array([.5,1.]),max_updates=4))

    def test_fixed_variables_and_dependent_equalities(self):
        p=self.problem(); p['lb'][1]=p['ub'][1]=1.
        p['A_eq']=np.array([[0.,1.],[0.,2.]])
        p['b_eq']=np.array([1.,2.])
        x,dual=predict_active_set(p,np.array([.5,1.]))
        np.testing.assert_allclose(x,[.5,1.],atol=1e-10)
        self.assertEqual(dual.shape,(2,))
        p['b_eq'][1]=3.
        self.assertIsNone(predict_active_set(p,np.array([.5,1.])))

    def test_nonconvex_or_invalid_seed_requires_fallback(self):
        p=self.problem()
        self.assertIsNone(predict_active_set(p,None))
        p['H'][1,1]=-1.
        self.assertIsNone(predict_active_set(p,np.array([.5,1.])))


if __name__=='__main__': unittest.main()
