import unittest
import numpy as np
from scipy import sparse
from g1_offline_bound_qp import solve_bound_qp


class OfflineBoundQPTest(unittest.TestCase):
    def test_mixed_bounds_and_signed_multipliers(self):
        # Independent optimum: x0+x1=1 and x1>=2 force (x0,x1)=(-1,2).
        # x2 is forced against its upper bound; final row is unbounded.
        P = sparse.eye(3, format='csc')
        q = np.array([0., 0., -4.])
        A = sparse.csc_matrix([[1.,1.,0.], [0.,1.,0.], [0.,0.,1.], [1.,0.,0.]])
        result = solve_bound_qp(P,q,A,np.array([1.,2.,-np.inf,-np.inf]),
                               np.array([1.,np.inf,1.,np.inf]))
        self.assertEqual(result.info.status_val, 1)
        np.testing.assert_allclose(result.x,[-1.,2.,1.],atol=1e-7)
        np.testing.assert_allclose(result.y,[1.,-3.,3.,0.],atol=1e-7)
        self.assertLess(result.info.prim_res,1e-7)
        self.assertLess(result.info.dual_res,1e-7)

    def test_infeasibility_is_not_success(self):
        result=solve_bound_qp(sparse.eye(1,format='csc'),np.zeros(1),
            sparse.csc_matrix([[1.],[1.]]),np.array([2.,-np.inf]),np.array([np.inf,1.]))
        self.assertEqual(result.info.status_val,-1)
        with self.assertRaises(ValueError):
            solve_bound_qp(sparse.eye(1,format='csc'),np.zeros(1),sparse.eye(1),
                           np.array([2.]),np.array([1.]))


if __name__=='__main__':unittest.main()
