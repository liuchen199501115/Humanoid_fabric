"""Equivalent scaling must preserve primal/dual solutions and original checks."""
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
import torch
from fabrics_sim.solver.daqp_backend import DAQPBackend


class EquilibrationTests(unittest.TestCase):
    def problem(self):
        # Known optimum [1, 2, -1]: active upper bound x0=1, equality
        # x1+x2=1, and inequality x1<=2. The free objective prefers [2,4,-1].
        t=lambda x:torch.tensor(x,dtype=torch.float64)[None]
        return SimpleNamespace(H=t(np.diag([4000.,1.,1e-6])),
            h=t([-8000.,-4.,1e-6]),G=t([[0.,1.,0.]]),b=t([2.]),
            lb=t([-10.,-10.,-10.]),ub=t([1.,10.,10.]),
            A_eq=t([[0.,1.,1.]]),b_eq=t([1.]))

    def test_bound_and_equality_duals_and_warm_start_in_original_units(self):
        qp=self.problem();base=DAQPBackend(single_solve=True,single_proximal=1e-4).solve(qp)
        solver=DAQPBackend(single_solve=True,single_proximal=1e-4,equilibration_cap=100.)
        for warm in [False,True]:
            with patch.object(solver.daqp,'solve',wraps=solver.daqp.solve) as native:
                result=solver.solve(qp)
            self.assertEqual(native.call_count,1)
            self.assertTrue(result.converged.all())
            torch.testing.assert_close(result.z,torch.tensor([[1.,2.,-1.]],dtype=torch.float64),atol=1e-7,rtol=0)
            torch.testing.assert_close(result.z,base.z,atol=1e-7,rtol=0)
            torch.testing.assert_close(result.lam_eq,base.lam_eq,atol=1e-7,rtol=0)
            self.assertEqual('primal_start' in native.call_args.kwargs,warm)
            if warm:
                self.assertLess(abs(native.call_args.kwargs['primal_start'][1]),.1)
            np.testing.assert_allclose(solver._previous_solutions[('tracking',0)],[1.,2.,-1.],atol=1e-7)

    def test_scaled_native_success_cannot_hide_original_constraint_violation(self):
        solver=DAQPBackend(single_solve=True,single_proximal=1e-4,equilibration_cap=100.)
        original=solver.daqp.solve
        def corrupted(*args,**kwargs):
            y,obj,flag,info=original(*args,**kwargs)
            y[1]+=1e-6  # 1e-4 violation in original units after undoing scaling.
            return y,obj,flag,info
        with patch.object(solver.daqp,'solve',side_effect=corrupted) as native:
            result=solver.solve(self.problem())
        self.assertFalse(result.converged.any())
        self.assertEqual(native.call_count,1)
        self.assertFalse(solver._previous_solutions)

    def test_invalid_scaling_rejected(self):
        for cap in [0.,.5,float('nan'),float('inf')]:
            with self.assertRaises(ValueError):DAQPBackend(equilibration_cap=cap)


if __name__=='__main__':unittest.main()
