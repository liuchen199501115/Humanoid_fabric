import unittest
import numpy as np
from analyze_com_reachability import closest_task


class ReachabilityTests(unittest.TestCase):
    def test_hard_bound_limits_reachable_target(self):
        r=closest_task(np.array([[1.]]),np.array([1.]),np.zeros((0,1)),np.zeros(0),
                       np.array([-2.]),np.array([2.]),np.array([[1.]]),np.array([2.]))
        self.assertFalse(r['exact_target_feasible'])
        self.assertAlmostEqual(r['minimum_max_error'],1.)
        self.assertAlmostEqual(r['constraint_violation'],0.)

    def test_feasible_target_is_reached(self):
        r=closest_task(np.array([[1.]]),np.array([1.]),np.zeros((0,1)),np.zeros(0),
                       np.array([-2.]),np.array([2.]),np.array([[1.]]),np.array([.3]))
        self.assertTrue(r['exact_target_feasible'])
        self.assertAlmostEqual(r['minimum_max_error'],0.)


if __name__=='__main__':unittest.main()
