import unittest

import torch

from fabrics_sim.solver.momentum_bounds import momentum_acceleration_bounds


class MomentumBoundTests(unittest.TestCase):
    def test_matches_direct_rate_and_next_state_checks_with_bias(self):
        g=torch.Generator().manual_seed(5)
        A=torch.randn(2,2,5,generator=g,dtype=torch.float64)
        bias=torch.tensor([[2.,-1.],[.5,3.]],dtype=torch.float64)
        H=torch.tensor([[3.98,-3.97],[1.,-2.]],dtype=torch.float64)
        C,d=momentum_acceleration_bounds(A,bias,H,.002,4.,30.)
        for _ in range(100):
            acc=20*torch.randn(2,5,generator=g,dtype=torch.float64)
            rate=(A@acc[...,None]).squeeze(-1)+bias
            after=H+.002*rate
            expected=((rate.abs()<=30.).all(1)&(after.abs()<=4.).all(1))
            actual=((C@acc[...,None]).squeeze(-1)<=d).all(1)
            self.assertTrue(torch.equal(actual,expected))

    def test_unrecoverable_measured_state_is_not_silently_clipped(self):
        A=torch.eye(2)[None];H=torch.tensor([[4.1,0.]])
        C,d=momentum_acceleration_bounds(A,torch.zeros_like(H),H,.002,4.,30.)
        # Returning from 4.1 to 4.0 in one tick requires -50, beyond -30.
        self.assertLess(float(d[0,0]),-30.)
        self.assertEqual(float(d[0,2]),30.)
        torch.testing.assert_close(H,torch.tensor([[4.1,0.]]))

    def test_invalid_inputs_are_rejected(self):
        A=torch.eye(2)[None];H=torch.zeros(1,2)
        for dt in (0.,float('nan')):
            with self.assertRaises(ValueError):momentum_acceleration_bounds(A,H,H,dt,4.,30.)
        with self.assertRaises(ValueError):momentum_acceleration_bounds(A,H,H[:,:1],.002,4.,30.)


if __name__=='__main__':unittest.main()
