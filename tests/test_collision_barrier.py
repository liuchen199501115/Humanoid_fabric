import unittest
import torch
from fabrics_sim.solver.collision_barrier import wrist_collision_barrier

class CollisionBarrierTests(unittest.TestCase):
    def test_default_matches_original_bounds(self):
        distance=torch.tensor([[.03,.10,.50,.05]])
        jacobian=torch.arange(140,dtype=torch.float32).reshape(1,4,35)
        bias=torch.tensor([[1.,2.,3.,4.]])
        speed=torch.tensor([[.1,-.2,.3,.4]])
        C,b=wrist_collision_barrier(distance,jacobian,bias,speed,1,3,.02)
        torch.testing.assert_close(C,-jacobian[:,1:2])
        torch.testing.assert_close(b,(bias+16*speed+64*(distance-.02))[:,1:2])

    def test_selected_rate_preserves_mask_and_barrier_formula(self):
        distance=torch.tensor([[.03,.10,.50,.05]])
        jacobian=torch.ones(1,4,35)
        bias=torch.ones(1,4);speed=-torch.ones(1,4)*.1
        C,b=wrist_collision_barrier(distance,jacobian,bias,speed,1,3,.02,True,16.)
        torch.testing.assert_close(C[:,0],-jacobian[:,1])
        torch.testing.assert_close(C[:,1],torch.zeros(1,35))
        torch.testing.assert_close(b[:,0],(bias+32*speed+256*(distance-.02))[:,1])
        torch.testing.assert_close(b[:,1],torch.zeros(1))

    def test_invalid_rate_is_rejected(self):
        x=torch.zeros(1,2)
        for rate in [0.,-1.,float('nan'),float('inf')]:
            with self.subTest(rate=rate),self.assertRaises(ValueError):
                wrist_collision_barrier(x,torch.zeros(1,2,35),x,x,0,2,.02,bandwidth=rate)
