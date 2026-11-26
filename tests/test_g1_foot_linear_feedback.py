import unittest
import torch
from g1_foot_linear_feedback import linear_tracking_target


class FootLinearFeedbackTest(unittest.TestCase):
    def test_feedforward_preserved_and_feedback_norm_bounded(self):
        position=torch.tensor([[[1.,0.,0.],[0.,0.,0.]]],dtype=torch.float64)
        zero=torch.zeros_like(position);ff=torch.tensor([[[50.,-30.,20.],[2.,4.,6.]]],dtype=torch.float64)
        result=linear_tracking_target(position,zero,zero,zero,ff,torch.ones(2,3)*100,
                                      torch.ones(2,3)*20,15.)
        torch.testing.assert_close(result[0,0],ff[0,0]+torch.tensor([-15.,0.,0.]))
        torch.testing.assert_close(result[0,1],ff[0,1],rtol=0,atol=0)
        velocity=torch.tensor([[[3.,4.,0.],[0.,0.,0.]]],dtype=torch.float64)
        result=linear_tracking_target(zero,zero,velocity,zero,ff,torch.ones(2,3),torch.ones(2,3)*20,30.)
        torch.testing.assert_close((result-ff)[0,0],torch.tensor([-18.,-24.,0.],dtype=torch.float64))

    def test_invalid_cap_rejected(self):
        x=torch.zeros(1,2,3)
        for limit in [0.,-1.,float('nan'),float('inf')]:
            with self.assertRaises(ValueError):linear_tracking_target(x,x,x,x,x,x,x,limit)


if __name__=='__main__':unittest.main()
