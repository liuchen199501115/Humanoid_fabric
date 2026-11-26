import sys, unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'examples'))
from g1_foot_angular_feedback import angular_tracking_target


class FootAngularFeedbackTest(unittest.TestCase):
    def test_feedback_norm_cap_preserves_large_feedforward(self):
        R=torch.eye(3,dtype=torch.float64)[None,None].repeat(1,2,1,1)
        v=torch.tensor([[[3.,4.,0.],[0.,0.,0.]]],dtype=torch.float64)
        acceleration=torch.tensor([[[10.,20.,30.],[100.,200.,300.]]],dtype=torch.float64)
        actual=angular_tracking_target(R,R,v,torch.zeros_like(v),acceleration,
                                       torch.ones(2,3),2*torch.ones(2,3),5.)
        torch.testing.assert_close(actual,torch.tensor([[[7.,16.,30.],[100.,200.,300.]]],dtype=torch.float64))
        torch.testing.assert_close(acceleration,torch.tensor([[[10.,20.,30.],[100.,200.,300.]]],dtype=torch.float64))

    def test_world_rotation_error_and_invalid_cap(self):
        angle=torch.tensor(.2,dtype=torch.float64);c,s=angle.cos(),angle.sin()
        R=torch.tensor([[c,-s,0.],[s,c,0.],[0.,0.,1.]],dtype=torch.float64)[None]
        target=torch.eye(3,dtype=torch.float64)[None];zero=torch.zeros(1,3,dtype=torch.float64)
        actual=angular_tracking_target(R,target,zero,zero,zero,2.,1.,1.)
        torch.testing.assert_close(actual,torch.tensor([[0.,0.,-.4]],dtype=torch.float64))
        for cap in (0.,-1.,float('nan'),float('inf')):
            with self.assertRaises(ValueError):angular_tracking_target(R,target,zero,zero,zero,2.,1.,cap)

if __name__=='__main__':unittest.main()
