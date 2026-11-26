import sys
from pathlib import Path
import unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'acc_data/amp_walk_20261103/feedback_followup'))
from tracking_tube import tracking_tube_constraints


class TrackingTubeRateTest(unittest.TestCase):
    def test_position_rate_preserves_joint_and_orientation_rows(self):
        q=torch.zeros(1,36,dtype=torch.float64);q[:,6]=1.
        ref=q.clone();q[:,:3]=torch.tensor([.1,-.08,-.05]);q[:,7:]=.12
        v=torch.full((1,35),.15,dtype=q.dtype);z=torch.zeros_like(v)
        a=tracking_tube_constraints(q,v,ref,z,z,bandwidth=10.)
        b=tracking_tube_constraints(q,v,ref,z,z,bandwidth=2.,joint_bandwidth=10.,orientation_bandwidth=10.)
        torch.testing.assert_close(a[0],b[0],atol=0,rtol=0)
        for x,y in zip(a[1:],b[1:]):
            torch.testing.assert_close(x[:,[0,2]],y[:,[0,2]],atol=0,rtol=0)
        self.assertNotEqual(a[1][0,1],b[1][0,1])
        self.assertEqual(b[2][0,2,2],b[2][0,2,1]+10*b[2][0,2,0])


if __name__=='__main__':unittest.main()
