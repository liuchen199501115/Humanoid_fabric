import unittest
import torch
from scipy.spatial.transform import Rotation
from tracking_margin import margin_weights

class TrackingMarginTests(unittest.TestCase):
    def setUp(self):
        self.q=torch.zeros(1,36,dtype=torch.float64);self.q[:,6]=1
        self.v=torch.zeros(1,35,dtype=torch.float64)
    def test_exact_tracking_has_unit_weights(self):
        w,r=margin_weights(self.q,self.v,self.q,self.v)
        torch.testing.assert_close(w,torch.ones_like(w));torch.testing.assert_close(r,torch.zeros_like(r))
    def test_velocity_and_frame_consistent_rotation(self):
        q=self.q.clone();ref=self.q.clone()
        ref[:,3:7]=torch.tensor(Rotation.from_euler('xyz',[.3,-.2,1.]).as_quat())
        q[:,3:7]=torch.tensor((Rotation.from_rotvec([.1,0,0])*Rotation.from_quat(ref[0,3:7])).as_quat())
        v=self.v.clone();v[:,3]=1
        w,r=margin_weights(q,v,ref,self.v,reaction_time=.1)
        self.assertAlmostEqual(r[0,1].item(),.2/.35,places=10)
        flipped=ref.clone();flipped[:,3:7]*=-1
        torch.testing.assert_close(margin_weights(q,v,flipped,self.v)[0],margin_weights(q,v,ref,self.v)[0])
    def test_monotonic_bounded_and_invalid_parameters(self):
        weights=[]
        for error in [0,.05,.1,.15,.2,.3]:
            q=self.q.clone();q[:,7:]=error
            w,_=margin_weights(q,self.v,self.q,self.v,maximum=16)
            weights.append(w[0,2].item())
        self.assertEqual(weights,sorted(weights));self.assertLessEqual(max(weights),16)
        for kwargs in [dict(reaction_time=-1),dict(maximum=0),dict(maximum=float('nan'))]:
            with self.assertRaises(ValueError):margin_weights(self.q,self.v,self.q,self.v,**kwargs)

if __name__=='__main__':unittest.main()
