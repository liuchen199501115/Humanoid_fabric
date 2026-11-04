import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from fabrics_sim.reference.balance import support_balance_ready, source_transfer_reference


class BalanceGateTests(unittest.TestCase):



    def test_transfer_retains_source_forward_motion_in_rotated_heading(self):
        stance=np.r_[2.,3.,.035,Rotation.from_euler('z',np.pi/2).as_quat()]
        p,v,a=source_transfer_reference([2.1,3.3,.7],[.2,.5,0.],[.3,.4,0.],stance)
        np.testing.assert_allclose(p,[2.,3.3],atol=1e-12)
        np.testing.assert_allclose(v,[0.,.5],atol=1e-12)
        np.testing.assert_allclose(a,[0.,.4],atol=1e-12)




    def test_requires_slow_capture_point_inside_support_foot(self):
        feet=np.array([[0,0,.035,0,0,0,1],[0,.2,.035,0,0,0,1]],dtype=float)
        np.testing.assert_array_equal(support_balance_ready([.02,0,.7],[0,0,0],feet),[True,False])
        self.assertFalse(support_balance_ready([.02,0,.7],[.2,0,0],feet).any())
        self.assertFalse(support_balance_ready([.02,.1,.7],[0,0,0],feet).any())

    def test_uses_rotated_support_polygon(self):
        feet=np.array([[0,0,.035,0,0,0,1],[0,.2,.035,0,0,0,1]],dtype=float)
        base=support_balance_ready([.08,0,.7],[.01,0,0],feet)
        r=Rotation.from_euler('z',np.pi/2)
        feet[:,:3]=r.apply(feet[:,:3]);feet[:,3:]=r.as_quat()
        rotated=support_balance_ready(r.apply([.08,0,.7]),r.apply([.01,0,0]),feet)
        np.testing.assert_array_equal(rotated,base)


if __name__=='__main__':unittest.main()
