import unittest
import numpy as np
from fabrics_sim.reference.manager import infer_corner_contact_plan


class CornerContactTests(unittest.TestCase):
    def test_fixed_ground_point_supports_a_rolling_foot(self):
        points=np.zeros((10,2,2,3))
        points[:,1,:,2]=.1
        points[:,0,1,0]=np.arange(10)*.02
        points[:,0,1,2]=.05
        mode,ambiguous,speed=infer_corner_contact_plan(points,50)
        np.testing.assert_array_equal(mode,1)
        self.assertFalse(ambiguous.any())
        np.testing.assert_array_equal(speed[:,0],0)

    def test_height_and_speed_must_come_from_same_corner(self):
        points=np.zeros((10,2,2,3))
        points[:,:,1,2]=.1
        points[:,:,0,0]=np.arange(10)[:,None]*.02
        _,ambiguous,speed=infer_corner_contact_plan(points,50)
        self.assertTrue(ambiguous.all())
        self.assertTrue((speed>.2).all())


if __name__=='__main__':unittest.main()
