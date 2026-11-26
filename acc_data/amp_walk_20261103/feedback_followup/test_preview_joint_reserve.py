import unittest
import numpy as np
from preview_joint_reserve import joint_reserve
class JointReserveTests(unittest.TestCase):
 def test_valid_measured_state_near_limit_is_not_rejected_by_reserve(self):
  limits=np.array([[-.2618,.2618],[-1.,1.]])
  q=np.array([.2604,0.]);m=joint_reserve(q,limits,0.)
  self.assertTrue(np.all(q>=limits[:,0]+m));self.assertTrue(np.all(q<=limits[:,1]-m))
  self.assertGreater(q[0],limits[0,1]-.005);self.assertEqual(m[1],.005)
 def test_reserve_returns_without_expanding_physical_limits(self):
  limits=np.array([[-1.,1.]]);q=np.array([.999])
  previous=-1.
  for t in [0.,.002,.01,.03,.05,.1]:
   margin=float(joint_reserve(q,limits,t)[0]);self.assertGreaterEqual(margin,previous);self.assertGreaterEqual(margin,0.);self.assertLessEqual(margin,.005);previous=margin
  self.assertEqual(previous,.005)
 def test_outside_hardware_range_remains_outside(self):
  limits=np.array([[-1.,1.]]);q=np.array([1.001]);m=joint_reserve(q,limits,0.)
  self.assertGreater(q[0],limits[0,1]-m[0])
if __name__=='__main__':unittest.main()
