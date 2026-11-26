import unittest
import numpy as np
from scipy.interpolate import make_interp_spline
from reference_joint_extrema import joint_extrema
class JointExtremaTests(unittest.TestCase):
 def test_detects_interior_minimum_and_endpoint_velocity(self):
  nodes=np.arange(11.);q=np.zeros((11,36));q[:,6]=1;q[:,7]=.01*(nodes-5)**2
  r=joint_extrema(make_interp_spline(nodes,q,k=3),2,8,240,np.tile([-1.,1.],(29,1)),np.full(29,20.))
  self.assertAlmostEqual(r['joint_minimum'][0],0,places=12)
  self.assertAlmostEqual(r['joint_maximum'][0],.09,places=12)
  self.assertAlmostEqual(r['minimum_continuous_joint_margin'],.91,places=12)
  self.assertAlmostEqual(r['maximum_continuous_joint_velocity_fraction'],.72,places=12)
 def test_rejects_invalid_interval(self):
  with self.assertRaises(ValueError):joint_extrema(None,3,2,240,np.zeros((29,2)),np.ones(29))
if __name__=='__main__':unittest.main()
