import unittest
import numpy as np,torch
from scipy.interpolate import BSpline
from fabrics_sim.reference.interpolation import MotionInterpolator
from blended_spline import BlendedSplineReference

class MovingEndpointTests(unittest.TestCase):
 def make(self,length):
  q=torch.zeros((length,36),dtype=torch.float64);q[:,6]=1.;q[:,0]=torch.arange(length)*.01;q[:,7]=torch.arange(length)*.001
  return q,BlendedSplineReference(MotionInterpolator(q),q,stride=6,start_frame=20,ramp_frames=10)
 def test_uniform_motion_does_not_brake_at_short_final_interval(self):
  for length in (98,100,101):
   q,s=self.make(length)
   for frame in np.linspace(length-16,length-1,31):
    pose,v,a=s.kinematics(float(frame),200.)
    self.assertAlmostEqual(float(pose[0,0]),frame*.01,places=6)
    self.assertAlmostEqual(float(v[0,0]),2.,places=4)
    self.assertLess(abs(float(a[0,0])),.005)
   np.testing.assert_allclose(s.kinematics(length-1,200.)[0].numpy()[0],q[-1].numpy(),atol=1e-12)
 def test_retiming_chain_rule_is_retained_at_endpoint(self):
  q,s=self.make(100);pose,v,a=s.kinematics(99.,120.,-30.)
  self.assertAlmostEqual(float(v[0,0]),1.2,places=4)
  self.assertAlmostEqual(float(a[0,0]),-.3,places=3)
 def test_interior_coefficients_are_unchanged(self):
  q,s=self.make(2000);nodes=np.unique(np.r_[np.arange(0,len(q),6),len(q)-1]);knots=np.r_[np.full(4,nodes[0]),nodes[2:-2],np.full(4,nodes[-1])];old=BSpline(knots,q.numpy()[nodes],3)
  for frame in [100.,500.,900.,1500.,1950.]:
   for order in range(3):np.testing.assert_array_equal(s.smooth(frame,order),old(frame,order))

if __name__=='__main__':unittest.main()
