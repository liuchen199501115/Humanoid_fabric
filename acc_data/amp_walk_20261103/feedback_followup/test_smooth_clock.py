import unittest
import numpy as np
from smooth_clock import smooth_clock

class ThreeStageClockTests(unittest.TestCase):
 args=(.9,.35,1.2,3.35,.2,.8,3.6,.2,1.2)
 def test_moving_rate_is_restored_without_phase_jump(self):
  np.testing.assert_allclose(smooth_clock(4.,*self.args),[4.485,1.2,0.],atol=1e-12)
  np.testing.assert_allclose(smooth_clock(4.,*self.args[:6]),[4.365,.8,0.],atol=1e-12)
  for t in [.9,1.25,3.35,3.55,3.6,3.8]:
   left=np.array(smooth_clock(t-1e-7,*self.args));right=np.array(smooth_clock(t+1e-7,*self.args))
   np.testing.assert_allclose(left,right,atol=3e-6,rtol=0)
 def test_rate_and_acceleration_match_the_phase_curve(self):
  h=1e-6
  for t in np.linspace(.01,12.,500):
   c=smooth_clock(float(t),*self.args);l=smooth_clock(float(t-h),*self.args);r=smooth_clock(float(t+h),*self.args)
   self.assertGreaterEqual(c[1],.8-1e-12)
   self.assertAlmostEqual((r[0]-l[0])/(2*h),c[1],places=6)
   self.assertAlmostEqual((r[1]-l[1])/(2*h),c[2],places=6)
 def test_overlapping_transitions_are_rejected(self):
  with self.assertRaises(ValueError):smooth_clock(4.,.9,.35,1.2,3.35,.2,.8,3.4,.2,1.2)
  with self.assertRaises(ValueError):smooth_clock(4.,.9,.35,1.2,third_speed=1.2)

if __name__=='__main__':unittest.main()
