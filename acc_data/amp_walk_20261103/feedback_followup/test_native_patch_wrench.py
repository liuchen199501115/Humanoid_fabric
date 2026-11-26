import unittest
import numpy as np
from native_patch_wrench import native_patch_wrench
class NativeWrenchTests(unittest.TestCase):
 def feasible(self,points,w):
  E,G,b=native_patch_wrench(points);return np.max(abs(E@w),initial=0)<1e-8 and np.max(G@w-b)<1e-8
 def test_point_has_no_free_moment(self):
  point=np.array([-.1,-.03,-.04]);f=np.array([10.,0.,100.]);w=np.r_[f,np.cross(point,f)]
  self.assertTrue(self.feasible([point],w));w[5]+=.1;self.assertFalse(self.feasible([point],w))
 def test_corner_origin_does_not_extend_support(self):
  points=np.array([[-.1,-.03,-.04],[-.1,.03,-.04],[.1,-.03,-.04],[.1,.03,-.04]])
  f=np.array([0.,0.,100.])
  self.assertTrue(self.feasible(points,np.r_[f,np.cross([0.,0.,-.04],f)]))
  self.assertFalse(self.feasible(points,np.r_[f,np.cross([-.22,-.03,-.04],f)]))
 def test_triangle_does_not_become_rectangle(self):
  points=np.array([[0.,0.,0.],[.1,0.,0.],[0.,.1,0.]])
  self.assertFalse(self.feasible(points,[0.,0.,100.,9.,-9.,0.]))
 def test_line_yaw_and_friction_are_coupled(self):
  points=np.array([[0.,-.03,0.],[0.,.03,0.]])
  self.assertTrue(self.feasible(points,[0.,0.,100.,0.,0.,1.]))
  self.assertFalse(self.feasible(points,[50.,0.,100.,0.,0.,1.]))
 def test_every_physical_ray_is_admitted(self):
  points=np.array([[-.08,-.03,-.04],[.12,-.03,-.04],[.12,.03,-.04]])
  for p in points:
   for f in [[50.,0.,100.],[-50.,0.,100.],[0.,50.,100.],[0.,-50.,100.]]:
    self.assertTrue(self.feasible(points,np.r_[f,np.cross(p,f)]))
if __name__=='__main__':unittest.main()
