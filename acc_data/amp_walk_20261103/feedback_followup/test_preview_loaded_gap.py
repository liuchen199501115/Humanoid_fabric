import unittest,numpy as np
from test_full_body_preview_audit import CartesianModel,StaticReference
from full_body_preview_tight_contact import FullBodyPreview
class GravityModel(CartesianModel):
 def evaluate(self,*args,**kwargs):
  result=super().evaluate(*args,**kwargs);result['bias'][2]=25.;return result
class LoadedGapTests(unittest.TestCase):
 def audit(self,height):
  planner=FullBodyPreview(GravityModel(),StaticReference(),None,None,[(np.zeros((1,3)),np.zeros(1))]*2,None,None,None,horizon=2)
  plan=dict(frames=np.arange(3),states=np.zeros((3,70)),controls=np.zeros((2,76)),impulses=np.zeros((2,12)),impact_maps={});plan['states'][:,2]=height;plan['controls'][:,37]=25.
  return planner.nonlinear_audit(plan,1.)
 def test_real_ground_support_is_admitted(self):self.assertTrue(self.audit(0.)['nonlinear_verified'])
 def test_loaded_point_above_ground_is_rejected(self):
  result=self.audit(.0007);self.assertFalse(result['nonlinear_verified']);self.assertAlmostEqual(result['nonlinear_support_gap'],.0007)
 def test_loaded_point_below_ground_is_rejected(self):
  result=self.audit(-.0007);self.assertFalse(result['nonlinear_verified']);self.assertAlmostEqual(result['nonlinear_support_gap'],.0007)
if __name__=='__main__':unittest.main()
