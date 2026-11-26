"""An elastic numerical seed must never be admitted as a control plan."""
import unittest
from full_body_preview import FullBodyPreview
class FakePreview(FullBodyPreview):
 def __init__(self,always_elastic=False):self.calls=0;self.always_elastic=always_elastic
 def solve(self,*args,**kwargs):
  self.calls+=1
  elastic=self.always_elastic or self.calls==1
  return dict(success=True,status='Solved',elastic_slacks={'future_ground':.01} if elastic else {})
 def nonlinear_audit(self,*args):return dict(nonlinear_verified=True)
class ElasticAdmissionTests(unittest.TestCase):
 def test_only_strict_solution_can_be_accepted(self):
  planner=FakePreview();result=planner.refine(None,None,0.,200.,{},iterations=3)
  self.assertTrue(result['success']);self.assertEqual(planner.calls,2)
  self.assertEqual(result['elastic_slacks'],{})
 def test_elastic_result_is_rejected_even_if_other_checks_pass(self):
  planner=FakePreview(True);result=planner.refine(None,None,0.,200.,{},iterations=3)
  self.assertFalse(result['success']);self.assertEqual(planner.calls,3)
if __name__=='__main__':unittest.main()
