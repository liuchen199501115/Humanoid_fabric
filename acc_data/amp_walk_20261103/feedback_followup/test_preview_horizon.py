import unittest,numpy as np
from preview_horizon import remaining_intervals
from test_preview_warm_start import Ref
from preview_warm_start import shift_prediction
from preview_manifold import integrate,difference
class HorizonTests(unittest.TestCase):
 def setUp(self):self.intervals=np.r_[.01,.01,np.full(22,.025)]
 def test_interior_prediction_grid_is_unchanged(self):np.testing.assert_array_equal(remaining_intervals(850.,240.,1999,self.intervals),self.intervals)
 def test_final_interval_reaches_source_end_once(self):
  intervals=remaining_intervals(1980.,240.,1999,self.intervals);self.assertTrue(np.all(intervals>0));self.assertAlmostEqual(1980.+240.*intervals.sum(),1999.);self.assertLess(intervals[-1],.025)
 def test_short_final_segment_does_not_add_settling(self):
  intervals=remaining_intervals(1998.7,240.,1999,self.intervals);self.assertEqual(len(intervals),1);self.assertAlmostEqual(intervals[0],.3/240.)
 def test_endpoint_has_no_fictitious_future(self):
  with self.assertRaises(ValueError):remaining_intervals(1999.,240.,1999,self.intervals)
 def test_moving_endpoint_velocity_is_preserved(self):
  states=np.zeros((3,70));states[:,0]=[0.,.025,.05];states[:,35]=1.
  previous=dict(frames=np.array([1990.,1995.,2000.]),states=states,controls=np.zeros((2,76)))
  intervals=remaining_intervals(1998.,200.,1999,self.intervals)
  result=shift_prediction(previous,Ref(),1998.,200.,intervals,len(intervals),integrate,difference)
  self.assertAlmostEqual(result['states'][-1,0],.045);self.assertEqual(result['states'][-1,35],1.)
if __name__=='__main__':unittest.main()
