import unittest
from types import SimpleNamespace
import numpy as np
from landing_gated_unload import anticipatory_unload_mode
class OverdueUnloadTests(unittest.TestCase):
 def tracker(self):
  return SimpleNamespace(complete=False,events=[dict(support=0,takeoff_frame=780,end_frame=802),dict(support=1,takeoff_frame=802,end_frame=884)],landings=[],airborne=True,tolerance=.04,loaded_seconds=.012)
 def call(self,tracker=None,frame=810,forces=(50,200),source=None,**kwargs):
  feet=np.zeros((2,3));return anticipatory_unload_mode(tracker or self.tracker(),frame,feet,feet.copy() if source is None else source,forces,200,3,**kwargs)
 def test_zero_double_support_requires_opt_in_and_actual_evidence(self):
  self.assertEqual(self.call(),(3,False));self.assertEqual(self.call(allow_overdue=True),(2,True))
 def test_no_transfer_without_load_hold_or_source_goal(self):
  self.assertEqual(self.call(forces=(200,99),allow_overdue=True),(3,False))
  tracker=self.tracker();tracker.loaded_seconds=.008;self.assertEqual(self.call(tracker=tracker,allow_overdue=True),(3,False))
  source=np.zeros((2,3));source[1,0]=.031;self.assertEqual(self.call(source=source,allow_overdue=True),(3,False))
 def test_no_transfer_without_verified_flight(self):
  tracker=self.tracker();tracker.airborne=False;self.assertEqual(self.call(tracker=tracker,allow_overdue=True),(3,False))
 def test_expired_or_too_early_swing_is_not_skipped(self):
  self.assertEqual(self.call(frame=884,allow_overdue=True),(3,False));self.assertEqual(self.call(frame=780,allow_overdue=True),(3,False))
if __name__=='__main__':unittest.main()
