import unittest
import numpy as np
from reference_contact_schedule import delayed_touchdowns
class ContactScheduleTests(unittest.TestCase):
 def test_only_future_touchdown_is_delayed(self):
  modes=np.array([2,2,3,3,3,1,1]);h=np.zeros((7,2));h[2:4,0]=.01
  changed,events=delayed_touchdowns(modes,h,2)
  np.testing.assert_array_equal(changed,[2,2,2,2,3,1,1]);self.assertEqual(events,[dict(side=0,source_onset=2,predicted_onset=4)])
  np.testing.assert_array_equal(modes,[2,2,3,3,3,1,1])
 def test_retains_old_support_until_new_foot_arrives(self):
  modes=np.array([1,1,2,2,2]);h=np.zeros((5,2));h[2:4,1]=.01
  changed,_=delayed_touchdowns(modes,h,2)
  np.testing.assert_array_equal(changed,[1,1,1,1,2])
 def test_does_not_change_past_contact_schedule(self):
  modes=np.array([2,3,3,1]);h=np.ones((4,2))
  changed,events=delayed_touchdowns(modes,h,3)
  np.testing.assert_array_equal(changed,modes);self.assertEqual(events,[])
 def test_never_invents_a_touchdown_without_ground_arrival(self):
  with self.assertRaisesRegex(ValueError,'No ground arrival'):delayed_touchdowns(np.array([1,2,2]),np.ones((3,2)),1)
 def test_nonfinite_geometry_is_rejected(self):
  with self.assertRaises(ValueError):delayed_touchdowns(np.array([1,2]),np.full((2,2),np.nan),1)
if __name__=='__main__':unittest.main()
