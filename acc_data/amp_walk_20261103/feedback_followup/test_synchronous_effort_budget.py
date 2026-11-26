import unittest
from synchronous_effort_budget import effort_fraction
class EffortBudgetTests(unittest.TestCase):
 def call(self,time,**kw):
  args=dict(start=4.2,fraction=1.,control_dt=.002,physics_dt=.002,direct_wbc=True,tracking_servo=True);args.update(kw);return effort_fraction(time,**args)
 def test_unchanged_prefix_and_physical_ceiling(self):
  self.assertEqual(self.call(4.19),.8);self.assertEqual(self.call(4.2),.8);self.assertEqual(self.call(4.3),1.)
  for i in range(1001):self.assertTrue(.8<=self.call(4.1+i*.0002)<=1.)
 def test_rejects_slow_or_unsynchronized_controller(self):
  for kw in [dict(control_dt=.004),dict(direct_wbc=False),dict(tracking_servo=False)]:
   with self.assertRaises(ValueError):self.call(4.3,**kw)
 def test_rejects_motor_limit_increase(self):
  for value in [1.01,.79,float('nan')]:
   with self.assertRaises(ValueError):self.call(4.3,fraction=value)
 def test_smooth_start_and_end(self):
  for t in [4.2,4.26]:
   derivative=(self.call(t+1e-5)-self.call(t-1e-5))/2e-5
   self.assertLess(abs(derivative),1e-5)
if __name__=='__main__':unittest.main()
