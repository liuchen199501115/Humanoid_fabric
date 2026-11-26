import unittest,numpy as np
from preview_integration import advance,position_factor
from preview_manifold import integrate,difference
from preview_warm_start import shift_prediction
from test_preview_warm_start import Ref
class IntegrationTests(unittest.TestCase):
 def test_two_real_control_ticks_use_updated_velocity_for_position(self):
  q=np.r_[np.zeros(6),1.,np.zeros(29)];v=np.zeros(35);a=np.zeros(35);a[6]=2350.
  actual,velocity=advance(q,v,a,.004)
  self.assertAlmostEqual(actual[7],.0282);self.assertAlmostEqual(velocity[6],9.4)
  self.assertAlmostEqual(position_factor(.004)*2350.,.0282)
 def test_partial_final_tick_remains_consistent(self):
  q=np.r_[np.zeros(6),1.,np.zeros(29)];v=np.zeros(35);a=np.zeros(35);a[6]=2350.
  actual,velocity=advance(q,v,a,.003)
  self.assertAlmostEqual(actual[7],.01645);self.assertAlmostEqual(velocity[6],7.05)
 def test_warm_shift_uses_same_simulator_integration(self):
  times=np.array([0.,.01,.034]);a=np.zeros(35);a[6]=235.;states=np.zeros((3,70));controls=np.zeros((2,76));controls[:,:35]=a
  states[:,6]=[0.,.0141,.14382];states[:,41]=235.*times
  previous=dict(frames=times*200.,states=states,controls=controls)
  result=shift_prediction(previous,Ref(),.8,200.,np.array([.01,.024]),2,integrate,difference,integration='semiimplicit')
  for k,t in enumerate([.004,.014,.038]):
   self.assertAlmostEqual(result['states'][k,6],235.*.5*t*(t+.002),places=10)
   self.assertAlmostEqual(result['states'][k,41],235.*t,places=10)
if __name__=='__main__':unittest.main()
