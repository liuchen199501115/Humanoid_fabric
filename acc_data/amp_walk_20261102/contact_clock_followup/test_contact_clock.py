import copy
import unittest
import numpy as np
from contact_clock import ContactTrackingClock

class ClockChecks(unittest.TestCase):
    def test_monotone_bounded_under_contact_switches(self):
        clock=ContactTrackingClock(start=0)
        previous=clock.frame
        for index in range(1000):
            frame,rate,_=clock.advance(.002,index*.002,1 if index<500 else 2,[200.,0.])
            self.assertGreater(frame,previous)
            self.assertGreaterEqual(rate,200*.45-1e-10)
            self.assertLessEqual(rate,200+1e-10)
            previous=frame

    def test_exact_integration_and_derivatives(self):
        one=ContactTrackingClock(start=0);many=ContactTrackingClock(start=0)
        one.advance(.012,.012,2,[200.,0.])
        for index in range(6):many.advance(.002,(index+1)*.002,2,[200.,0.])
        np.testing.assert_allclose([one.frame,one.rate,one.acceleration],
                                   [many.frame,many.rate,many.acceleration],atol=1e-12)
        point=copy.deepcopy(one);step=1e-5;start=point.frame
        middle,velocity,acceleration=point.advance(step,.013,2,[200.,0.])
        end,_,_=point.advance(step,.014,2,[200.,0.])
        self.assertAlmostEqual((end-start)/(2*step),velocity,delta=1e-4)
        self.assertAlmostEqual((end-2*middle+start)/step**2,acceleration,delta=.01)

    def test_target_switch_preserves_rate_and_acceleration(self):
        clock=ContactTrackingClock(start=0)
        clock.advance(.1,.1,2,[200.,0.])
        before=(clock.frame,clock.rate,clock.acceleration)
        clock.advance(0.,.1,1,[200.,0.])
        np.testing.assert_allclose(before,[clock.frame,clock.rate,clock.acceleration],atol=1e-12)

    def test_optional_landing_gate_and_start(self):
        clock=ContactTrackingClock(start=1.,include_landing=True)
        clock.advance(.1,.5,3,[200.,0.]);self.assertEqual(clock.rate,1.)
        clock.advance(.1,1.,3,[200.,0.]);self.assertLess(clock.rate,1.)
        nominal=ContactTrackingClock(start=0,include_landing=False)
        nominal.advance(.1,.1,3,[200.,0.]);self.assertEqual(nominal.rate,1.)

if __name__=='__main__':unittest.main()
