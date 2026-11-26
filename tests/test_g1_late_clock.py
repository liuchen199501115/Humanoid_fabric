import unittest
import numpy as np
from g1_late_clock import late_speed_transition


class LateClockTest(unittest.TestCase):
    def sample(self,t):
        return late_speed_transition((1.2*t-.2,1.2,0.),t,start=3.5,ramp=.2,previous_speed=1.2,speed=.8)

    def test_derivatives_and_strict_progress(self):
        for t in np.linspace(3.49,3.71,61):
            h=1e-6;state=self.sample(t);lo=self.sample(t-h);hi=self.sample(t+h)
            self.assertAlmostEqual((hi[0]-lo[0])/(2*h),state[1],places=7)
            self.assertAlmostEqual((hi[1]-lo[1])/(2*h),state[2],places=6)
            self.assertGreaterEqual(state[1],.8-1e-14)
            self.assertLessEqual(state[1],1.2+1e-14)

    def test_prefix_and_end(self):
        for t in [-.002,0.,2.,3.5]:self.assertEqual(self.sample(t),(1.2*t-.2,1.2,0.))
        self.assertAlmostEqual(self.sample(3.7)[0],1.2*3.7-.2-.04)
        self.assertAlmostEqual(self.sample(3.7)[1],.8)
        self.assertEqual(self.sample(3.7)[2],0.)

    def test_reject_invalid_parameters(self):
        for value in [0.,-1.,float('nan')]:
            with self.assertRaises(ValueError):
                late_speed_transition((1.,1.2,0.),2.,start=1.,ramp=.2,previous_speed=1.2,speed=value)

    def test_sequential_transition_preserves_prefix_and_exact_derivatives(self):
        def sample(t):
            return late_speed_transition(self.sample(t),t,start=4.42,ramp=.12,previous_speed=.8,speed=1.1)
        for t in [0.,3.5,3.7,4.344,4.42]:
            self.assertEqual(sample(t),self.sample(t))
        for t in np.linspace(4.41,4.55,37):
            h=1e-6;state=sample(t);lo=sample(t-h);hi=sample(t+h)
            self.assertAlmostEqual((hi[0]-lo[0])/(2*h),state[1],places=7)
            self.assertAlmostEqual((hi[1]-lo[1])/(2*h),state[2],places=6)
            self.assertGreaterEqual(state[1],.8-1e-14)
            self.assertLessEqual(state[1],1.1+1e-14)


if __name__=='__main__':unittest.main()
