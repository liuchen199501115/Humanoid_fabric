import unittest
import numpy as np
from early_resume_clock import EarlyResumeLandingClock
from landing_pause_clock import LandingPauseClock

class EarlyResumeClockTests(unittest.TestCase):
    def test_gate_requires_measured_landings(self):
        new=EarlyResumeLandingClock(4.3,4.5,.2,11);old=LandingPauseClock(4.3,4.5,.2,11)
        for time in np.linspace(0,6,100):
            new.update(time,10)
            self.assertEqual(new.sample(time),old.sample(time))
        self.assertIsNone(new.release)
    def test_c2_monotone_bridge_for_early_and_late_landings(self):
        for release in [4.3,4.31,4.4,4.5,4.6,4.69,4.69999,4.7,4.9]:
            clock=EarlyResumeLandingClock(4.3,4.5,.2,11)
            before=clock.sample(release);clock.update(release,11)
            np.testing.assert_allclose(clock.sample(release),before,atol=1e-10)
            end=release+clock.bridge_duration
            values=np.array([clock.sample(t) for t in np.linspace(release,end+.1,500)])
            self.assertGreaterEqual(values[:,1].min(),-1e-9)
            self.assertLessEqual(values[:,2].max(),1.5/clock.up+1e-9)
            self.assertLessEqual(values[:,1].max(),1+1e-9)
            self.assertGreaterEqual(np.diff(values[:,0]).min(),-1e-9)
            np.testing.assert_allclose(clock.sample(end)[1:],[1.,0.],atol=1e-8)
            if clock.bridge_duration>1e-4:
                for t in np.linspace(release+.05*clock.bridge_duration,end-.05*clock.bridge_duration,5):
                    h=1e-6;lo=clock.sample(t-h);hi=clock.sample(t+h);now=clock.sample(t)
                    self.assertAlmostEqual((hi[0]-lo[0])/(2*h),now[1],places=7)
                    np.testing.assert_allclose((hi[1]-lo[1])/(2*h),now[2],rtol=1e-6,atol=1e-6)
    def test_release_is_latched_and_precedes_fixed_stop(self):
        clock=EarlyResumeLandingClock(4.3,4.5,.2,11);clock.update(4.5,11)
        saved=clock.sample(4.7);clock.update(4.6,10)
        self.assertEqual(clock.release,4.5);self.assertEqual(clock.sample(4.7),saved)
        self.assertGreater(saved[0],4.5)

if __name__=='__main__':unittest.main()
