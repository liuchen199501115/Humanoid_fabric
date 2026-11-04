import unittest
import numpy as np
from fabrics_sim.reference.phase_clock import SwingClock
from fabrics_sim.reference.manager import ReferenceManager


class PhaseClockTests(unittest.TestCase):
    def test_aligned_takeoff_has_no_one_tick_clock_stop(self):
        manager=ReferenceManager([3]*114+[1]*15+[3]*30,30,
            ramp_duration=.25,align_takeoff_frame=True)
        manager.tau=113.9/30;manager.rate=.62;manager.confirmed=30
        manager.transfer_load_latched=True;manager.pending_takeoff=114
        frames=[manager.tau*30]
        for _ in range(12):
            clock=manager.step([330.,5.],[True,True],.002,
                               sole_clearance=[0,0],balance_ready=[True,True])
            frames.append(clock.frame)
            self.assertAlmostEqual((clock.next_frame-clock.frame)/30,
                clock.rate*.002+.5*4.*.002**2,places=10)
        self.assertEqual(clock.mode,1)
        accelerations=np.diff(frames,n=2)/30/.002**2
        np.testing.assert_allclose(accelerations,4.,atol=1e-7)

    def test_continuous_settle_keeps_rate_but_cannot_cross_unconfirmed_takeoff(self):
        manager=ReferenceManager([3]*129+[3]*8+[2]*18+[3]*20,30,
            ramp_duration=.25,continuous_source_clock=True,settle_duration=1.)
        manager.tau=129/30;manager.rate=1.;manager.mode=3
        manager.settling=True;manager.landing_support=1;manager.search_from=129
        first=manager.step([10,300],[True,True],.002,sole_clearance=[0,0])
        self.assertAlmostEqual(first.rate,1.)
        self.assertGreater(first.frame,129.)
        for _ in range(400):
            clock=manager.step([10,300],[True,True],.002,sole_clearance=[0,0])
            self.assertLess(clock.frame,137.)
        self.assertEqual(clock.mode,3)
        self.assertLess(clock.rate,1e-6)

    def test_stretched_swing_can_resume_nominal_rate_when_stopping_room_allows(self):
        manager = ReferenceManager([3]*114+[1]*15+[3]*8+[2]*18+[3]*20,30,
                                   ramp_duration=.25,continuous_source_clock=True)
        manager.tau=114/30;manager.rate=.536;manager.mode=1
        manager.landing_frame=129;manager.last_dt=.002
        manager.synchronize_swing(.8)
        self.assertAlmostEqual(manager.swing_clock.end_rate,1.)
        values=np.array([manager.swing_clock.sample(t) for t in np.linspace(0,.8,1001)])
        self.assertTrue((values[:,1]>=0).all())
        self.assertAlmostEqual(values[-1,0],129/30)
        # The same source gap with a weaker braking limit must cap the rate.
        manager.max_rate_change=1.
        manager.synchronize_swing(.8)
        self.assertLess(manager.swing_clock.end_rate,.66)

    def test_short_double_support_can_stop_before_next_unconfirmed_takeoff(self):
        manager=ReferenceManager([3]*10+[1]*20+[3]*3+[2]*20+[3]*20,30,
            continuous_source_clock=True,early_touchdown=True,
            settle_on_incoming=True,settle_duration=.06)
        for _ in range(100):
            clock=manager.step([300,20],[True,True],.02,sole_clearance=[0,0],
                               sole_max_height=[0,0],balance_ready=[True,True])
            if clock.mode==1:break
        self.assertEqual(clock.mode,1)
        manager.synchronize_swing(1.)
        self.assertGreater(manager.swing_clock.end_rate,0.)
        for _ in range(40):
            manager.step([320,0],[True,False],.02,sole_clearance=[0,.05],
                         sole_max_height=[0,.05],balance_ready=[True,True])
        previous=manager.tau*30
        for _ in range(100):
            clock=manager.step([20,300],[True,True],.02,sole_clearance=[0,0],
                               sole_max_height=[0,0],balance_ready=[False,False])
            self.assertGreaterEqual(clock.frame+1e-9,previous)
            self.assertLessEqual(clock.frame,33.+1e-6)
            previous=clock.frame
        self.assertEqual(clock.mode,3)
        self.assertLess(clock.rate,1e-6)

    def test_continuous_terminal_rate_preserves_derivatives_and_progress(self):
        clock=SwingClock(0.,.6,.6,1.,end_rate=.6)
        for t in np.linspace(0,1,21):
            np.testing.assert_allclose(clock.sample(t),[.6*t,.6,0.],atol=1e-10)

    def test_continuous_manager_resumes_late_touchdown_without_catching_up(self):
        manager=ReferenceManager([3]*10+[1]*20+[3]*20+[2]*20+[3]*10,30,
                                 continuous_source_clock=True)
        for _ in range(100):
            a=manager.step([300,20],[True,True],.02,sole_clearance=[0,0],balance_ready=[True,True])
            if a.mode==1:break
        manager.synchronize_swing(1.)
        self.assertGreater(manager.swing_clock.end_rate,0.)
        for _ in range(60):
            a=manager.step([320,0],[True,False],.02,sole_clearance=[0,.05],balance_ready=[True,True])
        self.assertAlmostEqual(a.frame,30.)
        self.assertEqual(a.rate,0.)
        for _ in range(3):
            a=manager.step([200,130],[True,True],.02,sole_clearance=[0,0],balance_ready=[True,True])
        self.assertEqual(a.mode,3)
        self.assertAlmostEqual(a.frame,30.)
        self.assertEqual(a.rate,0.)

    def test_confirmed_landing_retains_a_continuous_source_rate(self):
        manager=ReferenceManager([3]*10+[1]*20+[3]*20+[2]*20+[3]*10,30,
                                 continuous_source_clock=True,early_touchdown=True)
        for _ in range(100):
            a=manager.step([300,20],[True,True],.02,sole_clearance=[0,0],sole_max_height=[0,0],balance_ready=[True,True])
            if a.mode==1:break
        manager.synchronize_swing(1.)
        for _ in range(30):
            manager.step([320,0],[True,False],.02,sole_clearance=[0,.05],sole_max_height=[0,.05],balance_ready=[True,True])
        for _ in range(20):
            a=manager.step([150,180],[True,True],.02,sole_clearance=[0,0],sole_max_height=[0,0],balance_ready=[True,True])
        self.assertEqual(a.mode,3)
        self.assertIsNone(manager.swing_clock)
        self.assertGreater(a.rate,0.)
        self.assertGreater(a.next_frame,a.frame)

    def test_boundary_rates_and_monotonicity(self):
        clock=SwingClock(3.,.7,4.,.8)
        np.testing.assert_allclose(clock.sample(0),[3.,.7,0.],atol=1e-10)
        np.testing.assert_allclose(clock.sample(.8),[4.,0.,0.],atol=1e-10)
        positions=[clock.sample(t)[0] for t in np.linspace(0,1,201)]
        self.assertTrue((np.diff(positions)>=-1e-10).all())

    def test_derivative_and_reversal_rejection(self):
        clock=SwingClock(3.,.7,4.,.8);eps=1e-5;t=.3
        a=clock.sample(t-eps);b=clock.sample(t+eps);s=clock.sample(t)
        np.testing.assert_allclose((b[0]-a[0])/(2*eps),s[1],atol=1e-8)
        np.testing.assert_allclose((b[1]-a[1])/(2*eps),s[2],atol=1e-7)
        with self.assertRaises(ValueError):SwingClock(0,10,1,2)

    def test_manager_holds_at_endpoint_until_physical_touchdown(self):
        manager=ReferenceManager([3]*10+[1]*20+[2]*20+[3]*10,30)
        for _ in range(100):
            clock=manager.step([300,20],[True,True],.02,sole_clearance=[0,0],balance_ready=[True,True])
            if clock.mode==1:break
        self.assertEqual(clock.mode,1)
        manager.synchronize_swing(1.)
        for _ in range(60):
            clock=manager.step([320,0],[True,False],.02,sole_clearance=[0,.05],balance_ready=[True,True])
        self.assertAlmostEqual(clock.frame,30.)
        self.assertEqual(clock.mode,1)
        self.assertEqual(clock.rate,0.)
        for _ in range(3):
            clock=manager.step([300,30],[True,True],.02,sole_clearance=[0,0],balance_ready=[True,True])
        self.assertEqual(clock.mode,3)
        self.assertAlmostEqual(clock.frame,30.)


if __name__=='__main__':unittest.main()
