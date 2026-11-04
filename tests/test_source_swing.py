import unittest
import numpy as np
from scipy.interpolate import PchipInterpolator

from fabrics_sim.reference.source_swing import source_swing_speed_bound, minimum_source_swing_duration
from fabrics_sim.reference.phase_clock import SwingClock
from fabrics_sim.reference.bridge import ReferenceBridge


class SourceSwingTests(unittest.TestCase):
    def test_shared_nonzero_terminal_rate_bounds_entire_short_bridge_path(self):
        from fabrics_sim.reference.phase_clock import monotone_swing_clock
        frames=np.arange(31)
        points=np.c_[frames*.03,np.zeros(31),.1*np.sin(frames*np.pi/30)]
        pp=PchipInterpolator(frames,points,axis=0)
        fps=30.;rate=.6;duration=1.;bridge_time=.2
        position=points[0]+[.01,0.,-.02];velocity=np.array([.1,0.,0.])
        clock=monotone_swing_clock(0.,rate,1.,duration,1.)
        self.assertEqual(clock.end_rate,1.)
        bound=source_swing_speed_bound(pp.c,fps,0.,rate,30.,duration,position,velocity,
                                       bridge_duration=bridge_time,end_rate=1.)
        bridge=ReferenceBridge(position-pp(0),velocity-pp(0,1)*fps*rate,
                               -pp(0,2)*(fps*rate)**2,bridge_time)
        speeds=[]
        for t in np.linspace(0,duration,2001):
            phase,phase_rate,_=clock.sample(t)
            speeds.append(np.linalg.norm(pp(phase*fps,1)*phase_rate*fps+bridge.sample(t)[1]))
        self.assertGreaterEqual(bound,max(speeds)-1e-8)
        self.assertLess(bound,1.05*max(speeds))
        with self.assertRaises(ValueError):monotone_swing_clock(0.,rate,1.,duration,-1.)

    def test_start_within_roundoff_of_source_knot_has_finite_speed_bound(self):
        frames=np.arange(21);points=np.c_[frames/20.,np.zeros((21,2))]
        pp=PchipInterpolator(frames,points,axis=0)
        start=np.nextafter(1.,0.)
        bound=source_swing_speed_bound(pp.c,20.,start,.5,20.,1.,pp(start),np.array([.5,0.,0.]))
        clock=SwingClock(start/20.,.5,1.,1.)
        maximum=max(abs(clock.sample(t)[1]) for t in np.linspace(0,1,1001))
        self.assertTrue(np.isfinite(bound))
        self.assertGreaterEqual(bound,maximum-1e-8)
        self.assertLess(bound,1.02*maximum)

    def test_straight_path_bounds_known_quintic_peak(self):
        points=np.c_[np.linspace(0,1,21),np.zeros((21,2))]
        c=PchipInterpolator(np.arange(21),points,axis=0).c
        bound=source_swing_speed_bound(c,20.,0.,0.,20.,1.,np.zeros(3),np.zeros(3))
        self.assertGreaterEqual(bound,1.875-1e-10)
        self.assertLess(bound,1.89)
        duration=minimum_source_swing_duration(c,20.,0.,0.,20.,np.zeros(3),np.zeros(3),0.,.5,1.)
        self.assertGreaterEqual(duration,1.875)
        self.assertLessEqual(source_swing_speed_bound(c,20.,0.,0.,20.,duration,np.zeros(3),np.zeros(3)),1.)

    def test_curved_source_bridge_and_ground_correction_stay_inside_bound(self):
        frames=np.arange(21);points=np.c_[frames*.02,.025*np.sin(frames*.6),.06*np.sin(frames*np.pi/20)]
        pp=PchipInterpolator(frames,points,axis=0);duration=.8;fps=30.;rate=.5
        position=points[0]+[.02,0,-.01];velocity=np.array([.2,.1,0.]);height=.02
        for bridge_time in (duration,.2,.35):
            bound=source_swing_speed_bound(pp.c,fps,0.,rate,20.,duration,position,velocity,height,bridge_time)
            clock=SwingClock(0.,rate,20/fps,duration)
            bridge=ReferenceBridge(position-pp(0),velocity-pp(0,1)*fps*rate,-pp(0,2)*(fps*rate)**2,bridge_time)
            speeds=[]
            for t in sorted(set(np.linspace(0,duration,1001))|{bridge_time-1e-9,bridge_time,bridge_time+1e-9}):
                if t>duration:continue
                phase,phase_rate,_=clock.sample(t);s=t/duration
                v=pp(phase*fps,1)*phase_rate*fps+bridge.sample(t)[1]
                v[2]-=30*s*s*(1-s)**2/duration*height
                speeds.append(np.linalg.norm(v))
            self.assertGreaterEqual(bound,max(speeds)-1e-8)
            self.assertLess(bound,1.05*max(speeds))


if __name__=='__main__':unittest.main()
