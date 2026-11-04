import unittest
import numpy as np
from scipy.interpolate import PchipInterpolator
from fabrics_sim.reference.source_com import SourceComReference,SourceComContinuation


class SourceComTests(unittest.TestCase):
    def test_replanning_preserves_the_existing_clock_and_momentum(self):
        from fabrics_sim.reference.phase_clock import monotone_swing_clock
        frames=np.arange(31)
        com=PchipInterpolator(frames,np.c_[.03*frames,.01*frames,np.ones(31)])
        momentum=PchipInterpolator(frames,np.c_[.01*frames,.03*frames,.02*frames])
        ref=SourceComReference(com,30.,0.,.5,30.,.75,momentum)
        clock=monotone_swing_clock(0.,.5,1.,1.2,.75)
        continuation=SourceComContinuation(ref,clock,.3)
        fractions=np.array([0.,.4,1.]);original=(.3+.9*fractions)/1.2
        actual,v=continuation(.9,fractions);expected,expected_v=ref(1.2,original)
        np.testing.assert_allclose(actual,expected,atol=1e-12)
        np.testing.assert_allclose(v,expected_v,atol=1e-12)
        np.testing.assert_allclose(continuation.momentum(.9,fractions),ref.momentum(1.2,original),atol=1e-12)
        self.assertIsNone(continuation(.901,fractions))
        with self.assertRaises(ValueError):SourceComContinuation(ref,clock,1.2)

    def test_angular_momentum_uses_actual_phase_rate(self):
        frames=np.arange(31)
        com=PchipInterpolator(frames,np.c_[frames,frames,np.ones(31)])
        H=PchipInterpolator(frames,np.tile([1.,2.,3.],(31,1)))
        ref=SourceComReference(com,30.,0.,.5,30.,.75,H)
        actual=ref.momentum(1.,np.array([0.,1.]))
        np.testing.assert_allclose(actual,[[.5,1.,1.5],[.75,1.5,2.25]],atol=1e-12)

    def test_world_positions_and_terminal_velocity_share_retimed_clock(self):
        frames=np.arange(31)
        pp=PchipInterpolator(frames,np.c_[1.+.03*frames,2.+.01*frames,np.full(31,.7)])
        ref=SourceComReference(pp,30.,0.,.5,30.,.75)
        eps=1e-5
        position,velocity=ref(1.,np.array([0.,1.-eps,1.]))
        np.testing.assert_allclose(position[[0,-1]],[[1.,2.],[1.9,2.3]],atol=1e-12)
        np.testing.assert_allclose((position[-1]-position[-2])/eps,velocity,atol=1e-7)
        np.testing.assert_allclose(velocity,[.675,.225],atol=1e-12)
        # In particular, use the retimed .75 terminal rate, not nominal 1.0.
        self.assertIsNone(ref(-1.,np.array([.5])))


if __name__=='__main__':unittest.main()
