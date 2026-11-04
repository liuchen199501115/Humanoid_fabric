import unittest
import numpy as np
from fabrics_sim.reference.height import BrakedHeightProfile


class HeightBrakingTests(unittest.TestCase):
    def test_measured_downward_velocity_stops_without_large_height_undershoot(self):
        p = BrakedHeightProfile(.627, -.232, .606, .984, 2.)
        np.testing.assert_allclose(p.sample(0), [.627, -.232, 0.], atol=1e-12)
        np.testing.assert_allclose(p.sample(p.duration), [.606, 0., 0.], atol=1e-12)
        samples = np.array([p.sample(t) for t in np.linspace(0, p.duration, 1001)])
        self.assertGreaterEqual(samples[:, 0].min(), .606-1e-12)
        self.assertLessEqual(abs(samples[:, 2]).max(), 2.+1e-12)
        for t in [.03, p.brake_time, .6]:
            eps = 1e-6
            before, after = np.array(p.sample(t-eps)), np.array(p.sample(t+eps))
            np.testing.assert_allclose((after[:2]-before[:2])/(2*eps), p.sample(t)[1:], atol=3e-5)
        np.testing.assert_allclose(p.sample(p.brake_time-1e-9), p.sample(p.brake_time+1e-9), atol=1e-7)

    def test_zero_and_upward_initial_velocity(self):
        for velocity in [0., .12]:
            p = BrakedHeightProfile(.7, velocity, .72, 1., 2.)
            np.testing.assert_allclose(p.sample(0), [.7, velocity, 0.], atol=1e-12)
            np.testing.assert_allclose(p.sample(1.2), [.72, 0., 0.], atol=1e-12)

    def test_rejects_unreachable_height_or_velocity_budget(self):
        for values in [(.6, -2., .6, .5, 2.), (.6, 0., .9, .2, 2.), (.6, 0., .6, 1., -1.)]:
            with self.assertRaises(ValueError):
                BrakedHeightProfile(*values)
