import sys
import unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'examples'))
sys.path.insert(0, str(ROOT/'acc_data/amp_walk_20261103/feedback_followup'))
from g1_landing_clock_chain import LandingClockChain


class LandingClockChainTest(unittest.TestCase):
    def test_each_gate_requires_its_own_measured_landing_and_preserves_progress(self):
        chain = LandingClockChain([(4.3, 4.5, 11), (4.6, 4.76, 12)], .1)
        for t in [0., 3., 4.3]:
            self.assertEqual(chain.sample((t, 1., 0.)), (t, 1., 0.))
        chain.update((5., 1., 0.), 10)
        self.assertEqual(chain.sample((5., 1., 0.)), (4.5, 0., 0.))
        chain.update((5., 1., 0.), 11)
        values = [chain.sample((t, 1., 0.)) for t in np.linspace(5., 6., 501)]
        self.assertTrue(np.all(np.diff([x[0] for x in values]) >= -1e-12))
        self.assertEqual(values[-1], (4.76, 0., 0.))
        chain.update((6., 1., 0.), 11)
        self.assertEqual(chain.sample((7., 1., 0.)), (4.76, 0., 0.))
        chain.update((7., 1., 0.), 12)
        self.assertGreater(chain.sample((7.2, 1., 0.))[0], 4.76)

    def test_chain_rule_matches_independent_finite_differences(self):
        chain = LandingClockChain([(4.3, 4.5, 11), (4.6, 4.76, 12)], .1)
        chain.update((5., 1., 0.), 11)
        def sample(t):
            return chain.sample((t+.03*(t-5)**2, 1+.06*(t-5), .06))
        for t in np.linspace(5.001, 5.9, 113):
            h=1e-6;lo,hi=sample(t-h),sample(t+h);actual=sample(t)
            np.testing.assert_allclose([(hi[0]-lo[0])/(2*h),(hi[1]-lo[1])/(2*h)], actual[1:], atol=2e-6, rtol=0)

    def test_invalid_intervals_and_counts_are_rejected(self):
        for schedules in [[(4.,3.,12)],[(4.,5.,12.5)],[(4.,5.,12),(4.9,6.,13)],[(4.,5.,12),(6.,7.,12)]]:
            with self.subTest(schedules=schedules), self.assertRaises(ValueError):
                LandingClockChain(schedules)
