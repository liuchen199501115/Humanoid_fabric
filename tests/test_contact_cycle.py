import unittest
from fabrics_sim.solver.contact_cycle import ContactCycle, smooth_height


class ContactCycleTest(unittest.TestCase):
    def test_no_release_without_measured_readiness(self):
        cycle = ContactCycle()
        for i in range(100):
            cycle.update(i*.02, ready=False, clearance=.04, swing_load=0., swing_speed=0., settled=True)
        self.assertEqual(cycle.phase, "transfer")

    def test_full_cycle_requires_clearance_and_stable_touchdown(self):
        cycle = ContactCycle()
        def confirm(time, **kwargs):
            obs = dict(ready=True, clearance=.04, swing_load=0., swing_speed=0., settled=False)
            obs.update(kwargs)
            for i in range(10):
                cycle.update(time+i*.02, **obs)
        confirm(0.)
        self.assertEqual(cycle.phase, "lift")
        confirm(3., clearance=0.)
        self.assertEqual(cycle.phase, "lift")
        confirm(4.)
        self.assertEqual(cycle.phase, "hold")
        confirm(6.)
        self.assertEqual(cycle.phase, "lower")
        confirm(10., swing_load=100., swing_speed=.5)
        self.assertEqual(cycle.phase, "lower")
        confirm(11., swing_load=100.)
        self.assertEqual(cycle.phase, "settle")
        confirm(13., swing_load=100., settled=True)
        self.assertEqual(cycle.phase, "done")
        self.assertTrue(cycle.ever_clear)

    def test_quintic_endpoints_and_derivatives(self):
        for start, end in [(0., .04), (.04, 0.)]:
            self.assertEqual(smooth_height(0., 3., start, end), (start, 0., 0.))
            self.assertEqual(smooth_height(3., 3., start, end), (end, 0., 0.))
            t, eps = .7, 1e-5
            p, v, a = smooth_height(t, 3., start, end)
            minus, plus = smooth_height(t-eps, 3., start, end), smooth_height(t+eps, 3., start, end)
            self.assertAlmostEqual(v, (plus[0]-minus[0])/(2*eps), places=8)
            self.assertAlmostEqual(a, (plus[1]-minus[1])/(2*eps), places=8)
