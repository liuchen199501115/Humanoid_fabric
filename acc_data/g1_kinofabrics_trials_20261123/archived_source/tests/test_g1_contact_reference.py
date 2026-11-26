import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'examples'))
from g1_contact_reference import ContactFootReference


class ContactReferenceTests(unittest.TestCase):
    def test_liftoff_matches_measured_state_and_returns_to_moving_source(self):
        bridge = ContactFootReference(.1)
        source = np.array([[0., 0., 0., 0., 0., 0., 1.], [0., 1., 0., 0., 0., 0., 1.]])
        velocity = np.full((2, 6), .2)
        acceleration = np.full((2, 6), .3)
        measured = source.copy()
        measured[0, :3] += .01
        twist = np.full((2, 6), .1)
        bridge.update(0., [True, True], measured, twist, source, velocity, acceleration)
        p, v, a, active = bridge.update(.02, [False, True], measured, twist, source, velocity, acceleration)
        np.testing.assert_allclose(p[0], measured[0], atol=1e-12)
        np.testing.assert_allclose(v[0], twist[0], atol=1e-12)
        np.testing.assert_allclose(a[0], 0., atol=1e-12)
        np.testing.assert_array_equal(active, [True, False])
        moved = source.copy(); moved[:, :3] += .3
        p, v, a, active = bridge.update(.13, [False, True], measured, twist, moved, velocity, acceleration)
        np.testing.assert_array_equal(p, moved)
        np.testing.assert_array_equal(v, velocity)
        np.testing.assert_array_equal(a, acceleration)
        self.assertFalse(active.any())
        np.testing.assert_array_equal(source[:, :3], [[0., 0., 0.], [0., 1., 0.]])

    def test_touchdown_cancels_bridge_and_initial_swing_is_unchanged(self):
        bridge = ContactFootReference(.1)
        pose = np.zeros((2, 7)); pose[:, 6] = 1.
        zero = np.zeros((2, 6))
        measured = pose.copy(); measured[:, 0] = .1
        p, _, _, active = bridge.update(0., [False, True], measured, zero, pose, zero, zero)
        np.testing.assert_array_equal(p, pose)
        self.assertFalse(active.any())
        bridge.update(.01, [True, True], measured, zero, pose, zero, zero)
        bridge.update(.02, [False, True], measured, zero, pose, zero, zero)
        p, _, _, active = bridge.update(.03, [True, True], measured, zero, pose, zero, zero)
        np.testing.assert_array_equal(p, pose)
        self.assertFalse(active.any())


if __name__ == '__main__':
    unittest.main()
