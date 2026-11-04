import unittest
import numpy as np
from fabrics_sim.solver.contact_observation import ground_support_forces
from fabrics_sim.reference.bounded_steps import select_dynamic_steps, MeasuredSteps


class GroundSupportTests(unittest.TestCase):
    def test_only_filtered_ground_can_load_swing_foot(self):
        # A self-collision could produce 150 N total on the airborne left foot.
        # The filtered ground report is zero, so it must not count as support.
        filtered = np.array([[[0., 0., 0.]], [[2., 0., 320.]]])
        support = ground_support_forces(filtered)
        np.testing.assert_array_equal(support[:,2], [0., 320.])
        monitor = MeasuredSteps(select_dynamic_steps([2,2,1,1,3],1), [[0,0,0]], continuous=True)
        feet = np.zeros((2,3))
        monitor.update(0,0,2,support[:,2],[.03,0],feet,.002)
        for _ in range(30):
            monitor.update(2,1,2,support[:,2],[0,0],feet,.002)
        self.assertFalse(monitor.complete)

    def test_sum_ground_colliders_without_clipping_signed_components(self):
        filtered = np.array([[[2,1,10],[-1,-2,20]], [[0,0,0],[0,0,100]]])
        np.testing.assert_array_equal(ground_support_forces(filtered), [[1,-1,30],[0,0,100]])

    def test_missing_or_corrupt_filter_never_falls_back_to_net(self):
        for invalid in (np.zeros((2,3)), np.zeros((2,0,3)), np.full((2,1,3),np.nan)):
            with self.subTest(shape=invalid.shape), self.assertRaises(ValueError):
                ground_support_forces(invalid)


if __name__ == '__main__':
    unittest.main()
