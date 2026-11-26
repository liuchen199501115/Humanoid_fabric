import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'examples'))
from prepare_g1_landing_reference import correct_landing, LegKinematics


class LandingReferenceTest(unittest.TestCase):
    def setUp(self):
        names = [side+'_'+joint+'_joint' for side in ('left', 'right')
                 for joint in ('hip_pitch', 'hip_roll', 'hip_yaw', 'knee', 'ankle_pitch', 'ankle_roll')]
        self.model = LegKinematics(ROOT/'src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf', names)
        self.q = np.zeros((31, 36), dtype=np.float32)
        self.q[:, 2], self.q[:, 6] = .8, 1.
        for start in (7, 13):
            self.q[:, start], self.q[:, start+3], self.q[:, start+4] = -.2, .5, -.3
        self.first = dict(support=0, takeoff_frame=5, end_frame=10)
        self.next = dict(support=0, takeoff_frame=20, end_frame=25)

    def test_full_pose_correction_preserves_stance_root_and_landing_rotation(self):
        original = self.q.copy()
        result, report = correct_landing(self.model, self.q, self.first, self.next, [0., -.015, 0.])
        np.testing.assert_array_equal(self.q, original)
        np.testing.assert_array_equal(result[:6], original[:6])
        np.testing.assert_array_equal(result[25:], original[25:])
        unchanged = list(range(13))+list(range(19, 36))
        np.testing.assert_array_equal(result[:, unchanged], original[:, unchanged])
        for k in range(10, 21):
            p, r, _ = self.model.evaluate(result[k], 1)
            p0, r0, _ = self.model.evaluate(original[k], 1)
            np.testing.assert_allclose(p-p0, [0., -.015, 0.], atol=2e-5)
            np.testing.assert_allclose(r, r0, atol=2e-4)
        self.assertLess(report['peak_position_residual_m'], 2e-5)

    def test_unreachable_target_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'IK failed'):
            correct_landing(self.model, self.q, self.first, self.next, [0., 100., 0.])

    def test_wrong_foot_and_nonfinite_offset_are_rejected(self):
        with self.assertRaises(ValueError):
            correct_landing(self.model, self.q, self.first, dict(self.next, support=1), [0., 0., 0.])
        with self.assertRaises(ValueError):
            correct_landing(self.model, self.q, self.first, self.next, [0., np.nan, 0.])


if __name__ == '__main__':
    unittest.main()
