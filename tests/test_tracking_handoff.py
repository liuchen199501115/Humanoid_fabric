import unittest
import numpy as np
from plan_tracking_first_step import handoff_errors

class HandoffTests(unittest.TestCase):
    def data(self):
        q = np.zeros((1, 1, 36)); q[..., 6] = 1.
        v = np.zeros((1, 1, 35))
        return dict(history_q=q.copy(), history_qd=v.copy(),
                    history_whole_body_reference_q=q.copy(), history_whole_body_reference_v=v.copy())

    def test_detects_measured_arm_difference_without_changing_reference(self):
        d = self.data(); d['history_q'][0, 0, 22] = .25
        d['history_qd'][0, 0, 21] = .01
        r = handoff_errors(d, [str(i) for i in range(29)])
        self.assertEqual(r['worst_joint'], '15')
        self.assertEqual(r['max_joint_error_rad'], .25)
        self.assertEqual(r['max_velocity_error'], .01)
        self.assertEqual(d['history_whole_body_reference_q'][0, 0, 22], 0.)

    def test_quaternion_sign_equivalence(self):
        d = self.data(); d['history_q'][0, 0, 6] = -1
        self.assertEqual(handoff_errors(d, list(range(29)))['root_orientation_error_rad'], 0.)

    def test_invalid_states_rejected(self):
        for field in ('history_q', 'history_qd'):
            d = self.data(); d[field][0, 0, 0] = np.nan
            with self.assertRaises(ValueError): handoff_errors(d, list(range(29)))
        d = self.data(); d['history_q'][0, 0, 6] = 0
        with self.assertRaises(ValueError): handoff_errors(d, list(range(29)))
