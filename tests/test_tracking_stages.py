import unittest
import numpy as np
from run_tracking_stages import summarize_standing


class StandingAuditTests(unittest.TestCase):
    def data(self):
        n = 2000
        force = np.zeros((n, 2, 3)); force[..., 2] = 150.
        return dict(history_time=np.arange(n)*.002, history_frame=np.zeros(n),
                    history_commanded_mode=np.full(n, 3),
                    history_com=np.zeros((n, 1, 3)), history_com_target=np.zeros((n, 1, 3)),
                    history_foot_forces=force, motion_tracking_valid=True)

    def test_stable_standing_is_not_tracking(self):
        result = summarize_standing(self.data())
        self.assertTrue(result['standing_passed'])
        self.assertFalse(result['full_pkl_tracking'])

    def test_rejects_failure_advance_unloading_drift_and_bad_clock(self):
        for issue in ('controller', 'frame', 'force', 'drift', 'clock', 'nan', 'mode', 'short'):
            d = self.data()
            if issue == 'controller': d.pop('motion_tracking_valid')
            if issue == 'frame': d['history_frame'][-1] = 1
            if issue == 'force': d['history_foot_forces'][-1, 1, 2] = 0
            if issue == 'drift': d['history_com'][-1, 0, 0] = .01
            if issue == 'clock': d['history_time'][-1] = d['history_time'][-2]
            if issue == 'nan': d['history_com'][-1, 0, 0] = np.nan
            if issue == 'mode': d['history_commanded_mode'][-1] = 1
            if issue == 'short':
                d = {k: v[:10] if isinstance(v, np.ndarray) else v for k, v in d.items()}
            with self.subTest(issue=issue):
                self.assertFalse(summarize_standing(d)['standing_passed'])
