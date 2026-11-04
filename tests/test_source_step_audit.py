import unittest
import numpy as np
from analyze_source_step import summarize_source_step


class SourceStepAuditTests(unittest.TestCase):
    def recording(self):
        n = 5
        feet = np.tile([0., 0., .035, 0., 0., 0., 1.], (n, 2, 1))
        forces = np.zeros((n, 2, 3)); forces[:, 0, 2] = 300.; forces[-1, 1, 2] = 30.
        t = np.arange(n)*.002
        return dict(history_time=t, history_source_step_elapsed=t,
            history_frame=np.linspace(114, 129, n), history_commanded_mode=np.array([1, 1, 1, 1, 3]),
            history_feet=feet, history_foot_forces=forces,
            history_source_step_goal=feet[:, 1].copy(), history_com=np.zeros((n, 3)),
            history_com_target=np.zeros((n, 3)), history_source_step_touchdown=np.array([0, 0, 0, 0, 1]),
            history_pkl_joint_rmse=np.ones(n)*.2, history_pkl_root_position_error=np.ones(n)*.1,
            servo_time=t, servo_physics_time=t, servo_qp_update_index=np.arange(n),
            motion_tracking_valid=True, stopped_at_source_step_end=True)

    def test_bounded_success_never_certifies_complete_tracking(self):
        result = summarize_source_step(self.recording())
        self.assertTrue(result['source_step_passed'])
        self.assertFalse(result['full_pkl_tracking'])
        self.assertFalse(result['continuous_walking'])

    def test_short_bridge_reports_original_landing_error(self):
        data = self.recording()
        data['history_source_step_source_goal'] = data['history_source_step_goal'].copy()
        data['history_source_step_source_goal'][:,0] += .25
        result = summarize_source_step(data)
        self.assertTrue(result['adapted_landing'])
        self.assertEqual(result['scope'], 'adapted_bridge_swing_and_touchdown')
        self.assertAlmostEqual(result['landing_adaptation_m'], .25)
        self.assertAlmostEqual(result['original_landing_position_error_m'], .25)
        self.assertFalse(result['full_pkl_tracking'])

    def test_stop_endpoint_does_not_certify_following_transfer(self):
        data=self.recording()
        data.update(history_reference_phase=np.array(['SOURCE_SWING']*4+['LANDING_STOP']),
                    history_com_velocity=np.zeros((5,1,3)),
                    history_whole_body_reference_v=np.zeros((5,1,35)),
                    history_whole_body_reference_a=np.zeros((5,1,35)))
        result=summarize_source_step(data)
        self.assertTrue(result['landing_stop_endpoint_passed'])
        self.assertFalse(result['post_stop_hold_validated'])
        self.assertFalse(result['continuous_walking'])
        data['history_com_velocity'][-1,0,0]=.1
        self.assertFalse(summarize_source_step(data)['landing_stop_endpoint_passed'])

    def test_goal_or_load_or_controller_failure_rejects_success(self):
        for failure in ('position', 'force', 'controller', 'endpoint'):
            data = self.recording()
            if failure == 'position':
                data['history_feet'][-1, 1, 0] = .1
            elif failure == 'force':
                data['history_foot_forces'][-1, 1, 2] = 0.
            elif failure == 'controller':
                data['motion_tracking_valid'] = False
            else:
                data['stopped_at_source_step_end'] = False
            self.assertFalse(summarize_source_step(data)['source_step_passed'], failure)
