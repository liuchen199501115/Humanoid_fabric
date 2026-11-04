import unittest
import numpy as np
from analyze_motion_tracking import summarize_tracking


class TrackingAuditTests(unittest.TestCase):
    def recording(self):
        q=np.zeros((4,1,36));q[:,:,6]=1
        return dict(history_q=q,history_reference_q=q.copy(),history_frame=[0,1,2,3],
                    source_frame_count=4,source_start_frame=0,source_completed=True,
                    simulation_completed=True,motion_tracking_valid=True)

    def test_complete_accurate_tracking_passes(self):
        self.assertTrue(summarize_tracking(self.recording())['full_tracking_passed'])

    def test_standing_or_partial_source_cannot_pass(self):
        data=self.recording();data['history_frame']=[0,1,1,1]
        self.assertFalse(summarize_tracking(data)['full_tracking_passed'])
        data=self.recording();data['source_start_frame']=1
        self.assertFalse(summarize_tracking(data)['full_tracking_passed'])

    def test_completed_scaled_path_does_not_pass(self):
        data=self.recording();data['history_q'][-1,0,0]=1
        result=summarize_tracking(data)
        self.assertTrue(result['full_source_covered'])
        self.assertFalse(result['pkl_fidelity_passed'])
        self.assertFalse(result['full_tracking_passed'])

    def test_full_clip_tracks_a_moving_endpoint_without_claiming_settling(self):
        data = self.recording()
        data['source_completed'] = False
        report = summarize_tracking(data)
        self.assertTrue(report['full_clip_tracking_passed'])
        self.assertFalse(report['full_tracking_passed'])
        self.assertFalse(report['final_support_settled'])
        data['history_frame'] = [0,1,2,2]
        self.assertFalse(summarize_tracking(data)['full_clip_tracking_passed'])
        data = self.recording()
        data['motion_tracking_valid'] = False
        self.assertFalse(summarize_tracking(data)['full_clip_tracking_passed'])

    def test_orientation_and_final_settling_are_required(self):
        data=self.recording();data['history_q'][-1,0,3:7]=[0,0,1,0]
        self.assertFalse(summarize_tracking(data)['full_tracking_passed'])
        data=self.recording();data['source_completed']=False
        self.assertFalse(summarize_tracking(data)['full_tracking_passed'])


if __name__=='__main__':unittest.main()
