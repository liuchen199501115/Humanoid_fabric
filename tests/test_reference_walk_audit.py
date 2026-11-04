import unittest
import numpy as np
from analyze_reference_walk import summarize_walk


class WalkAuditTests(unittest.TestCase):
    def recording(self, modes):
        n=len(modes)
        q=np.zeros((n,1,36));q[:,:,6]=1
        q[:,0,0]=np.linspace(0,.3,n)
        feet=np.zeros((n,1,2,7));feet[:,:,:,6]=1
        forces=np.zeros((n,2,3))
        for i,mode in enumerate(modes):
            for foot in range(2):
                if mode & (1<<foot):forces[i,foot,2]=150
        phases=['TRACK']*n
        for i in range(1,n-1):
            if modes[i-1] in (1,2) and modes[i]==3:
                phases[i]='LANDING';phases[i+1]='SETTLE'
        return dict(history_time=np.arange(n)*.02,history_commanded_mode=modes,
                    history_feet=feet,history_foot_forces=forces,history_q=q,
                    history_frame=np.arange(n),history_reference_phase=phases,
                    history_reference_wait=np.zeros(n),history_pkl_joint_rmse=np.zeros(n),
                    history_pkl_root_position_error=np.zeros(n),
                    simulation_completed=True,motion_tracking_valid=True)

    def test_controller_success_without_steps_is_not_walking(self):
        report=summarize_walk(self.recording([3]*10))
        self.assertTrue(report['controller_valid'])
        self.assertFalse(report['walking_segment_passed'])

    def test_alternating_measured_touchdowns_required(self):
        data=self.recording([3,1,3,3,3,2,3,3,3,1,3,3,3,2,3,3,3])
        self.assertTrue(summarize_walk(data)['walking_segment_passed'])
        data['history_foot_forces'][2,1,2]=0
        self.assertFalse(summarize_walk(data)['walking_segment_passed'])

    def test_slipping_support_rejects_segment(self):
        data=self.recording([3,1,3,3,3,2,3,3,3,1,3,3,3,2,3,3,3])
        data['history_feet'][1,0,0,0]=.1
        self.assertFalse(summarize_walk(data)['walking_segment_passed'])


if __name__=='__main__':unittest.main()
