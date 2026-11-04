import unittest
import numpy as np
from analyze_support_switch import summarize_support_switch


class SupportSwitchAuditTests(unittest.TestCase):
    def data(self):
        t=np.arange(3000)*.002;n=len(t)
        feet=np.tile([0.,0.,.035,0.,0.,0.,1.],(n,2,1));feet[t>=2.,0,2]=.075
        forces=np.zeros((n,2,3));forces[:,1,2]=320.
        return dict(history_time=t,history_support_switch_target=np.where(t<1.,-1,1),
            history_support_switch_release_time=np.where(t<2.,-1,2.),
            history_commanded_mode=np.where(t<1.,1,np.where(t<2.,3,2)),
            history_feet=feet,history_foot_forces=forces,history_com=np.zeros((n,1,3)),
            history_com_target=np.zeros((n,1,3)),qp_validated=True,servo_valid=True,
            servo_time=t,servo_physics_time=t,servo_qp_update_index=np.arange(n))

    def test_actual_side_change_and_hold_pass_without_certifying_gait(self):
        r=summarize_support_switch(self.data())
        self.assertTrue(r['support_switch_passed']);self.assertFalse(r['full_pkl_tracking'])
        self.assertFalse(r['continuous_walking'])

    def test_mode_change_alone_or_no_previous_support_or_bad_control_fails(self):
        for failure in ('clearance','force','old_support','controller','gate'):
            d=self.data()
            if failure=='clearance':d['history_feet'][-1,0,2]=.035
            elif failure=='force':d['history_foot_forces'][-1,0,2]=20.
            elif failure=='old_support':d['history_commanded_mode'][:500]=3
            elif failure=='controller':d['qp_validated']=False
            else:d['history_support_switch_release_time'][:]=-1
            self.assertFalse(summarize_support_switch(d)['support_switch_passed'],failure)
