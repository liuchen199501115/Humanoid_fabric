import unittest
import numpy as np
from analyze_source_sequence import summarize_source_sequence


class SourceSequenceAuditTests(unittest.TestCase):
    def data(self):
        t=np.arange(4500)*.002;n=len(t)
        feet=np.tile([0.,0.,.035,0.,0.,0.,1.],(n,2,1))
        forces=np.zeros((n,2,3));forces[:,:,2]=160.
        mode=np.full(n,3)
        for start,end,swing in ((.5,1.,1),(3.,4.,0)):
            active=(t>=start)&(t<end);feet[active,swing,2]=.075
            forces[active,swing,2]=0.;forces[active,1-swing,2]=320.;mode[active]=2-swing
        goal=np.tile([0.,0.,.035,0.,0.,0.,1.],(n,1))
        return dict(history_time=t,history_source_step_number=np.where(t<3.,1,2),
            history_frame=np.minimum(t/5.*155.,155.),history_source_step_touchdown=((t>=2.)&(t<3.))|(t>=4.8),
            history_source_steps_finished=t>=5.,history_feet=feet,history_foot_forces=forces,
            history_source_step_goal=goal,history_source_step_source_goal=goal.copy(),
            history_com=np.zeros((n,1,3)),history_com_target=np.zeros((n,1,3)),
            history_commanded_mode=mode,qp_validated=True,servo_valid=True,
            servo_time=t,servo_physics_time=t,servo_qp_update_index=np.arange(n))

    def test_one_complete_source_stride_still_requires_physical_landing(self):
        d=self.data();t=d['history_time']
        d['history_source_step_number'][:]=1
        d['history_source_steps_finished']=t>=2.
        d['history_frame']=np.minimum(t/2.*129.,129.)
        d['history_feet'][t>=2.,:,2]=.035
        d['history_foot_forces'][t>=2.,:,2]=160.
        d['history_commanded_mode'][t>=2.]=3
        d['history_source_step_touchdown']=t>=2.
        self.assertTrue(summarize_source_sequence(d,1,129.)['sequence_passed'])
        d['history_source_step_touchdown'][:]=False
        self.assertFalse(summarize_source_sequence(d,1,129.)['sequence_passed'])

    def test_two_landings_and_stopped_hold(self):
        r=summarize_source_sequence(self.data())
        self.assertTrue(r['two_steps_passed']);self.assertFalse(r['full_pkl_tracking'])
        self.assertEqual([x['swing'] for x in r['landings']],['right','left'])

    def test_source_phase_alone_does_not_certify_two_steps(self):
        for fault in ('touchdown','clearance','support','hold','controller','phase','speed'):
            d=self.data()
            if fault=='touchdown':d['history_source_step_touchdown'][1500:]=False
            elif fault=='clearance':d['history_feet'][1500:,0,2]=.035
            elif fault=='support':d['history_commanded_mode'][:]=3
            elif fault=='hold':d['history_source_steps_finished'][:]=False
            elif fault=='controller':d['qp_validated']=False
            elif fault=='phase':d['history_frame'][-1]=129.
            else:d['history_com'][-1,0,0]=.01
            self.assertFalse(summarize_source_sequence(d)['two_steps_passed'],fault)


    def test_three_steps_require_third_measured_landing(self):
        d=self.data();t=d['history_time']
        d['history_source_step_end_frame']=np.full(len(t),176.)
        self.assertFalse(summarize_source_sequence(d,3)['sequence_passed'])
        d['history_source_step_number'][t>=4.9]=3
        d['history_source_step_touchdown'][t>=4.9]=False
        swing=(t>=5.)&(t<5.2)
        d['history_feet'][swing,1,2]=.075
        d['history_foot_forces'][swing,1,2]=0.
        d['history_foot_forces'][swing,0,2]=320.
        d['history_commanded_mode'][swing]=1
        d['history_source_step_touchdown'][t>=5.3]=True
        d['history_source_steps_finished']=t>=5.5
        d['history_frame']=np.minimum(t/5.5*176.,176.)
        self.assertTrue(summarize_source_sequence(d,3)['sequence_passed'])
        self.assertFalse(summarize_source_sequence(d,2)['sequence_passed'])
        d['history_foot_forces'][swing,1,2]=30.
        self.assertFalse(summarize_source_sequence(d,3)['sequence_passed'])
