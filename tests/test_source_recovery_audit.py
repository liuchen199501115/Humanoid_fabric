import unittest
import numpy as np
from analyze_source_recovery import summarize_source_recovery
from tests import test_source_sequence_audit


class SourceRecoveryAuditTests(unittest.TestCase):
    def data(self):
        d = test_source_sequence_audit.SourceSequenceAuditTests().data()
        t = d['history_time']; n = len(t)
        active = t >= 3.
        progress = np.clip((t-3.)/1.8, 0., 1.)
        d['history_frame'] = np.minimum(t/3.*155.,155.)
        d['history_source_recovery'] = active
        targets = np.tile([1.,0.,.035,0.,0.,0.,1.], (n,2,1))
        d['history_source_recovery_target_feet'] = targets
        d['history_source_recovery_root'] = np.tile([1.,0.,.7], (n,1))
        d['history_q'] = np.zeros((n,36)); d['history_q'][:,0] = .2*progress
        d['history_feet'][:,0,0] = .4*progress
        d['history_source_step_goal'][active,0] = .4
        return d

    def report(self, d):
        return summarize_source_recovery(d,source_steps=1,recovery_steps=1,fixed_frame=155)

    def test_measured_error_reduction_without_source_progress(self):
        r = self.report(self.data())
        self.assertTrue(r['recovery_passed'])
        self.assertFalse(r['fixed_target_reached'])
        self.assertFalse(r['full_pkl_tracking'])
        self.assertAlmostEqual(r['final_root_xy_error_m'], .8)

    def test_two_recovery_steps_need_two_actual_landings(self):
        d=self.data();t=d['history_time']
        d['history_source_step_number'][t>=4.9]=3
        d['history_source_step_touchdown'][t>=4.9]=False
        swing=(t>=5.)&(t<5.2)
        d['history_feet'][swing,1,2]=.075
        d['history_foot_forces'][swing,1,2]=0.
        d['history_foot_forces'][swing,0,2]=320.
        d['history_commanded_mode'][swing]=1
        progress=np.clip((t-5.)/.3,0.,1.)
        d['history_feet'][:,1,0]=.4*progress
        d['history_q'][:,0]+=.2*progress
        d['history_source_step_touchdown'][t>=5.3]=True
        d['history_source_steps_finished']=t>=5.5
        r=summarize_source_recovery(d,source_steps=1,recovery_steps=2,fixed_frame=155)
        self.assertTrue(r['recovery_passed'])
        np.testing.assert_allclose(r['final_foot_xy_error_m'],[.6,.6])
        d['history_source_step_touchdown'][t>=4.9]=False
        self.assertFalse(summarize_source_recovery(d,1,2,155)['recovery_passed'])

    def test_moving_reference_cannot_look_like_recovery(self):
        for key in ('history_source_recovery_target_feet','history_source_recovery_root'):
            d = self.data(); d[key][-1] += .1
            self.assertFalse(self.report(d)['recovery_passed'])

    def test_phase_contact_and_real_error_are_required(self):
        for fault in ('clock','contact','feet','root','count','flag'):
            d = self.data()
            if fault == 'clock': d['history_frame'][1600] += 1.
            elif fault == 'contact': d['history_source_step_touchdown'][1500:] = False
            elif fault == 'feet': d['history_feet'][:,0,0] = 0.
            elif fault == 'root': d['history_q'][:,0] = 0.
            elif fault == 'count': d['history_source_step_number'][1600] = 3
            else: d['history_source_recovery'][-1] = False
            self.assertFalse(self.report(d)['recovery_passed'], fault)
