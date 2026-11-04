import unittest
import numpy as np
from fabrics_sim.reference.bounded_steps import select_dynamic_steps, select_all_dynamic_steps, MeasuredSteps
from analyze_bounded_steps import summarize


class DynamicBoundedStepsTests(unittest.TestCase):
    def test_full_clip_excludes_unfinished_tail_but_keeps_direct_switches(self):
        modes = [2,2,1,1,3,2,2,3,1,1]
        events = select_all_dynamic_steps(modes)
        self.assertEqual([e['end_frame'] for e in events], [2,4,7])
        self.assertTrue(events[0]['starts_mid_swing'])
        self.assertTrue(events[0]['direct_support_switch'])
        self.assertEqual(select_all_dynamic_steps([3,3,1,1]), [])
        with self.assertRaises(ValueError):
            select_all_dynamic_steps([3,0,3])
        with self.assertRaises(ValueError):
            select_all_dynamic_steps([2,3,2,3])

    def test_full_clip_endpoint_does_not_claim_stationary_support(self):
        plan = dict(events=select_dynamic_steps([2,2,1,1,3],1),
                    landing_positions_world=[[.0,.0,.035]], continuous=True, full_motion=True)
        data = self.recording()
        data['stopped_at_tracking_steps'] = False
        data['stopped_at_motion_end'] = True
        report = summarize(data,plan)
        self.assertTrue(report['steps_tracking_passed'])
        self.assertFalse(report['stable_final_support'])

    def test_full_motion_requires_verified_flights_as_well_as_source_coverage(self):
        from analyze_motion_tracking import summarize_full_motion
        data = self.recording()
        data.update(source_frame_count=3, source_start_frame=0, simulation_completed=True,
                    source_completed=False, stopped_at_motion_end=True,
                    stopped_at_tracking_steps=False)
        plan = dict(events=select_dynamic_steps([2,2,1],1),
                    landing_positions_world=[[0,0,.035]], continuous=True, full_motion=True)
        self.assertTrue(summarize_full_motion(data,plan)['full_motion_tracking_passed'])
        data['history_feet'][0,0,2] = .035
        report = summarize_full_motion(data,plan)
        self.assertTrue(report['full_clip_tracking_passed'])
        self.assertFalse(report['all_source_landings_verified'])
        self.assertFalse(report['full_motion_tracking_passed'])

    def test_partial_first_swing_and_direct_switch_are_explicit(self):
        events = select_dynamic_steps([2,2,1,1,3,2,2,3,1,1,3],4)
        self.assertEqual([e['end_frame'] for e in events], [2,4,7,10])
        self.assertEqual([e['support'] for e in events], [1,0,1,0])
        self.assertTrue(events[0]['starts_mid_swing'])
        self.assertTrue(events[0]['direct_support_switch'])
        self.assertFalse(events[1]['starts_mid_swing'])
        self.assertEqual(events[0]['landing_window_end_frame'],5)
        with self.assertRaises(ValueError):
            select_dynamic_steps([2,2,1,1,3],3)

    def test_continuous_landing_needs_sustained_incoming_load(self):
        events = select_dynamic_steps([2,2,1,1,3],1)
        monitor = MeasuredSteps(events,[[.2,0,0]],continuous=True)
        feet = np.zeros((2,3))
        monitor.update(0,0,2,[0,320],[.03,0],feet,.002)
        feet[0,0]=.2
        for i in range(24):
            self.assertFalse(monitor.update(2,.1+i*.002,1,[300,0],[0,.03],feet,.002))
        self.assertTrue(monitor.update(2,.148,1,[300,0],[0,.03],feet,.002))
        self.assertTrue(monitor.report()['initial_partial_step'])

    def test_late_recovery_cannot_count_as_missed_landing(self):
        events = select_dynamic_steps([2,2,1,1,3,2,2,3],1)
        monitor = MeasuredSteps(events,[[0,0,0]],continuous=True)
        feet = np.zeros((2,3))
        monitor.update(0,0,2,[0,320],[.03,0],feet,.002)
        for i in range(50):
            monitor.update(7,1+i*.002,3,[150,150],[0,0],feet,.002)
        self.assertFalse(monitor.complete)

    def recording(self):
        n=28
        feet=np.zeros((n,2,7)); feet[:,:,2]=.035; feet[:,:,6]=1
        feet[0,0,2]=.07
        forces=np.zeros((n,2,3)); forces[0,1,2]=320; forces[1:,0,2]=320
        q=np.zeros((n,36)); q[:,2]=.8; q[:,6]=1
        return dict(history_time=np.arange(n)*.002, history_frame=np.r_[0,np.full(n-1,2.)],
            history_feet=feet, history_foot_forces=forces.copy(), history_ground_foot_forces=forces,
            history_measured_mode=np.r_[2,np.ones(n-1)], history_commanded_mode=np.r_[2,np.ones(n-1)],
            history_q=q,history_reference_q=q.copy(), motion_tracking_valid=True,
            stopped_at_tracking_steps=True, stopped_at_motion_end=False, source_frame_count=5)

    def test_audit_requires_ground_and_fidelity_as_well_as_landings(self):
        plan=dict(events=select_dynamic_steps([2,2,1,1,3],1),
                  landing_positions_world=[[0,0,.035]],continuous=True)
        data=self.recording()
        report=summarize(data,plan)
        self.assertTrue(report['steps_tracking_passed'])
        self.assertFalse(report['stable_final_support'])
        data['history_q'][:,7:]=1.
        self.assertFalse(summarize(data,plan)['steps_tracking_passed'])
        data['history_q'][:,7:]=0.
        del data['history_ground_foot_forces']
        self.assertFalse(summarize(data,plan)['steps_tracking_passed'])

    def test_clock_advance_and_stop_flag_without_swing_do_not_pass(self):
        plan=dict(events=select_dynamic_steps([2,2,1,1,3],1),
                  landing_positions_world=[[0,0,.035]],continuous=True)
        data=self.recording()
        data['history_feet'][0,0,2]=.035
        self.assertFalse(summarize(data,plan)['steps_tracking_passed'])


if __name__=='__main__':
    unittest.main()
