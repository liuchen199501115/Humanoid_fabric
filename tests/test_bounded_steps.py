import unittest
import numpy as np
from fabrics_sim.reference.bounded_steps import select_steps, MeasuredSteps

class BoundedStepsTests(unittest.TestCase):
    def test_selects_four_without_recovery_or_extra_frames(self):
        modes=[3,3,1,1,3,3,2,2,3,1,1,3,2,2,3,3,1,1,3]
        events=select_steps(modes,4)
        self.assertEqual([e['support'] for e in events],[0,1,0,1])
        self.assertEqual([e['end_frame'] for e in events],[4,8,11,14])
        with self.assertRaises(ValueError):select_steps([3,1,3,1,3],2)

    def test_clock_and_false_landings_do_not_count(self):
        events=select_steps([3,1,1,3],1)
        m=MeasuredSteps(events,[[.2,0,0]])
        feet=np.zeros((2,3))
        for _ in range(100):m.update(3,1,3,[150,150],[0,0],feet,.002)
        self.assertFalse(m.complete)
        m.update(2,1,1,[320,0],[0,.03],feet,.002)
        for _ in range(100):m.update(3,2,3,[150,150],[0,0],feet,.002)
        self.assertFalse(m.complete) # landed at wrong location
        feet[1,0]=.2
        for _ in range(25):m.update(3,2,3,[150,150],[0,0],feet,.002)
        self.assertTrue(m.complete)
        self.assertEqual(m.landings[0]['swing'],'right')

    def test_recording_cannot_pass_from_completion_flags_alone(self):
        from analyze_bounded_steps import summarize
        feet=np.zeros((30,2,7));feet[:,:,2]=.035;feet[:,:,6]=1.
        data=dict(history_time=np.arange(30)*.002,history_frame=np.full(30,3.),
            history_feet=feet,history_foot_forces=np.tile([[[0,0,150],[0,0,150]]],(30,1,1)),
            history_commanded_mode=np.full(30,3),motion_tracking_valid=True,stopped_at_motion_end=True)
        r=summarize(data,dict(events=select_steps([3,1,1,3],1),landing_positions_world=[[0,0,.035]]))
        self.assertFalse(r['steps_tracking_passed'])
        self.assertEqual(r['measured_steps'],0)

    def test_left_first(self):
        m=MeasuredSteps(select_steps([3,2,2,3],1),[[0,0,0]])
        m.update(2,1,2,[0,320],[.03,0],np.zeros((2,3)),.002)
        for _ in range(25):m.update(3,2,3,[150,150],[0,0],np.zeros((2,3)),.002)
        self.assertEqual(m.landings[0]['swing'],'left')

class LandingDiagnosticTests(unittest.TestCase):
    def test_low_clearance_is_not_hidden_by_long_loaded_hold(self):
        from analyze_bounded_steps import next_step_diagnostics
        from fabrics_sim.reference.bounded_steps import select_dynamic_steps
        events=select_dynamic_steps([1,1,3,3],1)
        m=MeasuredSteps(events,[[0,0,0]],continuous=True)
        t=np.arange(33)*.002;frame=np.r_[0,1,np.full(31,2)]
        modes=np.r_[1,1,np.full(31,3)]
        forces=np.tile([150.,150.],(33,1));forces[:2]=[320.,0.]
        clear=np.zeros((33,2));clear[:2,1]=.019
        feet=np.zeros((33,2,7))
        for i in range(33):m.update(frame[i],t[i],modes[i],forces[i],clear[i],feet[i,:,:3],.002)
        d=next_step_diagnostics(m,t,frame,modes,forces,clear,feet)
        self.assertEqual(len(m.landings),0)
        self.assertEqual(d['missing_evidence'],'verified_flight')
        self.assertAlmostEqual(d['peak_unloaded_clearance_m'],.019)
        self.assertAlmostEqual(d['best_loaded_landing_error_m'],0.)
        self.assertEqual(d['longest_verified_landing_hold_s'],0.)
        # Crossing the existing threshold enables the SAME unchanged monitor.
        clear[:2,1]=.021;m=MeasuredSteps(events,[[0,0,0]],continuous=True)
        for i in range(33):m.update(frame[i],t[i],modes[i],forces[i],clear[i],feet[i,:,:3],.002)
        self.assertTrue(m.complete)
        self.assertIsNone(next_step_diagnostics(m,t,frame,modes,forces,clear,feet))

    def test_discontinuous_contact_does_not_count_as_continuous_hold(self):
        from analyze_bounded_steps import next_step_diagnostics
        from fabrics_sim.reference.bounded_steps import select_dynamic_steps
        m=MeasuredSteps(select_dynamic_steps([1,1,3,3],1),[[0,0,0]],continuous=True)
        t=np.arange(42)*.002;frame=np.r_[0,1,np.full(40,2)]
        modes=np.r_[1,1,np.full(40,3)]
        forces=np.tile([150.,150.],(42,1));forces[:2]=[320.,0.];forces[20,1]=0.
        clear=np.zeros((42,2));clear[:2,1]=.03;feet=np.zeros((42,2,7))
        for i in range(42):m.update(frame[i],t[i],modes[i],forces[i],clear[i],feet[i,:,:3],.002)
        d=next_step_diagnostics(m,t,frame,modes,forces,clear,feet)
        self.assertFalse(m.complete)
        self.assertEqual(d['missing_evidence'],'continuous_landing_hold')
        self.assertAlmostEqual(d['longest_verified_landing_hold_s'],.042)

class SourcePoseLandingTests(unittest.TestCase):
    def test_rolling_link_follows_original_pose_without_changing_evidence_gates(self):
        from fabrics_sim.reference.bounded_steps import select_dynamic_steps
        events=select_dynamic_steps([1,1,3,3],1)
        fixed=MeasuredSteps(events,[[0,0,0]],continuous=True)
        moving=MeasuredSteps(events,[[0,0,0]],continuous=True,track_source_pose=True)
        feet=np.zeros((2,3))
        for monitor in [fixed,moving]:
            monitor.update(1,0,1,[320,0],[0,.03],feet,.002,reference_feet=feet)
        for i in range(35):
            feet[1,0]=min(.06,i*.004)
            for monitor in [fixed,moving]:
                monitor.update(2,.002*(i+1),3,[150,150],[0,0],feet,.002,reference_feet=feet)
        self.assertTrue(moving.complete)
        self.assertFalse(fixed.complete)
        self.assertEqual(moving.landings[0]['source_contact_onset_goal'],[0.,0.,0.])
        self.assertAlmostEqual(moving.landings[0]['landing_error_m'],0.)
        self.assertEqual(moving.report()['landing_goal_policy'],'source_pose')
        # Same-frame target does not permit a foot displaced 5 cm from source.
        wrong=MeasuredSteps(events,[[0,0,0]],continuous=True,track_source_pose=True)
        wrong.update(1,0,1,[320,0],[0,.03],feet,.002,reference_feet=feet)
        for i in range(35):
            actual=feet.copy();actual[1,0]+=.05
            wrong.update(2,.002*(i+1),3,[150,150],[0,0],actual,.002,reference_feet=feet)
        self.assertFalse(wrong.complete)

    def test_offline_audit_reports_source_pose_and_keeps_fixed_goal_comparison(self):
        from analyze_bounded_steps import summarize
        from fabrics_sim.reference.bounded_steps import select_dynamic_steps
        count=37;feet=np.zeros((count,2,7));feet[:,:,6]=1.
        feet[:,:,2]=.035;feet[:2,1,2]=.065
        feet[2:,1,0]=np.minimum(.06,np.arange(count-2)*.004)
        forces=np.tile([[[0.,0.,150.],[0.,0.,150.]]],(count,1,1));forces[:2,:,2]=[320.,0.]
        data=dict(history_time=np.arange(count)*.002,history_frame=np.r_[0,1,np.full(count-2,2)],
            history_feet=feet,history_ground_foot_forces=forces,
            history_measured_mode=np.r_[1,1,np.full(count-2,3)],
            history_source_feet_reference=feet[:,:,:3].copy())
        plan=dict(events=select_dynamic_steps([1,1,3,3],1),landing_positions_world=[[0,0,.035]],
                  continuous=True,landing_goal_policy='source_pose')
        result=summarize(data,plan)
        self.assertEqual(result['measured_steps'],1)
        self.assertEqual(result['fixed_contact_onset_audit']['measured_steps'],0)
        self.assertIsNone(result['next_step_diagnostics'])
        self.assertFalse(result['steps_tracking_passed']) # no controller or PKL certificate

    def test_missing_original_foot_history_rejects_source_pose_audit(self):
        from analyze_bounded_steps import summarize
        from fabrics_sim.reference.bounded_steps import select_dynamic_steps
        feet=np.zeros((30,2,7));feet[:,:,6]=1.
        data=dict(history_time=np.arange(30)*.002,history_frame=np.full(30,2.),
            history_feet=feet,history_ground_foot_forces=np.tile([[[0,0,150],[0,0,150]]],(30,1,1)),
            history_measured_mode=np.full(30,3))
        plan=dict(events=select_dynamic_steps([1,1,3,3],1),landing_positions_world=[[0,0,0]],
                  continuous=True,landing_goal_policy='source_pose')
        with self.assertRaisesRegex(ValueError,'original source feet'):summarize(data,plan)

if __name__=='__main__':unittest.main()
