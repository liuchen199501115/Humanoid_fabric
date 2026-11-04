import unittest
import numpy as np
from fabrics_sim.reference.source_sequence import audit_source_schedule,source_recovery_active
from audit_full_source_clip import audit_clip


class FullSourceScheduleTests(unittest.TestCase):
    def test_entire_schedule_includes_tail_and_repeated_support(self):
        r=audit_source_schedule([3,1,1,3,2,3,2,3,3])
        self.assertEqual(r['frame_count'],9)
        self.assertEqual(r['final_double_support']['end_frame'],8)
        self.assertEqual(r['first_legacy_schedule_obstacle']['frame'],6)
        self.assertFalse(r['legacy_alternating_schedule_compatible'])
        covered=[i for s in r['segments'] for i in range(s['start_frame'],s['end_frame']+1)]
        self.assertEqual(covered,list(range(9)))
        self.assertFalse(r['full_tracking_passed'])

    def test_direct_switch_flight_and_missing_tail_are_explicit(self):
        r=audit_source_schedule([3,1,2,3,0,1])
        self.assertEqual(r['direct_switches'],[dict(frame=2,previous_mode=1,next_mode=2)])
        self.assertEqual(r['flight_segments'][0]['start_frame'],4)
        self.assertIsNone(r['final_double_support'])
        self.assertFalse(r['physical_feasibility_checked'])
        for invalid in ([],[3,float('nan')],[3,4],[[3,1]]):
            with self.assertRaises(ValueError):audit_source_schedule(invalid)

    def test_finite_recovery_resumes_instead_of_freezing_forever(self):
        self.assertEqual([source_recovery_active(i,3,2) for i in range(7)],
                         [False,False,False,True,True,False,False])
        self.assertTrue(source_recovery_active(100,3))
        self.assertFalse(source_recovery_active(100,0))
        for args in ((0,0,2),(1,3,1),(1,3,-2),(1.,3,2)):
            with self.assertRaises(ValueError):source_recovery_active(*args)

    def test_full_data_audit_checks_last_frame_limits_and_quaternion(self):
        q=np.zeros((10,36));q[:,6]=1.;q[-1,7]=2.;q[-1,6]=0.
        labels=np.ones((10,2),dtype=bool)
        r=audit_clip(q,30.,labels,-np.ones(29),np.ones(29))
        self.assertEqual(r['joint_limit_violation_frames'],1)
        self.assertEqual(r['invalid_quaternion_frames'],1)
        self.assertEqual(r['maximum_joint_limit_violation_rad'],1.)
        self.assertFalse(r['full_tracking_passed'])
        with self.assertRaises(ValueError):audit_clip(q,30.,labels[:-1],-np.ones(29),np.ones(29))
