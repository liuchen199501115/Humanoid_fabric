import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'examples'))
from g1_unload_timing import unload_ramp_for_source


class UnloadTimingTests(unittest.TestCase):
    events = [dict(support=0, takeoff_frame=572, end_frame=640),
              dict(support=1, takeoff_frame=677, end_frame=762),
              dict(support=0, takeoff_frame=780, end_frame=802),
              dict(support=1, takeoff_frame=802, end_frame=884)]

    def test_late_measured_landing_leaves_only_25ms_before_right_swing(self):
        self.assertAlmostEqual(unload_ramp_for_source(
            self.events, 774, 240, 1, .04, .002), .025)
        # A slower source clock gives the ordinary ramp enough time.
        self.assertEqual(unload_ramp_for_source(
            self.events, 774, 100, 1, .04, .002), .04)

    def test_overdue_and_paused_transfers_preserve_positive_duration(self):
        self.assertEqual(unload_ramp_for_source(
            self.events, 790, 240, 1, .04, .002), .002)
        self.assertEqual(unload_ramp_for_source(
            self.events, 790, 0, 1, .04, .002), .04)
        self.assertEqual(unload_ramp_for_source(
            self.events, 810, 240, 1, .04, .002), .04)

    def test_opposite_foot_and_completed_events_do_not_set_the_deadline(self):
        self.assertEqual(unload_ramp_for_source(
            self.events, 774, 240, 0, .04, .002), .04)
        with self.assertRaises(ValueError):
            unload_ramp_for_source(self.events, 774, -1, 1, .04, .002)
