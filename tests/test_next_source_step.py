import unittest
from fabrics_sim.reference.source_sequence import next_source_landing


class NextSourceStepTests(unittest.TestCase):
    def test_next_alternating_swing_and_landing(self):
        self.assertEqual(next_source_landing([1,3,3,2,2,3,1,3],1,1),(3,5))
        self.assertEqual(next_source_landing([2,3,1,1,3],1,0),(2,4))

    def test_missing_or_wrong_support_is_rejected(self):
        for modes,start,support in (([1,3,1,3],1,1),([1,3,3],1,1),
                                    ([1,3,2,2],1,1),([1,3,2,1],1,1),
                                    ([1,0,2,3],1,1),([1,3,2,3],1.5,1)):
            with self.assertRaises(ValueError):next_source_landing(modes,start,support)


class SourceBodyJoinTests(unittest.TestCase):
    def test_boundary_end_and_quaternion_sign(self):
        import torch
        from fabrics_sim.reference.source_step import blend_source_body
        source=torch.zeros(27,36);source[:,6]=-1.;source[:,19:]=.4
        initial=source[:1].clone();initial[:,6]=1.;initial[:,19:]=-.2
        original=source.clone()
        result=blend_source_body(source,initial)
        self.assertTrue(torch.equal(source,original))
        self.assertTrue(torch.allclose(result[0],initial[0]))
        self.assertTrue(torch.allclose(result[-1,19:],source[-1,19:]))
        self.assertTrue(torch.allclose(result[:,3:7].norm(dim=-1),torch.ones(27)))
        self.assertLess(float((result[1,19:]-result[0,19:]).abs().max()),.001)


class TakeoffClockTests(unittest.TestCase):
    def test_transfer_handoff_does_not_advance_past_source_start(self):
        from fabrics_sim.reference.manager import ManagedClock
        from fabrics_sim.reference.source_sequence import hold_takeoff_clock
        clock=ManagedClock(114.036,113.99,114.07,1,1,'SWING',.6,0.)
        held=hold_takeoff_clock(clock,114)
        self.assertEqual((held.frame,held.next_frame,held.rate),(114.,114.,0.))
        self.assertEqual(held.previous_frame,clock.previous_frame)
        self.assertEqual(held.mode,1)

    def test_handoff_rejects_a_real_rewind_or_skip(self):
        from fabrics_sim.reference.manager import ManagedClock
        from fabrics_sim.reference.source_sequence import hold_takeoff_clock
        for current,previous in ((114.1,114.05),(113.9,113.8)):
            with self.assertRaises(RuntimeError):
                hold_takeoff_clock(ManagedClock(current,previous,current,1,1,'SWING',.6,0.),114)
