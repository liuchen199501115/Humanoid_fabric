import unittest
import numpy as np
from fabrics_sim.reference.manager import ReferenceManager, infer_contact_plan


class ReferenceManagerTests(unittest.TestCase):
    def test_observed_takeoff_keeps_confirmed_live_gates_without_source_jump(self):
        m=ReferenceManager(np.r_[np.full(137,3),np.full(18,2),np.full(30,3)],30,
                           align_takeoff_frame=True,takeoff_lead=.15,continuous_source_clock=True)
        m.tau=130/30;m.rate=.1;m.pending_takeoff=137
        for _ in range(30):
            clock=m.step([5.,330.],[True,True],.002,sole_clearance=[0,0],balance_ready=[False,True])
            self.assertEqual(clock.mode,3)
        old=m.tau*30
        self.assertLess(old,132.5)
        clock=m.step([0.,330.],[False,True],.002,sole_clearance=[.01,0],balance_ready=[False,True])
        self.assertEqual(clock.mode,2);self.assertEqual(clock.phase,'SWING')
        self.assertLessEqual(clock.frame-old,.061)
        self.assertEqual(m.landing_frame,155)

    def test_observed_takeoff_never_bypasses_settle_preview_or_load_confirmation(self):
        for reason in ('settling','preview','confirmation','load','wrong_support'):
            m=ReferenceManager(np.r_[np.full(137,3),np.full(18,2),np.full(30,3)],30,
                               align_takeoff_frame=True,continuous_source_clock=True)
            m.tau=130/30;m.pending_takeoff=137;m.confirmed=30;m.transfer_load_latched=True
            m.settling=reason=='settling'
            if reason=='confirmation':m.confirmed=29
            forces=[0.,90.] if reason=='load' else [0.,330.]
            contacts=[True,False] if reason=='wrong_support' else [False,True]
            clearance=[0,.01] if reason=='wrong_support' else [.01,0]
            with self.subTest(reason=reason),self.assertRaises(RuntimeError):
                m.step(forces,contacts,.002,sole_clearance=clearance,balance_ready=[False,reason!='preview'])

    def test_incoming_readiness_spans_settle_without_duplicate_confirmation(self):
        for preview in (False,True):
            m=ReferenceManager(np.r_[np.full(20,3),np.full(30,2)],30,
                               settle_duration=.1,settle_on_incoming=True)
            m.tau=10/30;m.mode=3;m.settling=True;m.landing_support=1
            for _ in range(50):
                clock=m.step([5.,300.],[True,True],.002,sole_clearance=[0,0],
                             sole_max_height=[0,0],balance_ready=[False,preview])
                self.assertEqual(clock.mode,3)  # Settling still owns release.
            self.assertFalse(m.settling)
            clock=m.step([5.,300.],[True,True],.002,sole_clearance=[0,0],
                         sole_max_height=[0,0],balance_ready=[False,preview])
            self.assertEqual(clock.mode,2 if preview else 3)

    def test_lost_load_clears_confirmation_accumulated_during_settle(self):
        m=ReferenceManager(np.r_[np.full(20,3),np.full(30,2)],30,
                           settle_duration=.1,settle_on_incoming=True)
        m.tau=10/30;m.mode=3;m.settling=True;m.landing_support=1
        for _ in range(40):
            m.step([5.,300.],[True,True],.002,sole_clearance=[0,0],balance_ready=[True,True])
        self.assertTrue(m.transfer_load_latched)
        m.step([150.,150.],[True,True],.002,sole_clearance=[0,0],balance_ready=[True,True])
        self.assertEqual(m.confirmed,0)
        self.assertFalse(m.transfer_load_latched)

    def test_takeoff_lead_preserves_load_and_preview_gates(self):
        for ready in (False,True):
            m=ReferenceManager(np.r_[np.full(137,3),np.full(18,2),np.full(20,3)],30,
                               align_takeoff_frame=True,takeoff_lead=.1,ramp_duration=.25)
            m.tau=133/30;m.rate=.5;m.pending_takeoff=137
            previous=m.tau*30
            for _ in range(100):
                clock=m.step([5.,330.],[True,True],.002,sole_clearance=[0,0],
                             balance_ready=[ready,ready])
                self.assertLessEqual(clock.frame-previous,.061)
                previous=clock.frame
                if clock.mode==2:break
            if ready:
                self.assertEqual(clock.mode,2)
                self.assertGreaterEqual(clock.frame,134.-1e-6)
                self.assertLess(clock.frame,137.)
            else:self.assertEqual(clock.mode,3)
        with self.assertRaises(ValueError):
            ReferenceManager([3,1],30,align_takeoff_frame=True,takeoff_lead=.2)

    def test_confirmed_transfer_tolerates_load_redistribution_but_not_lost_support(self):
        m=ReferenceManager(np.r_[np.full(137,3),np.full(18,2),np.full(20,3)],30,
                           align_takeoff_frame=True,ramp_duration=.25)
        m.tau=130/30; m.rate=.5; m.pending_takeoff=137
        for _ in range(30):
            m.step([5.,330.],[True,True],.002,sole_clearance=[0,0],balance_ready=[True,True])
        self.assertTrue(m.transfer_load_latched)
        # Actual failure: the incoming load fell from 98% to 74% near takeoff.
        # This is still a loaded support, following a completed confirmation.
        while m.tau*30<136.8:
            m.step([5.,330.],[True,True],.002,sole_clearance=[0,0],balance_ready=[True,True])
        for _ in range(20):
            clock=m.step([89.,250.],[True,True],.002,sole_clearance=[0,0],balance_ready=[True,True])
            if clock.mode==2:break
        self.assertEqual(clock.mode,2)
        self.assertFalse(m.transfer_load_latched)
        m=ReferenceManager(np.r_[np.full(137,3),np.full(18,2)],30,align_takeoff_frame=True)
        m.tau=130/30;m.pending_takeoff=137;m.transfer_load_latched=True;m.confirmed=30
        m.step([150.,180.],[True,True],.002,sole_clearance=[0,0],balance_ready=[True,True])
        self.assertFalse(m.transfer_load_latched)
        self.assertEqual(m.confirmed,0)

    def test_preview_load_gate_matches_the_scheduler_force_requirement(self):
        manager = ReferenceManager(np.full(50,3),50)
        self.assertFalse(manager.transfer_load_ready([150.,150.],1))
        self.assertFalse(manager.transfer_load_ready([60.,0.],1))
        self.assertTrue(manager.transfer_load_ready([240.,60.],1))
        self.assertFalse(manager.transfer_load_ready([240.,60.],2))
        self.assertTrue(manager.transfer_load_ready([0.,300.],2))
        with self.assertRaises(ValueError):manager.transfer_load_ready([300.,0.],3)

    def test_early_balance_readiness_does_not_skip_source_takeoff_preparation(self):
        m=ReferenceManager(np.r_[np.full(20,3),np.full(30,1),np.full(30,3)],50,
                           align_takeoff_frame=True)
        previous=0.
        for _ in range(100):
            a=m.step([300,0],[True,True],.02,sole_clearance=[0,0],balance_ready=[True,False])
            self.assertGreaterEqual(a.frame,previous)
            self.assertLessEqual(a.frame-previous,1.001)
            previous=a.frame
            if a.mode==1:
                self.assertGreaterEqual(a.frame,20.)
                break
            self.assertLessEqual(a.frame,20.)
        else:
            self.fail('Aligned takeoff never released')

    def early_manager(self):
        m=ReferenceManager(np.r_[np.full(50,1),np.full(50,2)],50,early_touchdown=True)
        for _ in range(3):
            m.step([290,10],[True,True],.02,sole_clearance=[0,0],sole_max_height=[0,0])
        m.step([300,0],[True,False],.02,sole_clearance=[0,.03],sole_max_height=[0,.03])
        return m

    def test_flat_touchdown_enters_contact_before_load_confirmation(self):
        m=self.early_manager()
        previous=m.tau
        a=m.step([300,8],[True,False],.002,sole_clearance=[0,0],sole_max_height=[0,.0002])
        self.assertEqual(a.mode,3)
        self.assertEqual(a.phase,'LANDING')
        self.assertTrue(m.settling)
        self.assertLess(a.frame-previous*50,.11)
        for _ in range(200):
            a=m.step([300,8],[True,False],.002,sole_clearance=[0,0],sole_max_height=[0,0])
        self.assertTrue(m.settling)
        self.assertEqual(a.phase,'SETTLE')

    def test_loaded_heel_enters_only_with_an_observed_patch_model(self):
        for patch,force in ((True,60.),(False,60.),(True,0.)):
            m=self.early_manager()
            clock=m.step([270.,force],[True,False],.002,sole_clearance=[0.,0.],
                         sole_max_height=[0.,.04],touchdown_contact=[False,patch])
            self.assertEqual(clock.mode,3 if patch and force else 1)
            if patch and force:self.assertTrue(m.settling)

    def test_partial_or_unloaded_foot_does_not_trigger_early_full_contact(self):
        m=self.early_manager()
        a=m.step([300,8],[True,False],.002,sole_clearance=[0,0],sole_max_height=[0,.01])
        self.assertEqual(a.mode,1)
        a=m.step([300,0],[True,False],.002,sole_clearance=[0,0],sole_max_height=[0,0])
        self.assertEqual(a.mode,1)

    def test_light_flat_touchdown_at_50hz_still_requires_load_settling(self):
        m=self.early_manager()
        a=m.step([325,.2],[True,False],.02,sole_clearance=[0,0],sole_max_height=[0,.0006])
        self.assertEqual(a.mode,1)
        a=m.step([325,2.1],[True,False],.02,sole_clearance=[0,0],sole_max_height=[0,.0006])
        self.assertEqual(a.mode,3)
        self.assertTrue(m.settling)
        for _ in range(20):
            m.step([325,2.1],[True,False],.02,sole_clearance=[0,0],sole_max_height=[0,.0006])
        self.assertTrue(m.settling)

    def test_persistent_edge_contact_never_becomes_a_flat_sole(self):
        m = self.early_manager()
        # Sustained load used to bypass the geometric check after 60 ms.
        for _ in range(100):
            a = m.step([200,100], [True,True], .002,
                       sole_clearance=[0,0], sole_max_height=[0,.08])
            self.assertEqual(a.mode,1)
        a = m.step([200,100], [True,True], .002,
                   sole_clearance=[0,0], sole_max_height=[0,.001])
        self.assertEqual(a.mode,3)
        self.assertTrue(m.settling)

    def test_outgoing_contact_survives_transient_track_phase(self):
        manager=ReferenceManager(np.full(300,3),50)
        self.assertIsNone(manager.outgoing_foot(3))
        manager.landing_support=1
        for phase,support in [('SETTLE',1),('TRACK',3),('PREPARE_STEP',2)]:
            manager.phase=phase
            self.assertEqual(manager.outgoing_foot(support),0)
        manager.mode=2
        self.assertIsNone(manager.outgoing_foot(2))

    def test_continuous_transfer_confirms_incoming_load_without_reloading_old_foot(self):
        m = self.early_manager();m.settle_on_incoming=True;m.settle_duration=.06
        m.step([150,150],[True,True],.002,sole_clearance=[0,0],sole_max_height=[0,0])
        self.assertTrue(m.settling)
        for _ in range(30):
            m.step([0,300],[False,True],.002,sole_clearance=[0,0],sole_max_height=[0,0])
        self.assertFalse(m.settling)
        self.assertEqual(m.mode,3)  # Release still needs the separate balance/load gate.

    def test_continuous_transfer_resets_confirmation_when_new_load_disappears(self):
        m = self.early_manager();m.settle_on_incoming=True;m.settle_duration=.06
        m.step([150,150],[True,True],.002,sole_clearance=[0,0],sole_max_height=[0,0])
        for _ in range(20):
            m.step([0,300],[False,True],.002,sole_clearance=[0,0],sole_max_height=[0,0])
        m.step([300,0],[True,False],.002,sole_clearance=[0,0],sole_max_height=[0,0])
        self.assertEqual(m.settle_time,0.)
        self.assertTrue(m.settling)

    def test_plan_uses_motion_and_reports_ambiguous_frames(self):
        heights=np.zeros((30,2));speed=np.zeros_like(heights)
        speed[10:20,0]=.5
        modes,ambiguous=infer_contact_plan(heights,speed,50)
        self.assertEqual(modes[15],2)
        self.assertEqual(modes[25],3)
        self.assertFalse(ambiguous.any())
        speed[:]=1
        _,ambiguous=infer_contact_plan(heights,speed,50)
        self.assertTrue(ambiguous.all())

    def test_warmup_then_consistent_ramp(self):
        m=ReferenceManager(np.full(300,3),50)
        a=m.step([150,150],[True,True],.02,True)
        self.assertEqual(a.frame,0)
        last=0
        for _ in range(30):
            a=m.step([150,150],[True,True],.02)
            self.assertGreaterEqual(a.frame,last)
            self.assertAlmostEqual(a.previous_frame,last)
            self.assertGreaterEqual(a.next_frame,a.frame)
            last=a.frame
        self.assertLess(a.rate,1.)

    def test_unloading_gate_and_early_release_do_not_deadlock(self):
        modes=np.r_[np.full(30,3),np.full(100,2),np.full(50,3)]
        m=ReferenceManager(modes,50)
        for _ in range(100):
            a=m.step([150,150],[True,True],.02)
            if a.phase=='PREPARE_STEP':break
        self.assertEqual(a.support_mode,2)
        self.assertEqual(a.mode,3)
        for _ in range(3):a=m.step([10,290],[True,True],.02)
        self.assertEqual(a.mode,2)
        start=a.frame
        for _ in range(30):a=m.step([0,300],[False,True],.02)
        self.assertGreater(a.frame,start)
        self.assertEqual(a.mode,2)

    def test_clock_does_not_cross_unconfirmed_touchdown(self):
        m=ReferenceManager(np.r_[np.full(150,1),np.full(50,2)],50,timeout=10.)
        for _ in range(3):m.step([290,10],[True,True],.02)
        for _ in range(400):
            a=m.step([300,0],[True,False],.02)
            self.assertLess(a.frame,150)
        self.assertEqual(a.phase,'LANDING')
        self.assertEqual(a.mode,1)
        self.assertLess(a.rate,.001)

    def test_missing_support_fails_and_wait_is_bounded(self):
        m=ReferenceManager(np.full(100,3),50)
        with self.assertRaisesRegex(RuntimeError,'lost support'):
            m.step([0,0],[False,False],.02)
        m=ReferenceManager(np.full(100,2),50,timeout=.1)
        with self.assertRaisesRegex(RuntimeError,'timed out'):
            for _ in range(20):m.step([150,150],[True,True],.02)

    def test_touchdown_preserves_clock_continuity(self):
        m=ReferenceManager(np.r_[np.full(50,1),np.full(50,3)],50)
        for _ in range(3):a=m.step([290,10],[True,True],.02)
        m.step([300,0],[True,False],.02)
        old=m.tau;rate=m.rate
        a=m.step([200,100],[True,True],.02)
        self.assertEqual(a.mode,3)
        self.assertEqual(a.phase,'LANDING')
        self.assertGreaterEqual(a.frame,old*50)
        self.assertLessEqual(abs(a.rate-rate),.020001)

    def test_early_landing_does_not_repeat_finished_swing(self):
        m=ReferenceManager(np.r_[np.full(50,1),np.full(50,2),np.full(50,1)],50)
        for _ in range(3):m.step([290,10],[True,True],.02)
        for _ in range(30):m.step([300,0],[True,False],.02)
        a=m.step([200,100],[True,True],.02)
        self.assertEqual(a.mode,3)
        self.assertLess(a.frame,50)
        for _ in range(100):
            a=m.step([150,150],[True,True],.02)
            if a.phase=='PREPARE_STEP':break
        self.assertEqual(a.support_mode,2)
        self.assertEqual(m.pending_takeoff,50)

    def test_force_flicker_before_clearance_does_not_complete_swing(self):
        m=ReferenceManager(np.r_[np.full(50,1),np.full(50,2)],50)
        for _ in range(3):m.step([290,10],[True,True],.02,sole_clearance=[0,0])
        m.step([300,0],[True,False],.02,sole_clearance=[0,.003])
        a=m.step([180,120],[True,True],.02,sole_clearance=[0,0])
        self.assertEqual(a.mode,1)
        m.step([300,0],[True,False],.02,sole_clearance=[0,.02])
        for _ in range(3):a=m.step([180,120],[True,True],.02,sole_clearance=[0,0])
        self.assertEqual(a.mode,3)
        a=m.step([250,50],[True,True],.02,sole_clearance=[0,0])
        self.assertEqual(a.phase,'SETTLE')
        self.assertEqual(a.support_mode,1)
        for _ in range(20):a=m.step([250,50],[True,True],.02,sole_clearance=[0,0])
        self.assertFalse(m.settling)

    def test_light_touchdown_requires_geometry_and_does_not_skip_load_settling(self):
        m=ReferenceManager(np.r_[np.full(50,1),np.full(50,2)],50)
        for _ in range(3):m.step([290,10],[True,True],.02,sole_clearance=[0,0])
        for _ in range(60):m.step([300,0],[True,False],.02,sole_clearance=[0,.02])
        for _ in range(3):a=m.step([320,8],[True,False],.02,sole_clearance=[0,0])
        self.assertEqual(a.mode,3)
        for _ in range(30):a=m.step([320,8],[True,False],.02,sole_clearance=[0,0])
        self.assertEqual(a.phase,'SETTLE')
        self.assertTrue(m.settling)

    def test_unloaded_ground_contact_is_not_airborne_support(self):
        m=ReferenceManager(np.full(100,3),50)
        a=m.step([300,0],[True,False],.02,sole_clearance=[0,0])
        self.assertEqual(a.mode,3)
        with self.assertRaisesRegex(RuntimeError,'lost support'):
            m.step([300,0],[True,False],.02,sole_clearance=[0,.01])

    def test_load_alone_cannot_release_when_balance_gate_is_false(self):
        m=ReferenceManager(np.full(100,1),50)
        for _ in range(20):a=m.step([300,0],[True,True],.02,balance_ready=[False,False])
        self.assertEqual(a.mode,3)
        for _ in range(3):a=m.step([300,0],[True,True],.02,balance_ready=[True,False])
        self.assertEqual(a.mode,1)

if __name__=='__main__':unittest.main()
