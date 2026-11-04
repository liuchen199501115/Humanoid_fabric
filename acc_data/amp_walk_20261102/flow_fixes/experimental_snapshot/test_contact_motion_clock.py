import unittest
import numpy as np
from fabrics_sim.reference.contact_clock import ContactMotionClock


class ContactClockTests(unittest.TestCase):
    def events(self):
        return [dict(end_frame=32,support=1),dict(end_frame=79,support=0)]

    def test_brakes_without_crossing_unconfirmed_landing(self):
        c=ContactMotionClock(self.events(),200.)
        previous=0.
        for _ in range(500):
            frame,v,a=c.update([0,300],.002)
            self.assertGreaterEqual(frame,previous)
            self.assertLessEqual(frame,32.+1e-8)
            self.assertLessEqual(abs(a),800.+1e-6)
            self.assertGreaterEqual(v,0.)
            previous=frame
        self.assertAlmostEqual(c.frame,32.,places=6)
        self.assertAlmostEqual(c.rate,0.)

    def test_resumes_only_on_confirmed_incoming_ground_load(self):
        c=ContactMotionClock(self.events(),200.)
        for _ in range(250):c.update([0,300],.002)
        c.update([100,200],.002)
        c.update([0,300],.002)
        self.assertLessEqual(c.frame,32.+1e-8)
        for _ in range(30):c.update([300,0],.002)
        self.assertGreater(c.frame,32.)
        self.assertEqual(c.index,1)
        self.assertLessEqual(c.rate,c.nominal_rate)

    def test_phase_derivatives_match_integrated_position(self):
        c=ContactMotionClock(self.events(),200.)
        for _ in range(100):
            p,v=c.frame,c.rate
            frame,rate,acc=c.update([0,300],.002)
            self.assertAlmostEqual(rate,v+acc*.002,places=6)
            self.assertAlmostEqual(frame,p+v*.002+.5*acc*.002**2,places=6)

    def test_existing_stance_load_cannot_confirm_a_future_landing(self):
        c=ContactMotionClock(self.events(),200.)
        for _ in range(500):c.update([150,150],.002)
        self.assertFalse(c.ready)
        self.assertEqual(c.index,0)
        self.assertLessEqual(c.frame,32.+1e-8)

    def test_rejects_initial_boundary_too_near_to_brake(self):
        with self.assertRaises(ValueError):ContactMotionClock([dict(end_frame=2,support=1)],200.)


if __name__=='__main__':unittest.main()
