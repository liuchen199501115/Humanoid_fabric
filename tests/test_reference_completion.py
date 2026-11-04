import unittest
from fabrics_sim.reference.completion import MotionCompletion


class CompletionTests(unittest.TestCase):
    def test_frozen_mid_clip_is_not_complete(self):
        check=MotionCompletion(99)
        for _ in range(100):
            self.assertFalse(check.update(98,3,False,[160,160],[0,0,0],.02))

    def test_last_frame_requires_stable_landing_and_load(self):
        check=MotionCompletion(99)
        for mode,settling,forces,velocity in ((1,False,[320,0],[0,0,0]),
                (3,True,[160,160],[0,0,0]),(3,False,[310,10],[0,0,0]),
                (3,False,[160,160],[.1,0,0])):
            for _ in range(30):
                self.assertFalse(check.update(99,mode,settling,forces,velocity,.02))
        for _ in range(24):
            self.assertFalse(check.update(99,3,False,[160,160],[0,0,0],.02))
        self.assertTrue(check.update(99,3,False,[160,160],[0,0,0],.02))

    def test_disturbance_restarts_confirmation(self):
        check=MotionCompletion(99)
        for _ in range(24):check.update(99,3,False,[160,160],[0,0,0],.02)
        check.update(99,3,False,[160,160],[.1,0,0],.02)
        self.assertFalse(check.update(99,3,False,[160,160],[0,0,0],.02))


if __name__=='__main__':unittest.main()
