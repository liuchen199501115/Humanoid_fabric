import unittest
from types import SimpleNamespace
import torch
from transition_swing_task import activate_transition_swing
class TransitionSwingTests(unittest.TestCase):
    def requests(self):
        return [SimpleNamespace(name='feet_pose',active=torch.zeros((1,12),dtype=torch.bool)),SimpleNamespace(name='joint_posture',active=torch.ones((1,29),dtype=torch.bool))]
    def test_only_outgoing_rows_with_actual_new_support(self):
        for intended in (1,2):
            r=self.requests();forces=torch.tensor([200.,200.]);rows=activate_transition_swing(r,3,intended,forces);outgoing=2-intended
            self.assertEqual(rows,tuple(range(6*outgoing,6*outgoing+6)))
            expected=torch.zeros((1,12),dtype=torch.bool);expected[:,6*outgoing:6*outgoing+6]=True
            self.assertTrue(torch.equal(r[0].active,expected));self.assertTrue(r[1].active.all());self.assertTrue(torch.equal(forces,torch.tensor([200.,200.])))
    def test_no_activation_without_bearing_incoming_foot(self):
        r=self.requests();self.assertIsNone(activate_transition_swing(r,3,1,torch.tensor([99.,300.])));self.assertFalse(r[0].active.any())
    def test_no_change_outside_double_support_transfer(self):
        for mode,intended in [(0,1),(1,1),(2,2),(3,3),(3,0)]:
            r=self.requests();self.assertIsNone(activate_transition_swing(r,mode,intended,torch.tensor([300.,300.])));self.assertFalse(r[0].active.any())
if __name__=='__main__':unittest.main()
