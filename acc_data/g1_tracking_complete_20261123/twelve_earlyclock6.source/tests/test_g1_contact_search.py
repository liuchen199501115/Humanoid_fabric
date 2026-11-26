import sys
import unittest
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'examples'))
from g1_contact_search import contact_search_target


class ContactSearchTest(unittest.TestCase):
    def inputs(self, batch=1):
        return [torch.zeros(batch,12), torch.zeros(batch,2,3), torch.zeros(batch,2,3),
                torch.zeros(batch,2), torch.full((batch,),2,dtype=torch.int64),
                torch.ones(batch,12,dtype=torch.bool)]

    def test_only_intended_unloaded_near_goal_foot_descends(self):
        values=self.inputs(); values[1][0,1]=torch.tensor([.01,-.02,.004])
        out, selected=contact_search_target(*values,acceleration=2.,blend=.5)
        self.assertEqual(selected.tolist(),[[False,True]])
        expected=torch.zeros(1,12);expected[0,8]=-1.
        torch.testing.assert_close(out,expected)
        self.assertFalse(bool(values[0].any()))

    def test_contact_or_inactive_task_immediately_ends_search(self):
        for normal,active in [(20.,True),(100.,True),(0.,False)]:
            values=self.inputs();values[3][0,1]=normal;values[5][0,6:]=active
            out,selected=contact_search_target(*values,acceleration=2.)
            self.assertFalse(bool(selected.any()));self.assertFalse(bool(out.any()))

    def test_no_remote_or_excessive_descent_and_no_planned_flight(self):
        for gap,mode in [([.041,0.,0.],2),([0.,0.,.021],2),([0.,0.,-.006],2),([0.,0.,0.],1)]:
            values=self.inputs();values[1][0,1]=torch.tensor(gap);values[4].fill_(mode)
            values[5][0,:6]=False
            out,selected=contact_search_target(*values,acceleration=2.)
            self.assertFalse(bool(selected.any()));self.assertFalse(bool(out.any()))

    def test_zero_blend_and_invalid_inputs(self):
        values=self.inputs();out,selected=contact_search_target(*values,acceleration=2.,blend=0.)
        torch.testing.assert_close(out,values[0]);self.assertFalse(bool(selected.any()))
        for value in [-1.,0.,11.,float('nan')]:
            with self.assertRaises(ValueError):contact_search_target(*values,acceleration=value)
        values[4].fill_(4)
        with self.assertRaises(ValueError):contact_search_target(*values,acceleration=1.)


if __name__=='__main__':unittest.main()
