import unittest
import torch
from fabrics_sim.reference.arm_clearance import arm_clearance_reference


class ArmClearanceTests(unittest.TestCase):
    def fixture(self):
        names=[str(i) for i in range(29)]
        names[16]='left_shoulder_roll_joint';names[23]='right_shoulder_roll_joint'
        source=torch.zeros(20,36);source[:,6]=1.
        return source,names,-torch.ones(29),torch.ones(29)

    def test_changes_only_two_arm_coordinates_and_keeps_original(self):
        source,names,lo,hi=self.fixture();original=source.clone()
        corrected=arm_clearance_reference(source,names,.2,lo,hi)
        delta=corrected-original
        self.assertTrue(torch.allclose(delta[:,23],torch.full((20,),.2)))
        self.assertTrue(torch.allclose(delta[:,30],torch.full((20,),-.2)))
        delta[:,[23,30]]=0.;torch.testing.assert_close(delta,torch.zeros_like(delta))
        torch.testing.assert_close(source,original)

    def test_rejects_whole_clip_limit_violation_without_clipping(self):
        source,names,lo,hi=self.fixture();source[-1,23]=.9
        with self.assertRaises(ValueError):arm_clearance_reference(source,names,.2,lo,hi)
        for offset in (-.1,float('nan')):
            with self.assertRaises(ValueError):arm_clearance_reference(source,names,offset,lo,hi)
