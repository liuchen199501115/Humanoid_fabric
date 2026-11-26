import unittest
from types import SimpleNamespace
import torch
from pelvis_orientation_allocation import replace_torso_weight

class AllocationTests(unittest.TestCase):
    def test_weight_is_transferred_without_changing_references(self):
        for blend in [0.,.25,1.]:
            M=torch.diag(torch.tensor([4.,4.,2.]))[None]
            body=SimpleNamespace(name='body_orientation',metric=M.clone(),desired=torch.tensor([[1.,2.,3.]]))
            pelvis=SimpleNamespace(name='source_pelvis_orientation',metric=10*M.clone(),desired=torch.tensor([[-1.,-2.,-3.]]))
            saved=[x.desired.clone() for x in [body,pelvis]]
            replace_torso_weight([body,pelvis],blend)
            torch.testing.assert_close(body.metric+pelvis.metric,M)
            torch.testing.assert_close(pelvis.metric,M*blend)
            for x,v in zip([body,pelvis],saved):torch.testing.assert_close(x.desired,v)
    def test_missing_tasks_and_invalid_blend_fail(self):
        for blend in [-.1,1.1,float('nan'),float('inf'),.5]:
            with self.subTest(blend=blend),self.assertRaises(ValueError):replace_torso_weight([],blend)

    def test_scaled_pelvis_preserves_off_diagonal_metric_and_torso_fade(self):
        M=torch.tensor([[[4.,1.,0.],[1.,3.,0.],[0.,0.,2.]]])
        body=SimpleNamespace(name='body_orientation',metric=M.clone())
        pelvis=SimpleNamespace(name='source_pelvis_orientation',metric=M.clone())
        replace_torso_weight([body,pelvis],.25,pelvis_scale=4.)
        torch.testing.assert_close(body.metric,.75*M)
        torch.testing.assert_close(pelvis.metric,M)
        for scale in [0.,-1.,float('nan'),float('inf')]:
            with self.assertRaises(ValueError):
                replace_torso_weight([body,pelvis],.5,pelvis_scale=scale)
