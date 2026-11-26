from types import SimpleNamespace
import unittest
import torch
from g1_root_position_reference import RootPositionReference


class RootPositionReferenceTest(unittest.TestCase):
    def test_source_stays_independent_of_prepared_control_and_clock_derivatives(self):
        q=torch.zeros(8,36,dtype=torch.float64);q[:,6]=1.
        q[:,0]=torch.arange(8,dtype=q.dtype)*.01
        sampler=RootPositionReference(q)
        control=SimpleNamespace(q=q[0:1]+10,v=torch.full((1,35),100.,dtype=q.dtype),
                                a=torch.full((1,35),-200.,dtype=q.dtype))
        pose,v,a=sampler.sample(3.5,240.,60.,mode='source',control=control)
        self.assertAlmostEqual(pose[0,0].item(),.035)
        self.assertAlmostEqual(v[0,0].item(),2.4)
        self.assertAlmostEqual(a[0,0].item(),.6)
        control.q+=20;control.v*=2;control.a*=3
        for old,new in zip((pose,v,a),sampler.sample(3.5,240.,60.,mode='source',control=control)):
            torch.testing.assert_close(old,new,rtol=0,atol=0)
        selected=sampler.sample(3.5,240.,60.,mode='control',control=control)
        for actual,expected in zip(selected,(control.q,control.v,control.a)):
            self.assertIs(actual,expected)

    def test_missing_control_or_unknown_mode_is_rejected(self):
        q=torch.zeros(8,36);q[:,6]=1.;sampler=RootPositionReference(q)
        for mode in ['control','invalid']:
            with self.assertRaises(ValueError):sampler.sample(3.,200.,0.,mode=mode)


if __name__=='__main__':unittest.main()
