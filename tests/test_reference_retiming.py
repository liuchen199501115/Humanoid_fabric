import unittest
from types import SimpleNamespace
import numpy as np
import torch
from fabrics_sim.reference.retiming import LandingRetimedReference
from fabrics_sim.reference.whole_body import WholeBodyReference


class RetimingTests(unittest.TestCase):
    def source(self):
        def sample(t):
            q=torch.zeros(1,36,dtype=torch.float64);q[:,6]=1.;q[:,0]=t+t*t
            v=torch.zeros(1,35,dtype=torch.float64);v[:,0]=1+2*t
            a=v*0.;a[:,0]=2.
            return WholeBodyReference(q,v,a)
        return SimpleNamespace(duration=2.,sample=sample,q=torch.zeros(1,36),
                               progress=lambda t:(t/2,.5,0.,None))

    def test_initial_state_and_landing_pose_preserved(self):
        source=self.source();path=LandingRetimedReference(source,.3)
        for key in ('q','v','a'):
            torch.testing.assert_close(getattr(path.sample(0),key),getattr(source.sample(0),key))
        torch.testing.assert_close(path.sample(2).q,source.sample(2).q)
        torch.testing.assert_close(path.sample(2).v,.3*source.sample(2).v)
        torch.testing.assert_close(path.sample(2).a,.09*source.sample(2).a)
        replacement=torch.ones(1,36);path.q=replacement
        self.assertIs(source.q,replacement)

    def test_monotone_clock_and_chain_rule(self):
        for rate in (0.,.3,1.):
            path=LandingRetimedReference(self.source(),rate)
            clock=np.array([path.clock(t) for t in np.linspace(0,2,1001)])
            self.assertTrue(np.all(np.diff(clock[:,0])>0))
            self.assertTrue(np.all(clock[:,1]>=-1e-12))
            for t in (.1,.8,1.8):
                h=1e-5;lo,mid,hi=[path.sample(x) for x in (t-h,t,t+h)]
                torch.testing.assert_close((hi.q[:,:3]-lo.q[:,:3])/(2*h),mid.v[:,:3],atol=1e-8,rtol=1e-8)
                torch.testing.assert_close((hi.v-lo.v)/(2*h),mid.a,atol=1e-8,rtol=1e-8)
                self.assertAlmostEqual(path.progress(t)[0],path.clock(t)[0]/2)

    def test_invalid_parameters_and_time(self):
        for rate in (-.1,1.1,float('nan')):
            with self.assertRaises(ValueError):LandingRetimedReference(self.source(),rate)
        path=LandingRetimedReference(self.source(),.5)
        for t in (-1.,3.,float('nan')):
            with self.assertRaises(ValueError):path.sample(t)
