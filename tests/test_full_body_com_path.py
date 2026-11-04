import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from fabrics_sim.reference.full_body_com_path import FullBodyComPath
from fabrics_sim.reference.whole_body import WholeBodyReference
from fabrics_sim.reference.source_step import SourceStepCandidate


class FullBodyComPathTests(unittest.TestCase):
    def setUp(self):
        self.source = torch.zeros(5,36,dtype=torch.float64)
        self.source[:,6] = 1.
        self.source[:,0] = torch.linspace(0.,.2,5,dtype=torch.float64)
        self.source[:,2] = .7
        self.source[:,7] = torch.linspace(0.,.3,5,dtype=torch.float64)
        def evaluate(q,v,*args):
            jac = q.new_zeros(1,3,35)
            jac[:,:,:3] = torch.eye(3,dtype=q.dtype)
            jac[:,0,6] = .2
            com = q[:,:3].clone();com[:,0] += .2*q[:,7]
            return dict(com=com,com_jac=jac,com_jac_bias=q.new_zeros(1,3)),None,None
        self.fabric = SimpleNamespace(dynamics_tasks=SimpleNamespace(
            model=SimpleNamespace(evaluate=evaluate),config={'difference_step':1e-4}),
            integrate_root_position=None)
        q = self.source[:1].clone();q[:,0] -= .1
        v = q.new_zeros(1,35);v[:,0] = .03
        a = q.new_zeros(1,35);a[:,0] = -.02
        self.initial = WholeBodyReference(q,v,a)

    def test_boundary_rates_and_source_preservation(self):
        original = self.source.clone()
        path = FullBodyComPath(self.fabric,self.initial,self.source,1.5)
        start = path.sample(0.)
        end = path.sample(1.5)
        np.testing.assert_allclose(start[0],[-.1,0.,.7])
        np.testing.assert_allclose(start[1],[.03,0.,0.])
        np.testing.assert_allclose(start[2],[-.02,0.,0.])
        np.testing.assert_allclose(end[0],[.26,0.,.7])
        np.testing.assert_allclose(end[1],[.26/1.5,0.,0.],atol=1e-12)
        np.testing.assert_allclose(end[2],0.,atol=1e-12)
        torch.testing.assert_close(self.source,original,atol=0.,rtol=0.)

    def test_analytic_derivatives(self):
        path = FullBodyComPath(self.fabric,self.initial,self.source,1.5)
        for time in (.2,.6,1.2):
            eps=1e-5
            lo,mid,hi=[path.sample(t) for t in (time-eps,time,time+eps)]
            np.testing.assert_allclose((hi[0]-lo[0])/(2*eps),mid[1],atol=1e-8)
            np.testing.assert_allclose((hi[1]-lo[1])/(2*eps),mid[2],atol=1e-8)
        for time in (-.1,1.6,float('nan')):
            with self.assertRaises(ValueError):path.sample(time)

    def test_source_clock_never_samples_a_dynamics_preview(self):
        plan=object.__new__(SourceStepCandidate)
        plan.duration=1.5;plan.com_preview='full_body_source'
        plan.phase_preview=SimpleNamespace(sample=lambda _:self.fail('preview used for clock'))
        self.assertEqual(plan.progress(0.)[3],(0.,0.,0.))
        self.assertEqual(plan.progress(1.5)[3],(1.,0.,0.))


class FullBodySourceFeetTests(FullBodyComPathTests):
    def test_source_swing_endpoint_anchor_and_derivatives(self):
        from fabrics_sim.reference.full_body_com_path import FullBodySourceFeet
        original_evaluate = self.fabric.dynamics_tasks.model.evaluate
        def evaluate(q,v,*args):
            state,_,_ = original_evaluate(q,v,*args)
            poses = q[:,:7,None].transpose(1,2).repeat(1,2,1)
            poses[:,0,1] += .1
            poses[:,1,1] -= .1
            jac = q.new_zeros(1,2,3,35)
            jac[:,:,:,:3] = torch.eye(3,dtype=q.dtype)
            state.update(transforms=poses,origin_jac=jac,angular_jac=torch.zeros_like(jac),
                         origin_jac_bias=q.new_zeros(1,2,3),angular_jac_bias=q.new_zeros(1,2,3))
            return state,None,None
        self.fabric.dynamics_tasks.model.evaluate = evaluate
        baseline=FullBodySourceFeet(self.fabric,self.initial,self.source,1.5,0,[0,1])
        path=FullBodySourceFeet(self.fabric,self.initial,self.source,1.5,0,[0,1],clearance_bump=.04)
        for t in (0.,1.5):
            for actual,expected in zip(path.sample(t),baseline.sample(t)):
                np.testing.assert_array_equal(actual,expected)
        self.assertAlmostEqual(path.sample(.75)[0][1,2]-baseline.sample(.75)[0][1,2],.04)
        with self.assertRaises(ValueError):
            FullBodySourceFeet(self.fabric,self.initial,self.source,1.5,0,[0,1],clearance_bump=-.01)
        first=path.sample(0.);end=path.sample(1.5)
        # A stance reference must not erase the measured entry twist.
        np.testing.assert_allclose(first[1][0,:3],[.03,0.,0.])
        np.testing.assert_allclose(first[2][0,:3],[-.02,0.,0.])
        t=.05; eps=1e-6
        lo,mid,hi=[path.sample(x) for x in (t-eps,t,t+eps)]
        np.testing.assert_allclose((hi[0][0,:3]-lo[0][0,:3])/(2*eps),mid[1][0,:3],atol=1e-8)
        np.testing.assert_allclose((hi[1][0,:3]-lo[1][0,:3])/(2*eps),mid[2][0,:3],atol=1e-8)
        self.assertLess(np.linalg.norm(mid[0][0,:3]-first[0][0,:3]),.002)
        np.testing.assert_allclose(first[0][1,:3],[-.1,-.1,.7])
        np.testing.assert_allclose(first[1][1,:3],[.03,0.,0.])
        np.testing.assert_allclose(first[2][1,:3],[-.02,0.,0.])
        np.testing.assert_allclose(end[0][1,:3],[.2,-.1,.7])
        np.testing.assert_allclose(end[1][1,:3],[.2/1.5,0.,0.])
        for t in (.2,.6,1.2):
            eps=1e-5
            lo,mid,hi=[path.sample(x) for x in (t-eps,t,t+eps)]
            np.testing.assert_array_equal(mid[0][0],first[0][0])
            np.testing.assert_array_equal(mid[1][0],0.)
            np.testing.assert_allclose((hi[0][1,:3]-lo[0][1,:3])/(2*eps),mid[1][1,:3],atol=1e-8)
            np.testing.assert_allclose((hi[1][1,:3]-lo[1][1,:3])/(2*eps),mid[2][1,:3],atol=1e-8)
        self.initial.v[:,0] = 1.
        fast = FullBodySourceFeet(self.fabric,self.initial,self.source,1.5,0,[0,1])
        with self.assertRaisesRegex(RuntimeError,'2 mm'):
            fast.sample(.05)


class OffsetComPathTests(FullBodyComPathTests):
    def test_offset_is_continuous_and_preserves_endpoint_rates(self):
        from fabrics_sim.reference.full_body_com_path import OffsetComPath
        source=FullBodyComPath(self.fabric,self.initial,self.source,1.5)
        offset=np.array([.02,-.03,-.04])
        path=OffsetComPath(source,offset)
        for actual,expected in zip(path.sample(0.),source.sample(0.)):
            np.testing.assert_allclose(actual,expected)
        np.testing.assert_allclose(path.sample(1.5)[0],source.sample(1.5)[0]+offset)
        for i in (1,2):np.testing.assert_allclose(path.sample(1.5)[i],source.sample(1.5)[i])
        eps=1e-5;t=.6
        lo,mid,hi=[path.sample(x) for x in (t-eps,t,t+eps)]
        np.testing.assert_allclose((hi[0]-lo[0])/(2*eps),mid[1],atol=1e-8)
        np.testing.assert_allclose((hi[1]-lo[1])/(2*eps),mid[2],atol=1e-8)
        offset[:]=100.
        np.testing.assert_allclose(path.offset,[.02,-.03,-.04])
        with self.assertRaises(ValueError):OffsetComPath(source,[0.,float('nan'),0.])


class EndpointComPathTests(FullBodyComPathTests):
    def test_terminal_state_and_initial_continuity(self):
        from fabrics_sim.reference.full_body_com_path import EndpointComPath
        base=FullBodyComPath(self.fabric,self.initial,self.source,1.5)
        targets=np.array([[.02,.04,-.06],[.1,-.2,0.],[1.,-2.,.3]])
        path=EndpointComPath(base,*targets)
        for actual,expected in zip(path.sample(0.),base.sample(0.)):
            np.testing.assert_allclose(actual,expected)
        for actual,expected,offset in zip(path.sample(1.5),base.sample(1.5),targets):
            np.testing.assert_allclose(actual,expected+offset,atol=1e-12)
        t=.7;h=1e-5
        lo,mid,hi=[path.sample(x) for x in (t-h,t,t+h)]
        np.testing.assert_allclose((hi[0]-lo[0])/(2*h),mid[1],atol=1e-8)
        np.testing.assert_allclose((hi[1]-lo[1])/(2*h),mid[2],atol=1e-8)
        with self.assertRaises(ValueError):EndpointComPath(base,[0.,0.],[0.,0.,0.],[0.,0.,0.])


class InteriorComPathTests(FullBodyComPathTests):
    def test_interior_derivatives_and_unchanged_boundaries(self):
        from fabrics_sim.reference.full_body_com_path import InteriorComPath
        base=FullBodyComPath(self.fabric,self.initial,self.source,1.5)
        path=InteriorComPath(base,[[.02,-.03,.01],[-.01,.02,.005]])
        for t in (0.,1.5):
            for actual,expected in zip(path.sample(t),base.sample(t)):
                np.testing.assert_array_equal(actual,expected)
        for t in (.2,.7,1.3):
            h=1e-5
            lo,mid,hi=[path.sample(x) for x in (t-h,t,t+h)]
            np.testing.assert_allclose((hi[0]-lo[0])/(2*h),mid[1],atol=1e-8)
            np.testing.assert_allclose((hi[1]-lo[1])/(2*h),mid[2],atol=1e-8)
        for t in (1e-6,1.5-1e-6):
            self.assertLess(np.max(np.abs(path.sample(t)[2]-base.sample(t)[2])),1e-4)
        with self.assertRaises(ValueError):InteriorComPath(base,[[0.,0.],[0.,0.]])


class WindowedComPathTests(FullBodyComPathTests):
    def test_local_offset_is_c2_and_leaves_the_rest_unchanged(self):
        from fabrics_sim.reference.full_body_com_path import WindowedComPath
        base=FullBodyComPath(self.fabric,self.initial,self.source,1.5)
        path=WindowedComPath(base,[.01,-.02,0.],.1,.8)
        for t in (0.,.1,.8,1.,1.5):
            for actual,expected in zip(path.sample(t),base.sample(t)):
                np.testing.assert_array_equal(actual,expected)
        np.testing.assert_allclose(path.sample(.45)[0]-base.sample(.45)[0],[.01,-.02,0.],atol=1e-12)
        for t in (.2,.45,.7):
            h=1e-6
            lo,mid,hi=[path.sample(x) for x in (t-h,t,t+h)]
            np.testing.assert_allclose((hi[0]-lo[0])/(2*h),mid[1],atol=1e-8)
            np.testing.assert_allclose((hi[1]-lo[1])/(2*h),mid[2],atol=1e-8)
        for t in (.1+1e-8,.8-1e-8):
            self.assertLess(np.max(np.abs(path.sample(t)[2]-base.sample(t)[2])),1e-5)
        for start,end in ((-.1,.8),(.8,.1),(0.,2.),(0.,float('nan'))):
            with self.assertRaises(ValueError):WindowedComPath(base,[0.,0.,0.],start,end)
