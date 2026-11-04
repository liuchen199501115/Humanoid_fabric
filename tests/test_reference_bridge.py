import unittest
import numpy as np
from fabrics_sim.reference.bridge import ReferenceBridge, RotationReferenceBridge
from scipy.spatial.transform import Rotation


class ReferenceBridgeTest(unittest.TestCase):
    def test_rotation_bridge_matches_measured_start_and_moving_source_endpoint(self):
        q0=Rotation.from_euler('xyz',[.2,-.3,.4]).as_quat()
        source0=Rotation.from_euler('xyz',[-.1,.4,.2])
        source_w=np.array([.3,-.2,.4]);measured_w=np.array([-.1,.5,.3])
        bridge=RotationReferenceBridge(q0,measured_w,source0.as_quat(),source_w,np.zeros(3),.6)
        def sample(t):
            source=(Rotation.from_rotvec(source_w*t)*source0).as_quat()
            return bridge.sample(t,source,source_w,np.zeros(3))
        q,w,a=sample(0.)
        self.assertLess((Rotation.from_quat(q)*Rotation.from_quat(q0).inv()).magnitude(),1e-10)
        np.testing.assert_allclose(w,measured_w,atol=1e-10)
        np.testing.assert_allclose(a,0.,atol=1e-10)
        for t in (.05,.2,.5):
            eps=1e-5
            qm,wm,_=sample(t-eps);qp,wp,_=sample(t+eps)
            _,w,a=sample(t)
            np.testing.assert_allclose((Rotation.from_quat(qp)*Rotation.from_quat(qm).inv()).as_rotvec()/(2*eps),w,atol=1e-7)
            np.testing.assert_allclose((wp-wm)/(2*eps),a,atol=1e-6)
        q,w,a=sample(.6)
        self.assertLess((Rotation.from_quat(q)*(Rotation.from_rotvec(source_w*.6)*source0).inv()).magnitude(),1e-10)
        np.testing.assert_allclose(w,source_w,atol=1e-10)
        np.testing.assert_allclose(a,0.,atol=1e-10)

    def test_moving_source_joins_previous_reference_and_preserves_endpoint(self):
        p, v, a = np.array([.1, -.2]), np.array([.3, .4]), np.array([-.5, .6])
        bridge = ReferenceBridge(p, v, a, .8)
        for actual, expected in zip(bridge.sample(0.), (p, v, a)):
            np.testing.assert_allclose(actual, expected)
        for t in (.8, 1.):
            for value in bridge.sample(t):
                np.testing.assert_array_equal(value, np.zeros(2))
        for value in bridge.sample(.8-1e-8):
            self.assertLess(np.linalg.norm(value), 1e-5)

    def test_derivatives_match_position_curve(self):
        bridge = ReferenceBridge([.1, -.1], [.2, .5], [.3, -.4], 1.2)
        t, eps = .4, 1e-5
        p, v, a = bridge.sample(t)
        pm, vm, _ = bridge.sample(t-eps)
        pp, vp, _ = bridge.sample(t+eps)
        np.testing.assert_allclose((pp-pm)/(2*eps), v, atol=1e-8)
        np.testing.assert_allclose((vp-vm)/(2*eps), a, atol=1e-8)

    def test_invalid_inputs(self):
        for duration in (0., -1., float('nan')):
            with self.assertRaises(ValueError):
                ReferenceBridge([0.], [0.], [0.], duration)
        with self.assertRaises(ValueError):
            ReferenceBridge([0.], [0., 1.], [0.], 1.)


class WholeBodyReferenceBridgeTest(unittest.TestCase):
    def test_complete_boundary_and_moving_endpoint(self):
        import torch
        from fabrics_sim.reference.bridge import WholeBodyReferenceBridge
        from fabrics_sim.reference.whole_body import WholeBodyReference
        def source(t):
            q=torch.zeros(1,36,dtype=torch.float64)
            q[:,6]=1.;q[:,0]=.2*t+.015*t*t;q[:,7:]=.1*t+.01*t*t
            q[:,3:7]=torch.tensor(Rotation.from_rotvec(np.array([.1,-.2,.3])*t).as_quat())
            v=torch.zeros(1,35,dtype=torch.float64);a=torch.zeros_like(v)
            v[:,0]=.2+.03*t;v[:,3:6]=torch.tensor([.1,-.2,.3],dtype=torch.float64);v[:,6:]=.1+.02*t
            a[:,0]=.03;a[:,6:]=.02
            return WholeBodyReference(q,v,a)
        q=source(0.).q.clone();q[:,0]=-.1;q[:,7:]=.05
        q[:,3:7]=torch.tensor(Rotation.from_euler('xyz',[.2,-.1,.15]).as_quat())
        initial=WholeBodyReference(q,torch.full((1,35),.03,dtype=torch.float64),
                                   torch.full((1,35),-.02,dtype=torch.float64))
        bridge=WholeBodyReferenceBridge(initial,source(0.),.8)
        for name in ('q','v','a'):
            torch.testing.assert_close(getattr(bridge.sample(0.,source(0.)),name),getattr(initial,name),atol=1e-12,rtol=0.)
            torch.testing.assert_close(getattr(bridge.sample(.8,source(.8)),name),getattr(source(.8),name),atol=0.,rtol=0.)
        h=1e-5;t=.3
        lo,mid,hi=[bridge.sample(x,source(x)) for x in (t-h,t,t+h)]
        cols=[0,1,2]+list(range(7,36));vcols=[0,1,2]+list(range(6,35))
        torch.testing.assert_close((hi.q[:,cols]-lo.q[:,cols])/(2*h),mid.v[:,vcols],atol=1e-8,rtol=0.)
        torch.testing.assert_close((hi.v-lo.v)/(2*h),mid.a,atol=1e-8,rtol=0.)
        angular=(Rotation.from_quat(hi.q[0,3:7])*Rotation.from_quat(lo.q[0,3:7]).inv()).as_rotvec()/(2*h)
        np.testing.assert_allclose(angular,mid.v[0,3:6],atol=1e-8)
