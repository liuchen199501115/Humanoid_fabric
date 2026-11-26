import unittest
import torch
from g1_root_xy_tube import root_xy_tube


class RootXYTubeTest(unittest.TestCase):
    def test_matches_independently_differentiated_moving_error(self):
        q=torch.zeros(1,36,dtype=torch.float64);rq=q.clone();v=torch.zeros(1,35,dtype=q.dtype);rv=v.clone();a=v.clone();ra=v.clone()
        q[0,:2]=torch.tensor([.10,-.12]);rq[0,:2]=torch.tensor([-.03,.02])
        v[0,:2]=torch.tensor([.2,-.1]);rv[0,:2]=torch.tensor([.05,.03])
        a[0,:2]=torch.tensor([2.,-1.]);ra[0,:2]=torch.tensor([.2,.4])
        C,d,diag=root_xy_tube(q,v,rq,rv,ra,radius=.18,bandwidth=10.)
        def h(t):
            error=q[:,:2]+v[:,:2]*t+.5*a[:,:2]*t*t-rq[:,:2]-rv[:,:2]*t-.5*ra[:,:2]*t*t
            return .18**2-error.square().sum()
        dt=1e-4;hd=(h(dt)-h(-dt))/(2*dt);hdd=(h(dt)-2*h(0)+h(-dt))/dt**2
        self.assertAlmostEqual(float(d-(C@a[...,None]).squeeze(-1)),float(hdd+20*hd+100*h(0)),places=6)
        self.assertAlmostEqual(float(diag[0,2]),float(hd+10*h(0)),places=6)
        q[:,2]=100.;v[:,2]=-100.;a[:,2]=100.
        C2,d2,_=root_xy_tube(q,v,rq,rv,ra)
        torch.testing.assert_close(C,C2,atol=0,rtol=0);torch.testing.assert_close(d,d2,atol=0,rtol=0)

    def test_reject_invalid_parameters(self):
        q=torch.zeros(1,36);v=torch.zeros(1,35)
        for radius in [0.,.21,float('nan')]:
            with self.assertRaises(ValueError):root_xy_tube(q,v,q,v,v,radius=radius)


if __name__=='__main__':unittest.main()
