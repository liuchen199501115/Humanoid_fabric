import unittest
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from fabrics_sim.solver.landing_approach import landing_approach_constraints


class LandingApproachTests(unittest.TestCase):
    def test_sphere_height_derivatives_use_center_and_world_radius(self):
        p=torch.tensor([[0.,0.,.09]],dtype=torch.float64)
        R=p.new_tensor(Rotation.from_euler('xyz',[.2,-.3,.1]).as_matrix())[None]
        J=p.new_zeros(1,6,35);J[0,:,:6]=torch.eye(6,dtype=p.dtype)
        v=p.new_zeros(1,35);v[0,:6]=p.new_tensor([.1,-.2,-.3,.5,-.4,.2])
        points=p.new_tensor([[.12,.03,-.03],[.14,0.,-.028]])
        radius=p.new_tensor([.005,.005])
        C,d,z,vz,active=landing_approach_constraints(p,R,J,p.new_zeros(1,6),v,points,radius,
                                                   activation_height=1.)
        eps=1e-4;world_omega=v[0,3:6].numpy();linear=v[0,:3].numpy()
        def height(t):
            rotation=Rotation.from_rotvec(t*world_omega).as_matrix()@R[0].numpy()
            return (p[0].numpy()+t*linear+points.numpy()@rotation.T)[:,2]-radius.numpy()
        np.testing.assert_allclose(z[0],height(0),atol=1e-12)
        np.testing.assert_allclose(vz[0],(height(eps)-height(-eps))/(2*eps),atol=1e-8)
        acceleration=(height(eps)-2*height(0)+height(-eps))/eps**2
        envelope=np.sqrt(.05**2+4*height(0))
        minimum=-2*vz[0].numpy()/envelope-20*(vz[0].numpy()+envelope)
        np.testing.assert_allclose(d[0],acceleration-minimum,atol=1e-7)
        np.testing.assert_allclose((-C@v[...,None])[0,:,0],vz[0],atol=1e-12)

    def test_descent_reaches_floor_at_bounded_speed(self):
        p=torch.tensor([[0.,0.,.06]],dtype=torch.float64)
        v=p.new_zeros(1,35);v[0,2]=-.4
        J=p.new_zeros(1,6,35);J[0,:,:6]=torch.eye(6,dtype=p.dtype)
        for _ in range(10000):
            C,d,_,_,_=landing_approach_constraints(p,torch.eye(3,dtype=p.dtype)[None],
                J,p.new_zeros(1,6),v,p.new_zeros(1,3),p.new_zeros(1))
            a=max(-2.,-float(d[0,0]))
            self.assertLess(a,5.)
            v[0,2]+=.0002*a;p[0,2]+=.0002*v[0,2]
            if p[0,2]<=0:break
        self.assertLessEqual(p[0,2].item(),0.)
        self.assertGreater(v[0,2].item(),-.06)
        self.assertLess(v[0,2].item(),0.)

    def test_high_clearance_is_inactive_and_invalid_speed_rejected(self):
        p=torch.tensor([[0.,0.,1.]],dtype=torch.float64);R=torch.eye(3,dtype=p.dtype)[None]
        args=(p,R,p.new_zeros(1,6,35),p.new_zeros(1,6),p.new_zeros(1,35),
              p.new_zeros(1,3),p.new_zeros(1))
        C,d,_,_,active=landing_approach_constraints(*args)
        self.assertFalse(active.any());self.assertEqual(float(C.abs().sum()+d.abs().sum()),0.)
        with self.assertRaises(ValueError):landing_approach_constraints(*args,touchdown_speed=0.)


if __name__=='__main__':unittest.main()
