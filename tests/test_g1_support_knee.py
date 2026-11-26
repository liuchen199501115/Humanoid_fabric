import sys
import unittest
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'examples'))
from g1_support_knee import support_knee_constraints

class SupportKneeTest(unittest.TestCase):
    def test_critically_damped_motion_satisfies_barrier_and_stays_flexed(self):
        w=20.;minimum=.2;h0=.4;v0=-5.;eps=1e-4
        height=lambda t:(h0+(v0+w*h0)*t)*np.exp(-w*t)
        for t in np.linspace(0.,.3,31):
            h=height(t);v=(height(t+eps)-height(t-eps))/(2*eps)
            a=(height(t+eps)-2*h+height(t-eps))/eps**2
            position=torch.zeros(1,29,dtype=torch.float64);velocity=position.clone()
            position[0,9]=minimum+h;velocity[0,9]=v
            C,d,diagnostics=support_knee_constraints(position,velocity,torch.tensor([[False,True]]),minimum_angle=minimum,bandwidth=w)
            acceleration=torch.zeros(1,35,dtype=torch.float64);acceleration[0,15]=a
            self.assertGreater(h,0.)
            self.assertLess(abs(float((C@acceleration[...,None]).squeeze(-1)[0,1]-d[0,1])),2e-4)
            self.assertGreaterEqual(float(diagnostics['support_knee_invariant'][0,1]),-1e-5)
            self.assertEqual(torch.count_nonzero(C[0,0]).item(),0)
            self.assertEqual(d[0,0].item(),0.)

    def test_contact_and_joint_order_select_only_the_support_knee(self):
        q=torch.full((2,29),.4);v=torch.zeros_like(q)
        C,d,fields=support_knee_constraints(q,v,torch.tensor([[True,False],[False,True]]),minimum_angle=.2,knee_indices=(5,11))
        self.assertEqual(C[0,0,11].item(),-1.)
        self.assertEqual(C[1,1,17].item(),-1.)
        self.assertEqual(torch.count_nonzero(C).item(),2)
        self.assertTrue(torch.all(d>=0))
        self.assertTrue(torch.equal(fields['support_knee_active'],torch.tensor([[True,False],[False,True]])))

    def test_invalid_parameters_and_states_are_rejected(self):
        q=torch.zeros(1,29);active=torch.ones(1,2,dtype=torch.bool)
        for value in [0.,-1.,1.,float('nan')]:
            with self.assertRaises(ValueError):support_knee_constraints(q,q,active,minimum_angle=value)
        for value in [0.,float('nan')]:
            with self.assertRaises(ValueError):support_knee_constraints(q,q,active,minimum_angle=.2,bandwidth=value)
        with self.assertRaises(ValueError):support_knee_constraints(q,q,active,minimum_angle=.2,knee_indices=(3,3))
        with self.assertRaises(ValueError):support_knee_constraints(q,q,active.float(),minimum_angle=.2)
        with self.assertRaises(ValueError):support_knee_constraints(q+float('nan'),q,active,minimum_angle=.2)

if __name__=='__main__':unittest.main()
