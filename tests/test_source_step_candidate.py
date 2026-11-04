import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from fabrics_sim.reference.source_step import SourceStepCandidate, centroidal_support_check
from fabrics_sim.reference.whole_body import WholeBodyReference


class SourceStepCandidateTests(unittest.TestCase):



    def test_pelvis_offsets_preserve_input_and_nonorientation_coordinates(self):
        from fabrics_sim.reference.source_step import adapt_pelvis_orientation
        from scipy.spatial.transform import Rotation
        q=torch.randn(5,36,dtype=torch.float64)
        q[:,3:7]=torch.tensor(Rotation.from_euler('z',.4).as_quat())
        original=q.clone()
        out=adapt_pelvis_orientation(q,.12,-.07)
        torch.testing.assert_close(q,original)
        torch.testing.assert_close(out[0],q[0])
        torch.testing.assert_close(out[:,:3],q[:,:3])
        torch.testing.assert_close(out[:,7:],q[:,7:])
        expected=(Rotation.from_quat(q[-1,3:7].numpy())*Rotation.from_euler('y',.12)*Rotation.from_euler('x',-.07)).as_quat()
        torch.testing.assert_close(out[-1,3:7],torch.tensor(expected))
        torch.testing.assert_close(out[:,3:7].norm(dim=-1),torch.ones(5,dtype=q.dtype))
        torch.testing.assert_close(adapt_pelvis_orientation(q),q)
        with self.assertRaises(ValueError):adapt_pelvis_orientation(q,roll=float('nan'))

    def test_contact_search_is_late_bounded_and_has_consistent_rates(self):
        plan = object.__new__(SourceStepCandidate)
        plan.duration = 1.5
        plan.support = 0
        plan.start_feet = torch.tensor([[[0.,0.,.035,0.,0.,0.,1.],
                                        [.2,-.1,.035,0.,0.,0.,1.]]],dtype=torch.float64)
        plan.end_feet = plan.start_feet.clone()
        plan.clearance = 0.
        plan.landing_search_depth = .004
        plan.landing_search_duration = .2
        for t in (0.,1.,1.3):
            feet,v,a,_ = plan.foot_kinematics(t)
            torch.testing.assert_close(feet,plan.start_feet)
            torch.testing.assert_close(v,torch.zeros_like(v))
            torch.testing.assert_close(a,torch.zeros_like(a))
        feet,v,a,_ = plan.foot_kinematics(1.5)
        self.assertAlmostEqual(float(feet[0,1,2]),.031)
        torch.testing.assert_close(feet[:,0],plan.start_feet[:,0])
        torch.testing.assert_close(v,torch.zeros_like(v),atol=1e-12,rtol=0.)
        torch.testing.assert_close(a,torch.zeros_like(a),atol=1e-12,rtol=0.)
        t,eps = 1.39,1e-6
        pm,vm,_,_ = plan.foot_kinematics(t-eps)
        pp,vp,_,_ = plan.foot_kinematics(t+eps)
        _,v,a,_ = plan.foot_kinematics(t)
        self.assertAlmostEqual(float((pp[0,1,2]-pm[0,1,2])/(2*eps)),float(v[0,1,2]),places=8)
        self.assertAlmostEqual(float((vp[0,1,2]-vm[0,1,2])/(2*eps)),float(a[0,1,2]),places=7)

    def test_retiming_rates_match_finite_differences(self):
        plan = object.__new__(SourceStepCandidate)
        plan.duration = .8
        h = 1e-5
        for t in (.1, .4, .7):
            left, center, right = [plan.progress(x)[3] for x in (t-h, t, t+h)]
            self.assertAlmostEqual(center[1], (right[0]-left[0])/(2*h), places=5)
            self.assertAlmostEqual(center[2], (right[1]-left[1])/(2*h), places=4)
        self.assertEqual(plan.progress(0.)[3], (0., 0., 0.))
        self.assertEqual(plan.progress(.8)[3], (1., 0., 0.))
        with self.assertRaisesRegex(ValueError, 'continuation'):
            plan.progress(.801)

    def wrench(self, com, momentum_rate_y=0.):
        q = torch.zeros(1, 36); q[:, 6] = 1.
        zero = torch.zeros(1, 35)
        reference = WholeBodyReference(q, zero, zero)
        model = SimpleNamespace(com_task=SimpleNamespace(link_masses=torch.tensor([10.])))
        state = dict(com_jac=torch.zeros(1, 3, 35), com_jac_bias=torch.zeros(1, 3),
                     momentum_matrix=torch.zeros(1, 3, 35),
                     momentum_matrix_bias=torch.tensor([[0., momentum_rate_y, 0.]]),
                     transforms=torch.tensor([[[0., 0., .035, 0., 0., 0., 1.]]]),
                     rotations=torch.eye(3)[None, None], com=torch.tensor([com]))
        return centroidal_support_check(model, reference, state, 0)

    def test_static_gravity_cop_is_com_projection(self):
        force, cop, passed = self.wrench([.02, .005, .65])
        torch.testing.assert_close(force, torch.tensor([[0., 0., 98.1]]))
        torch.testing.assert_close(cop, torch.tensor([[.02, .005]]))
        self.assertTrue(bool(passed))

    def test_whole_body_momentum_can_invalidate_point_mass_balance(self):
        _, cop, passed = self.wrench([.02, 0., .65], momentum_rate_y=20.)
        self.assertAlmostEqual(float(cop[0, 0]), .02-20/98.1, places=6)
        self.assertFalse(bool(passed))
        self.assertFalse(bool(self.wrench([.02, .04, .65])[2]))


if __name__ == '__main__':
    unittest.main()
