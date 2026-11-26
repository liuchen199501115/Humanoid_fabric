"""Geometric invariants and reference derivatives for measured heading adaptation."""
import sys
import unittest
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'examples'))
import test_g1  # Installs the existing experimental reference helper path.
from g1_stance_heading import StanceHeadingReference
from measured_stance_reference import Anchor
from blended_spline import BlendedSplineReference
from fabrics_sim.reference.interpolation import MotionInterpolator
from fabrics_sim.reference.whole_body import WholeBodyReference


class StanceHeadingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        folder = ROOT/'acc_data/amp_walk_20261103/feedback_followup'
        cls.trajectory = torch.tensor(np.load(folder/'wbc_convex_catch15_25_clear35.npz')['control_q'], dtype=torch.float64)
        cls.names = np.load(folder/'wrench_fixed_ten.npz')['final_joint_order']
        cls.sampler = BlendedSplineReference(MotionInterpolator(cls.trajectory), cls.trajectory,
                                            stride=6, start_frame=130, ramp_frames=20)

    def projector(self):
        return StanceHeadingReference(str(ROOT/'src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf'),
                                      self.names, np.full(2000, 3), 200)

    def test_heading_is_frozen_from_loaded_measured_contact(self):
        p = self.projector();q = self.trajectory[970].numpy()
        pos, rotation, _ = p.model.evaluate(q, 1)
        point = pos+rotation@np.array([.04, .01, -.035])
        data = (np.array([120.]), point[None], np.array([[0., 0., 1.]]),
                np.array([0.]), np.array([0, 1]), np.array([0, 0]))
        p.observe(970., q, np.array([0., 0.]), data)
        self.assertFalse(p.anchors)
        p.observe(970., q, np.array([0., 120.]), data)
        heading = p.headings[1];original = p.anchors[1].point.copy()
        changed = q.copy();changed[:3] += .01
        p.observe(971., changed, np.array([0., 120.]), data)
        self.assertEqual(len(p.events), 1);self.assertEqual(heading, p.headings[1])
        np.testing.assert_array_equal(original, p.anchors[1].point)

    def test_foot_link_path_fixed_and_fades_are_c2(self):
        p = self.projector();q = self.trajectory[970].numpy()
        pos, rotation, _ = p.model.evaluate(q, 1);local = np.array([.04, .01, -.035])
        p.anchors[1] = Anchor(950., 1039., pos+rotation@local, local)
        p.headings[1] = np.arctan2(rotation[1, 0], rotation[0, 0])-.3
        for f in (950., 965., 980., 1009., 1024., 1039.):
            target_pos, target_rot = p.target(f, 1, pos, rotation)
            np.testing.assert_allclose(target_pos, pos, atol=1e-14)
            self.assertLessEqual(Rotation.from_matrix(target_rot@rotation.T).magnitude(), p.limit)
        for edge in (950., 1039.):
            h=.01
            samples=[p.target(edge+d, 1, pos, rotation)[1] for d in (-h, 0., h)]
            np.testing.assert_allclose(samples[1], rotation, atol=1e-14)
            self.assertLess(abs((samples[2]-samples[0])/(2*h)).max(), 1e-7)
            self.assertLess(abs((samples[2]-2*samples[1]+samples[0])/h**2).max(), 1e-5)

    def test_leg_correction_derivatives_and_unchanged_root_upper_body(self):
        p = self.projector();q = self.sampler.kinematics(970., 240.)[0].numpy()[0]
        pos, rotation, _ = p.model.evaluate(q, 1);local=np.array([.04, 0., -.035])
        p.anchors[1] = Anchor(950., 1039., pos+rotation@local, local)
        p.headings[1] = np.arctan2(rotation[1, 0], rotation[0, 0])-.15
        def sample(f):
            reference = WholeBodyReference(*self.sampler.kinematics(f, 240.))
            return p.project(f, 240., 0., self.sampler, reference)
        for f in (950., 965.3, 980.3, 1009., 1024.3, 1039.):
            original=self.sampler.kinematics(f, 240.)[0];out=sample(f)
            before,after=sample(f-.24),sample(f+.24)
            self.assertTrue(torch.equal(out.q[:,:13], original[:,:13]))
            self.assertTrue(torch.equal(out.q[:,19:], original[:,19:]))
            self.assertLess(float(abs((after.q[:,7:]-before.q[:,7:])/.002-out.v[:,6:]).max()), .02)
            self.assertLess(float(abs((after.v[:,6:]-before.v[:,6:])/.002-out.a[:,6:]).max()), 2.)
        self.assertLess(p.peak_position_residual, 1e-6)
        self.assertLess(p.peak_orientation_residual, 1e-6)


if __name__ == '__main__':
    unittest.main()
