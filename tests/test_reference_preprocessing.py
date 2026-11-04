import unittest
from pathlib import Path
import torch
from fabrics_sim.reference.preprocessing import prepare_reference
from fabrics_sim.reference.interpolation import MotionInterpolator
from test_g1_two_tasks import G1_JOINT_NAMES, load_motion_trajectory

ROOT = Path(__file__).resolve().parents[1]

class PreparationTests(unittest.TestCase):
    def test_strict_landing_knots_are_original_poses(self):
        q, fps = load_motion_trajectory(ROOT/'retarget/lafan/walk1_g1.pkl')
        q = q[:199]
        sampler, report = prepare_reference(q, fps, G1_JOINT_NAMES,
            ROOT/'src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf', 50., method='pchip')
        for frame in (129, 155, 176, 198):
            torch.testing.assert_close(sampler(frame), q[frame:frame+1], rtol=0, atol=0)
        self.assertEqual(report['method'], 'pchip')

    def test_acceleration_cap_can_be_disabled_without_changing_knots(self):
        q,fps=load_motion_trajectory(ROOT/'retarget/lafan/walk1_g1.pkl')
        q=q[:199]
        urdf=ROOT/'src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf'
        _,limited=prepare_reference(q,fps,G1_JOINT_NAMES,urdf,50.,method='pchip')
        sampler,report=prepare_reference(q,fps,G1_JOINT_NAMES,urdf,None,method='pchip')
        self.assertFalse(report['acceleration_cap_enabled'])
        self.assertIsNone(report['planning_acceleration_cap'])
        self.assertGreaterEqual(report['playback_speed'],limited['playback_speed'])
        for frame in (129,155,176,198):
            torch.testing.assert_close(sampler(frame),q[frame:frame+1],atol=0,rtol=0)

    def test_original_spline_and_chain_rule(self):
        q, fps = load_motion_trajectory(ROOT/'retarget/lafan/walk1_g1.pkl')
        original = q.clone()
        sampler, report = prepare_reference(q, fps, G1_JOINT_NAMES,
            ROOT/'src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf', 50.)
        self.assertTrue(torch.equal(q, original))
        self.assertLessEqual(report['constant_rate_peak_joint_acceleration'], 50.00001)
        self.assertGreater(report['playback_speed'], 0)
        direct = MotionInterpolator(q, method='bspline')
        for frame in (0., 114.2, 129.8, len(q)-1.):
            for got, want in zip(sampler.kinematics(frame, 3., -.2), direct.kinematics(frame, 3., -.2)):
                torch.testing.assert_close(got, want)
        _, v, a = sampler.kinematics(114.2, 0., 0.)
        self.assertEqual(float(v.abs().max()), 0.)
        self.assertEqual(float(a.abs().max()), 0.)
        with self.assertRaises(ValueError):
            prepare_reference(q, fps, G1_JOINT_NAMES,
                ROOT/'src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf', -1.)

if __name__ == '__main__':
    unittest.main()
