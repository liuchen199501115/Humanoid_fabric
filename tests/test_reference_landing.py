import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from fabrics_sim.reference.landing import LandingTransition, minimum_swing_duration


class LandingTransitionTests(unittest.TestCase):
    def test_forward_initial_velocity_is_not_counted_as_adverse_speed(self):
        start=np.array([0.,0.,.035]);goal=np.array([1.126,0.,.035])
        velocity=np.array([.2,0.,0.])
        duration=minimum_swing_duration(start,velocity,goal,3.5)
        self.assertLess(duration,.65)
        path=LandingTransition(0.,np.r_[start,[0,0,0,1]],np.r_[velocity,[0,0,0]],
                               np.r_[goal,[0,0,0,1]],duration=duration,clearance=.06)
        speeds=[np.linalg.norm(path.sample(t)[1][:3]) for t in np.linspace(0,duration,1001)]
        self.assertLessEqual(max(speeds),3.5+1e-9)
        self.assertGreater(max(speeds),3.49)

    def test_duration_bounds_cartesian_speed_for_long_steps(self):
        rng = np.random.default_rng(42)
        for _ in range(12):
            start = np.r_[rng.uniform(-1., 1., 3), [0., 0., 0., 1.]]
            goal = np.r_[rng.uniform(-1., 1., 3), [0., 0., 0., 1.]]
            velocity = np.r_[rng.uniform(-.2, .2, 3), np.zeros(3)]
            duration = minimum_swing_duration(start[:3], velocity[:3], goal[:3], 2.)
            path = LandingTransition(0., start, velocity, goal, duration=duration, clearance=.06)
            speeds = [np.linalg.norm(path.sample(t)[1][:3]) for t in np.linspace(0., duration, 101)]
            self.assertLessEqual(max(speeds), 2. + 1e-10)
        with self.assertRaises(RuntimeError):
            minimum_swing_duration(np.zeros(3), [2., 0., 0.], np.ones(3), 2.)

    def setUp(self):
        self.start = np.r_[[.1, .2, .15], Rotation.from_euler('xyz', [.2, -.1, .4]).as_quat()]
        self.goal = np.r_[[.3, .1, .035], Rotation.from_euler('z', .7).as_quat()]
        self.velocity = np.array([.02, -.01, -.03, .1, -.2, .3])
        self.landing = LandingTransition(5., self.start, self.velocity, self.goal)

    def test_matches_initial_pose_and_twist_and_settles_at_goal(self):
        pose, velocity, acceleration = self.landing.sample(5.)
        np.testing.assert_allclose(pose, self.start, atol=1e-8)
        np.testing.assert_allclose(velocity, self.velocity, atol=1e-7)
        np.testing.assert_allclose(acceleration, 0., atol=1e-7)
        for time in (7., 8.):
            pose, velocity, acceleration = self.landing.sample(time)
            np.testing.assert_allclose(pose, self.goal, atol=1e-8)
            np.testing.assert_array_equal(velocity, np.zeros(6))
            np.testing.assert_array_equal(acceleration, np.zeros(6))

    def test_derivatives_match_pose_and_twist_changes(self):
        eps = 1e-4
        for time in (5.2, 6., 6.8):
            before, v_before, _ = self.landing.sample(time-eps)
            after, v_after, _ = self.landing.sample(time+eps)
            pose, velocity, acceleration = self.landing.sample(time)
            numerical_velocity = np.r_[(after[:3]-before[:3])/(2*eps),
                (Rotation.from_quat(after[3:])*Rotation.from_quat(before[3:]).inv()).as_rotvec()/(2*eps)]
            np.testing.assert_allclose(velocity, numerical_velocity, atol=1e-7)
            np.testing.assert_allclose(acceleration, (v_after-v_before)/(2*eps), atol=1e-7)
            self.assertAlmostEqual(np.linalg.norm(pose[3:]), 1.)

    def test_wall_time_lowers_target_without_source_frame_input(self):
        start = np.array([0., 0., .15, 0., 0., 0., 1.])
        goal = start.copy(); goal[2] = .035
        landing = LandingTransition(1., start, np.zeros(6), goal)
        heights = [landing.sample(t)[0][2] for t in np.linspace(1., 3., 30)]
        self.assertTrue(np.all(np.diff(heights) < 0))
        self.assertAlmostEqual(heights[-1], .035)

    def test_rejects_nonfinite_timing(self):
        for kwargs in ({'start_time': float('nan')}, {'duration': float('inf')}):
            args = dict(start_time=5., start_pose=self.start, start_velocity=self.velocity, goal_pose=self.goal)
            args.update(kwargs)
            with self.assertRaises(ValueError):
                LandingTransition(**args)

    def test_swing_clearance_and_bounded_search_derivatives(self):
        start=np.array([0.,0.,.035,0.,0.,0.,1.])
        goal=start.copy();goal[0]=.22
        trajectory=LandingTransition(0.,start,np.zeros(6),goal,
                                     clearance=.06,search_depth=.04,search_duration=1.5)
        self.assertAlmostEqual(trajectory.sample(1.)[0][2],.095)
        self.assertAlmostEqual(trajectory.sample(2.)[0][2],.035)
        self.assertAlmostEqual(trajectory.sample(10.)[0][2],-.005)
        for time in (.3,1.5,2.4,3.1):
            eps=1e-4
            p0,v0,_=trajectory.sample(time-eps)
            p1,v1,_=trajectory.sample(time+eps)
            _,v,a=trajectory.sample(time)
            np.testing.assert_allclose(v[:3],(p1[:3]-p0[:3])/(2*eps),atol=1e-7)
            np.testing.assert_allclose(a,(v1-v0)/(2*eps),atol=1e-7)
        for time in (0.,2.,3.5):
            _,v,a=trajectory.sample(time)
            np.testing.assert_allclose(v,0.,atol=1e-8)
            np.testing.assert_allclose(a,0.,atol=1e-8)


if __name__ == '__main__':
    unittest.main()
