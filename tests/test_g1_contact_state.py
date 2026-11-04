import sys
from pathlib import Path
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import torch


EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from test_g1 import (  # noqa: E402
    ground_motion_reference,
    ramped_motion_time,
    select_balance_support,
    update_contact_mode_state_machine,
    offset_left_shoulder_reference,
    G1_JOINT_NAMES,
    advance_contact_phase,
    align_pelvis_balance_reference,
    support_aware_com_target,
    single_support_ready,
)


class MotionRampTest(unittest.TestCase):
    def test_playback_clock_starts_gently_and_reaches_original_speed(self):
        self.assertEqual(ramped_motion_time(-1.,1.),0.)
        self.assertAlmostEqual(ramped_motion_time(1.,1.),.5)
        self.assertAlmostEqual(ramped_motion_time(2.,1.),1.5)
        self.assertEqual(ramped_motion_time(2.,0.),2.)
        dt=1e-4
        self.assertLess(ramped_motion_time(dt,1.)/dt,1e-6)
        self.assertAlmostEqual((ramped_motion_time(1.+dt,1.)-ramped_motion_time(1.-dt,1.))/(2*dt),1.,places=6)
        self.assertGreater(ramped_motion_time(.3,1.),ramped_motion_time(.2,1.))


class GroundMotionTest(unittest.TestCase):
    def test_alignment_translates_all_frames_without_mutating_source(self):
        source = torch.zeros(3,36)
        source[:,2] = torch.tensor([.8,.81,.82]); source[:,6] = 1
        original = source.clone()
        feet = torch.tensor([[[0.,0.,.055,0.,0.,0.,1.],
                              [0.,0.,.065,0.,0.,0.,1.]]])
        with patch("test_g1.reference_feet_transforms", return_value=feet):
            result, shift = ground_motion_reference(None,source,0)
        self.assertAlmostEqual(shift,-.02,places=6)
        torch.testing.assert_close(source,original)
        torch.testing.assert_close(result[:,2],source[:,2]-.02)
        torch.testing.assert_close(result[:,:2],source[:,:2])
        torch.testing.assert_close(result[:,3:],source[:,3:])
        self.assertAlmostEqual(.055+shift-.035,0.,places=6)


class UnexpectedTouchdownTest(unittest.TestCase):
    def test_loaded_swing_contact_overrides_unchanged_single_support_plan(self):
        mode, _ = update_contact_mode_state_machine(
            torch.tensor([2, 2]), torch.tensor([2, 2]),
            torch.tensor([[False, True], [True, True]]), torch.zeros(2, dtype=torch.long), 3,
            foot_normal_forces=torch.tensor([[280., 300.], [10., 300.]]))
        self.assertEqual(mode.tolist(), [3, 2])


class G1ContactStateMachineTest(unittest.TestCase):
    def test_single_support_gate_needs_balance_not_just_load(self):
        com = torch.tensor([[0., 0., .75]])
        zero = torch.zeros(1, 3)
        forces = torch.tensor([[5., 300.]])
        def ready(c=com, v=zero, target=com, speed=torch.zeros(1), load=forces):
            return bool(single_support_ready(c, v, target, zero, speed, load, 1).item())
        self.assertTrue(ready())
        self.assertFalse(ready(v=torch.tensor([[0., .1, 0.]])))
        self.assertFalse(ready(target=com + .1))
        self.assertFalse(ready(speed=torch.tensor([.03])))
        self.assertFalse(ready(load=torch.tensor([[100., 200.]])))
        displaced = com + torch.tensor([[0., .06, 0.]])
        self.assertFalse(ready(c=displaced, target=displaced))

    def test_stance_targets_share_anchor_and_swing_derivatives_are_unchanged(self):
        from fabrics_sim.fabrics.g1_pose_fabric import G1PoseFabric
        for kind in ("none", "velocity", "full"):
            with self.subTest(kind=kind):
                position = torch.arange(12.).reshape(2, 2, 3)
                quaternion = torch.zeros(2, 2, 4)
                quaternion[..., 3] = 1.
                velocity = torch.arange(24.).reshape(2, 12)
                acceleration = velocity + 30.
                features = None if kind == "none" else velocity
                if kind == "full":
                    features = {"velocity": velocity, "acceleration": acceleration}
                saved = []
                fabric = SimpleNamespace(
                    batch_size=2,
                    feet_pose_taskmap=SimpleNamespace(position_target=position, quaternion_target=quaternion),
                    fabrics_features={"feet_pose": {"feet_pose_attractor": features}},
                    set_feet_targets=lambda *args: saved.append(args),
                )
                anchors = torch.cat((position + 10., quaternion), dim=-1)
                anchors[..., 3:] = torch.tensor([0., 0., 1., 0.])
                active = torch.tensor([[True, False], [False, True]])
                G1PoseFabric.anchor_support_feet(fabric, anchors, active)
                p, quat, lv, av, la, aa = saved[0]
                torch.testing.assert_close(p[active], anchors[..., :3][active])
                torch.testing.assert_close(p[~active], position[~active])
                torch.testing.assert_close(quat[active], anchors[..., 3:][active])
                torch.testing.assert_close(quat[~active], quaternion[~active])
                for actual, original in ((None if lv is None else torch.cat((lv, av), -1), velocity),
                                         (None if la is None else torch.cat((la, aa), -1), acceleration)):
                    if actual is not None:
                        self.assertEqual(actual[active].count_nonzero(), 0)
                        torch.testing.assert_close(actual[~active], original.reshape(2, 2, 6)[~active])
                if kind == "none":
                    self.assertIsNone(lv)
                if kind != "full":
                    self.assertIsNone(la)
                torch.testing.assert_close(position, torch.arange(12.).reshape(2, 2, 3))
                torch.testing.assert_close(velocity, torch.arange(24.).reshape(2, 12))
                torch.testing.assert_close(anchors[..., :3], position + 10.)

    def test_balance_uses_planted_anchor_but_not_swing_anchor(self):
        class Fabric:
            def compute_center_of_mass(self, q):
                return q[:, :3]
        q = torch.zeros(1, 36)
        feet = torch.zeros(1, 2, 7)
        feet[0, 1, :3] = torch.tensor([20., -2., 0.])
        anchors = torch.tensor([[[5., 1., 0.], [6., -1., 0.]]])
        with patch("test_g1.reference_feet_transforms", return_value=feet):
            actual = support_aware_com_target(Fabric(), q, q, torch.tensor([2]),
                                             anchors, torch.tensor([[True, True]]))
            torch.testing.assert_close(actual, torch.tensor([[4.5, -.75, 0.]]))
            future = support_aware_com_target(Fabric(), q, q, torch.tensor([2]),
                                             anchors, torch.tensor([[True, False]]))
            torch.testing.assert_close(future, torch.tensor([[15., -1.5, 0.]]))

    def test_pelvis_balance_translation_preserves_height_and_rotation(self):
        class Fabric:
            def compute_center_of_mass(self, q):
                return q[:, :3] + torch.tensor([[.1, .2, -.1]])
            def set_pelvis_target(self, *args):
                self.target = args
        fabric = Fabric()
        q = torch.zeros(1, 36)
        q[:, 2], q[:, 6] = .8, 1.
        com = fabric.compute_center_of_mass(q) + torch.tensor([[.03, -.06, .4]])
        velocity = torch.tensor([[.2, .3, .8]])
        acceleration = torch.tensor([[.4, .5, .9]])
        align_pelvis_balance_reference(fabric, q, None, None, .02, com, velocity, acceleration)
        position, quaternion, linear_v, angular_v, linear_a, angular_a = fabric.target
        torch.testing.assert_close(position, torch.tensor([[.03, -.06, .8]]))
        torch.testing.assert_close(quaternion, q[:, 3:7])
        torch.testing.assert_close(linear_v, torch.tensor([[.2, .3, 0.]]))
        torch.testing.assert_close(linear_a, torch.tensor([[.4, .5, 0.]]))
        self.assertEqual(angular_v.count_nonzero(), 0)
        self.assertEqual(angular_a.count_nonzero(), 0)

    def test_phase_clock_waits_for_support_without_catching_up(self):
        modes = torch.tensor([3, 3, 2, 2, 3, 1, 1])
        frame, requested, held = advance_contact_phase(0., 3., modes, 3)
        self.assertEqual((frame, requested, held), (1., 2, True))
        self.assertEqual(advance_contact_phase(frame, 1., modes, 3), (1., 2, True))
        self.assertEqual(advance_contact_phase(frame, 1., modes, 2), (2., 2, False))
        # A long step may not skip the next unprepared support phase.
        self.assertEqual(advance_contact_phase(2., 4., modes, 2), (4., 1, True))
        # Double support is allowed so the swing foot can complete touchdown.
        self.assertEqual(advance_contact_phase(3., 1., modes, 2), (4., 3, False))
        self.assertEqual(advance_contact_phase(6., 1., modes, 1), (6., 1, False))
        # Holding at a fractional cursor must not rewind its reference pose.
        self.assertEqual(advance_contact_phase(1.4, .2, modes, 3), (1.4, 2, True))
        self.assertEqual(advance_contact_phase(1., 0., modes, 3), (1., 3, False))
        with self.assertRaises(ValueError):
            advance_contact_phase(0., -1., modes, 3)
        with self.assertRaises(ValueError):
            advance_contact_phase(float("nan"), 1., modes, 3)

    def test_single_support_waits_for_actual_load_transfer(self):
        mode, counter = torch.tensor([3]), torch.tensor([0])
        measured = torch.tensor([[True, True]])
        for _ in range(5):
            mode, counter = update_contact_mode_state_machine(
                mode, torch.tensor([2]), measured, counter, 2,
                foot_normal_forces=torch.tensor([[172., 183.]]))
        self.assertEqual(mode.item(), 3)
        self.assertEqual(counter.item(), 0)
        mode, counter = update_contact_mode_state_machine(
            mode, torch.tensor([2]), measured, counter, 2,
            foot_normal_forces=torch.tensor([[40., 280.]]))
        self.assertEqual(mode.item(), 3)
        mode, counter = update_contact_mode_state_machine(
            mode, torch.tensor([2]), measured, counter, 2,
            foot_normal_forces=torch.tensor([[40., 280.]]))
        self.assertEqual(mode.item(), 2)

    def test_clearance_reference_does_not_modify_source_or_other_coordinates(self):
        source = torch.randn(4, 36, dtype=torch.float64)
        saved = source.clone()
        adjusted = offset_left_shoulder_reference(source, .05)
        difference = torch.zeros_like(source)
        difference[:, 7 + G1_JOINT_NAMES.index("left_shoulder_roll_joint")] = .05
        torch.testing.assert_close(source, saved)
        torch.testing.assert_close(adjusted - source, difference)
        with self.assertRaises(ValueError):
            offset_left_shoulder_reference(source, float("nan"))
        with self.assertRaises(ValueError):
            offset_left_shoulder_reference(source[:, :35], .05)

    def test_single_support_is_not_overridden_by_lookahead(self):
        reference = torch.zeros((1, 36))
        reference[:, 0] = 1.0
        lookahead = torch.zeros((1, 36))
        lookahead[:, 0] = 2.0

        mode, selected = select_balance_support(
            commanded_mode=torch.tensor([2]),
            measured_mode=torch.tensor([3]),
            reference_q=reference,
            lookahead_reference_q=lookahead,
            lookahead_mode=torch.tensor([1]),
        )

        self.assertEqual(mode.item(), 2)
        torch.testing.assert_close(selected, reference)

    def test_double_support_uses_lookahead(self):
        reference = torch.zeros((1, 36))
        lookahead = torch.ones((1, 36))

        mode, selected = select_balance_support(
            commanded_mode=torch.tensor([3]),
            measured_mode=torch.tensor([3]),
            reference_q=reference,
            lookahead_reference_q=lookahead,
            lookahead_mode=torch.tensor([1]),
        )

        self.assertEqual(mode.item(), 1)
        torch.testing.assert_close(selected, lookahead)

    def test_double_to_single_requires_loaded_support(self):
        mode = torch.tensor([3])
        counter = torch.tensor([0])
        planned = torch.tensor([2])
        measured = torch.tensor([[False, True]])

        mode, counter = update_contact_mode_state_machine(
            mode, planned, measured, counter, confirm_updates=2
        )
        self.assertEqual(mode.item(), 3)
        mode, counter = update_contact_mode_state_machine(
            mode, planned, measured, counter, confirm_updates=2
        )
        self.assertEqual(mode.item(), 2)

    def test_flight_request_and_measured_touchdown(self):
        mode = torch.tensor([1])
        counter = torch.tensor([0])
        planned_flight = torch.tensor([0])
        no_contact = torch.tensor([[False, False]])

        mode, counter = update_contact_mode_state_machine(
            mode, planned_flight, no_contact, counter, confirm_updates=2
        )
        self.assertEqual(mode.item(), 1)
        mode, counter = update_contact_mode_state_machine(
            mode, planned_flight, no_contact, counter, confirm_updates=2
        )
        self.assertEqual(mode.item(), 0)

        mode, counter = update_contact_mode_state_machine(
            mode,
            torch.tensor([2]),
            torch.tensor([[False, True]]),
            counter,
            confirm_updates=2,
        )
        self.assertEqual(mode.item(), 2)

    def test_missing_commanded_contact_follows_measured_support(self):
        mode, counter = update_contact_mode_state_machine(
            torch.tensor([3]),
            torch.tensor([3]),
            torch.tensor([[False, True]]),
            torch.tensor([2]),
            confirm_updates=3,
        )
        self.assertEqual(mode.item(), 2)
        self.assertEqual(counter.item(), 0)

        mode, _ = update_contact_mode_state_machine(
            torch.tensor([1]),
            torch.tensor([1]),
            torch.tensor([[False, False]]),
            torch.tensor([0]),
            confirm_updates=3,
        )
        self.assertEqual(mode.item(), 0)


if __name__ == "__main__":
    unittest.main()
