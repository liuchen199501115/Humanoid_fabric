import unittest
from pathlib import Path

import torch

from retarget import DEFAULT_MOTION_ID, parse_args as parse_retarget_args
from retargeting.h5_motion import load_motion_h5
from retargeting.inference import (
    _crop_time,
    _pad_time_to_multiple,
    _resolve_output_fps,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
BUILTIN_MOTIONS = {
    "examples/human_motion.h5": 358,
    "examples/motions/cmu_35_17_run_jog.h5": 167,
    "examples/motions/cmu_49_02_jump_hop.h5": 2085,
    "examples/motions/cmu_49_19_one_leg_balance.h5": 957,
    "examples/motions/cmu_144_20_punch.h5": 2270,
}


class TemporalPaddingTest(unittest.TestCase):
    def test_short_motion_repeats_last_frame_to_minimum_length(self):
        pose = torch.arange(2 * 3 * 4 * 12, dtype=torch.float32).reshape(2, 3, 4, 12)

        padded, pad = _pad_time_to_multiple(pose, multiple=4, minimum=16)

        self.assertEqual(pad, 4)
        self.assertEqual(padded.shape, (2, 3, 4, 16))
        torch.testing.assert_close(padded[..., :12], pose)
        torch.testing.assert_close(
            padded[..., 12:],
            pose[..., -1:].expand(2, 3, 4, 4),
        )
        torch.testing.assert_close(_crop_time(padded, pad), pose)

    def test_regular_motion_only_pads_to_architecture_multiple(self):
        pose = torch.zeros(1, 2, 3, 17)

        padded, pad = _pad_time_to_multiple(pose, multiple=4, minimum=16)

        self.assertEqual(pad, 3)
        self.assertEqual(padded.shape[-1], 20)

    def test_empty_motion_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least one frame"):
            _pad_time_to_multiple(torch.zeros(1, 2, 3, 0), multiple=4, minimum=16)


class OutputFpsTest(unittest.TestCase):
    def test_example_fps_is_loaded_and_used_by_default(self):
        motion = load_motion_h5(
            REPO_ROOT / "examples/human_motion.h5",
            velocity_virtual_node=True,
            use_global_y=False,
        )

        self.assertEqual(motion.info["fps"], 120.0)
        self.assertEqual(_resolve_output_fps(None, motion.info), 120.0)

    def test_explicit_fps_overrides_input_metadata(self):
        self.assertEqual(_resolve_output_fps(60, {"fps": 120}), 60.0)

    def test_legacy_h5_falls_back_to_30_fps(self):
        self.assertEqual(_resolve_output_fps(None, {"fps": None}), 30.0)

    def test_non_positive_fps_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "must be positive"):
            _resolve_output_fps(0, {"fps": 120})


class BuiltinMotionTest(unittest.TestCase):
    def test_all_builtin_motions_are_loadable(self):
        for relative_path, expected_frames in BUILTIN_MOTIONS.items():
            with self.subTest(path=relative_path):
                motion = load_motion_h5(
                    REPO_ROOT / relative_path,
                    velocity_virtual_node=True,
                    use_global_y=False,
                )

                self.assertEqual(motion.pose.shape, (expected_frames, 21, 6))
                self.assertEqual(motion.info["fps"], 120.0)


class DefaultMotionTest(unittest.TestCase):
    def test_retarget_defaults_to_run_jog(self):
        args = parse_retarget_args([])

        self.assertEqual(DEFAULT_MOTION_ID, "cmu_35_17_run_jog")
        self.assertEqual(
            Path(args.input_h5),
            REPO_ROOT / "examples/motions/cmu_35_17_run_jog.h5",
        )
        self.assertEqual(args.tag, DEFAULT_MOTION_ID)


if __name__ == "__main__":
    unittest.main()
