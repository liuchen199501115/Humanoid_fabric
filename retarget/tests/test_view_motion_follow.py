import unittest
from pathlib import Path

from view_motion_follow import (
    DEFAULT_MOTION_ID,
    DEFAULT_VIDEO_HEIGHT,
    DEFAULT_VIDEO_WIDTH,
    _ensure_offscreen_size,
    _resolve_video_fps,
    parse_args as parse_view_args,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


class _FakeGlobalVisual:
    offwidth = 640
    offheight = 480


class _FakeVisual:
    global_ = _FakeGlobalVisual()


class _FakeModel:
    vis = _FakeVisual()


class OffscreenSizeTest(unittest.TestCase):
    def test_framebuffer_is_expanded_to_requested_resolution(self):
        model = _FakeModel()
        _ensure_offscreen_size(model, DEFAULT_VIDEO_WIDTH, DEFAULT_VIDEO_HEIGHT)
        self.assertEqual(model.vis.global_.offwidth, 1920)
        self.assertEqual(model.vis.global_.offheight, 1080)

        _ensure_offscreen_size(model, 320, 240)
        self.assertEqual(model.vis.global_.offwidth, 1920)
        self.assertEqual(model.vis.global_.offheight, 1080)

    def test_non_positive_dimensions_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "must be positive"):
            _ensure_offscreen_size(_FakeModel(), 0, 720)


class VideoFpsTest(unittest.TestCase):
    def test_motion_fps_is_used_by_default(self):
        self.assertEqual(_resolve_video_fps(120), 120.0)

    def test_requested_fps_changes_sampling_rate(self):
        self.assertEqual(_resolve_video_fps(120, 60), 60.0)

    def test_non_positive_fps_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "must be positive"):
            _resolve_video_fps(120, 0)


class DefaultMotionTest(unittest.TestCase):
    def test_viewer_defaults_to_run_jog_output(self):
        args = parse_view_args([])

        self.assertEqual(DEFAULT_MOTION_ID, "cmu_35_17_run_jog")
        self.assertEqual(
            Path(args.motion_file),
            REPO_ROOT / "outputs/B2A_cmu_35_17_run_jog.pkl",
        )


if __name__ == "__main__":
    unittest.main()
