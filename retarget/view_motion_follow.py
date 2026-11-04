#!/usr/bin/env python3
from pathlib import Path
import argparse
import logging
import time

import numpy as np
import torch

from retargeting.motion_lib import MotionLib


DEFAULT_VIDEO_WIDTH = 1920
DEFAULT_VIDEO_HEIGHT = 1080
DEFAULT_MOTION_ID = "cmu_35_17_run_jog"


def _ensure_offscreen_size(model, width, height):
    if width <= 0 or height <= 0:
        raise ValueError(f"Render dimensions must be positive, got {width}x{height}")
    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), int(width))
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), int(height))


def _resolve_video_fps(motion_fps, requested_fps=None):
    fps = motion_fps if requested_fps is None else requested_fps
    fps = float(fps)
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"Video FPS must be positive, got {fps}")
    return fps


class MotionFollowViewEnv:
    def __init__(
        self,
        motion_file,
        model_xml=None,
        device="auto",
        record_video=True,
        video_path=None,
        fps=None,
        camera_distance=4.0,
        camera_azimuth=120.0,
        camera_elevation=-20.0,
        lookat_height=-0.4,
        loops=1,
        width=DEFAULT_VIDEO_WIDTH,
        height=DEFAULT_VIDEO_HEIGHT,
    ):
        import mujoco

        self.mujoco = mujoco
        self.logger = logging.getLogger(self.__class__.__name__)
        root = Path(__file__).resolve().parent
        self.motion_file = Path(motion_file)
        self.model_xml = Path(model_xml) if model_xml is not None else root / "assets/g1_description/g1_29dof.xml"
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.record_video = record_video
        self.lookat_height = lookat_height
        self.loops = max(int(loops), 1)
        self.width = int(width)
        self.height = int(height)

        self.motion_lib = MotionLib(self.motion_file, device=device)
        self.motion_ids = torch.tensor([0], dtype=torch.long, device=device)
        self.motion_len = self.motion_lib.get_motion_length(self.motion_ids)

        self.sim_dt = 0.02
        self.video_fps = (
            _resolve_video_fps(self.motion_lib.fps, fps)
            if self.record_video
            else None
        )
        self.control_dt = (
            1.0 / self.video_fps if self.video_fps is not None else self.sim_dt
        )
        self.sim_duration = self.loops * self.motion_len.item()

        self.model = mujoco.MjModel.from_xml_path(str(self.model_xml))
        self.model.opt.timestep = self.sim_dt
        self.data = mujoco.MjData(self.model)
        mujoco.mj_step(self.model, self.data)

        self.camera_distance = camera_distance
        self.camera_azimuth = camera_azimuth
        self.camera_elevation = camera_elevation
        self.viewer = None
        self.renderer = None
        self.video_writer = None
        self.video_path = None
        if self.record_video:
            _ensure_offscreen_size(self.model, self.width, self.height)
            self.renderer = mujoco.Renderer(
                self.model, height=self.height, width=self.width
            )
            self.camera = mujoco.MjvCamera()
            self._configure_camera(self.camera)
            self._init_video_writer(video_path)
        else:
            from mujoco import viewer

            self.viewer = viewer.launch_passive(self.model, self.data)
            self._configure_camera(self.viewer.cam)

        self._init_dof_mapping()

    def _configure_camera(self, camera):
        camera.distance = self.camera_distance
        camera.azimuth = self.camera_azimuth
        camera.elevation = self.camera_elevation

    def _init_video_writer(self, video_path):
        import imageio

        if video_path is None:
            out_dir = Path("mujoco_videos")
            out_dir.mkdir(parents=True, exist_ok=True)
            video_path = out_dir / f"{self.motion_file.stem}_follow.mp4"
        else:
            video_path = Path(video_path)
            video_path.parent.mkdir(parents=True, exist_ok=True)
        self.video_path = video_path
        self.video_writer = imageio.get_writer(
            str(video_path),
            fps=self.video_fps,
            macro_block_size=1,
        )

    def _init_dof_mapping(self):
        mujoco = self.mujoco
        self.model_dof_count = self.model.nq - 7
        model_dof_names = []
        for dof_id in range(self.model.nv):
            joint_id = self.model.dof_jntid[dof_id]
            if self.model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
                continue
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            if joint_name is not None:
                model_dof_names.append(joint_name)
        self.model_dof_names = model_dof_names
        self.dof_index_map = None
        if self.motion_lib.get_dof_names() is not None:
            try:
                self.dof_index_map = self.motion_lib.build_dof_index_map(self.model_dof_names)
            except (KeyError, ValueError) as exc:
                self.logger.warning("Cannot map DOF names exactly, falling back to order alignment: %s", exc)
        self._printed_alignment_warning = False

    def _update_camera(self):
        lookat = self.data.qpos[:3].copy()
        lookat[2] += self.lookat_height
        if self.record_video:
            self.camera.lookat[:] = lookat
        else:
            self.viewer.cam.lookat[:] = lookat

    def _apply_motion(self, curr_time):
        motion_time = torch.tensor([curr_time], dtype=torch.float32, device=self.device) % self.motion_len
        root_pos, root_rot, _, _, dof_pos, _ = self.motion_lib.calc_motion_frame(self.motion_ids, motion_time)
        self.data.qpos[:3] = root_pos[0].detach().cpu().numpy()
        # MuJoCo free-joint quaternion is wxyz; retargeting pkl stores xyzw.
        self.data.qpos[3:7] = root_rot[0].detach().cpu().numpy()[[3, 0, 1, 2]]

        target_dof = dof_pos[0]
        if self.dof_index_map is not None:
            target_dof = target_dof[self.dof_index_map]
        elif target_dof.shape[0] != self.model_dof_count and not self._printed_alignment_warning:
            self.logger.warning(
                "Motion DOF count (%d) != model DOF count (%d); truncating or padding.",
                target_dof.shape[0],
                self.model_dof_count,
            )
            self._printed_alignment_warning = True

        if target_dof.shape[0] > self.model_dof_count:
            target_dof = target_dof[: self.model_dof_count]
        elif target_dof.shape[0] < self.model_dof_count:
            padding = torch.zeros(self.model_dof_count - target_dof.shape[0], device=target_dof.device)
            target_dof = torch.cat([target_dof, padding], dim=0)
        self.data.qpos[7 : 7 + self.model_dof_count] = target_dof.detach().cpu().numpy()

    def run(self):
        mujoco = self.mujoco
        frame_count = int(np.ceil(self.sim_duration / self.control_dt))
        started = time.monotonic()
        try:
            for i in range(frame_count):
                self._apply_motion(i * self.control_dt)
                mujoco.mj_forward(self.model, self.data)
                self._update_camera()
                if self.record_video:
                    self.renderer.update_scene(self.data, camera=self.camera)
                    self.video_writer.append_data(self.renderer.render())
                else:
                    if not self.viewer.is_running():
                        break
                    self.viewer.sync()
                    delay = started + (i + 1) * self.control_dt - time.monotonic()
                    if delay > 0:
                        time.sleep(delay)
        finally:
            if self.viewer is not None:
                self.viewer.close()
            if self.renderer is not None:
                self.renderer.close()
            if self.video_writer is not None:
                self.video_writer.close()
                print(
                    f"[done] video: {self.video_path} "
                    f"({self.video_fps:g} FPS, {self.sim_duration:.3f}s)"
                )


def parse_args(argv=None):
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="View or render a retargeted G1 pkl motion in MuJoCo.")
    parser.add_argument(
        "--motion-file",
        default=root / f"outputs/B2A_{DEFAULT_MOTION_ID}.pkl",
        help="Retargeted pkl motion.",
    )
    parser.add_argument("--model-xml", default=root / "assets/g1_description/g1_29dof.xml", help="G1 MuJoCo XML.")
    parser.add_argument("--video-path", default=None, help="Output mp4 path when recording video.")
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Output video FPS (default: motion FPS); sampling follows this rate without changing speed.",
    )
    parser.add_argument("--device", default="auto", help="'auto', 'cuda', 'cuda:0' or 'cpu'.")
    parser.add_argument("--camera-distance", type=float, default=4.0)
    parser.add_argument("--camera-azimuth", type=float, default=120.0)
    parser.add_argument("--camera-elevation", type=float, default=-20.0)
    parser.add_argument("--lookat-height", type=float, default=-0.4)
    parser.add_argument("--loops", type=int, default=1)
    parser.add_argument(
        "--width",
        type=int,
        default=DEFAULT_VIDEO_WIDTH,
        help=f"Rendered video width (default: {DEFAULT_VIDEO_WIDTH}).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=DEFAULT_VIDEO_HEIGHT,
        help=f"Rendered video height (default: {DEFAULT_VIDEO_HEIGHT}).",
    )
    parser.add_argument("--points", action="store_true", help="Preview raw posed_joints positions without G1 retargeting")
    parser.add_argument("--up-axis", choices=["y", "z"], default="y", help="Source up axis for --points mode")
    parser.add_argument("--no-video", action="store_true", help="Open an interactive MuJoCo viewer instead of rendering mp4.")
    return parser.parse_args(argv)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = parse_args()
    if args.points:
        from point_motion_view import run_points
        run_points(args)
        return
    env = MotionFollowViewEnv(
        motion_file=args.motion_file,
        model_xml=args.model_xml,
        device=args.device,
        record_video=not args.no_video,
        video_path=args.video_path,
        fps=args.fps,
        camera_distance=args.camera_distance,
        camera_azimuth=args.camera_azimuth,
        camera_elevation=args.camera_elevation,
        lookat_height=args.lookat_height,
        loops=args.loops,
        width=args.width,
        height=args.height,
    )
    env.run()


if __name__ == "__main__":
    main()
