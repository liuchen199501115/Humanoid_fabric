"""Render an audited measured G1 tracking prefix to MP4, without physics stepping.

Example (CPU rendering works without an NVIDIA driver):
    python examples/export_g1_tracking.py \
        acc_data/g1_full_tracking_20261106/baseline.npz \
        --output acc_data/videos/g1_tracking_12_steps.mp4 --software
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, help="Defaults to RECORDING.audit.json")
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--software", action="store_true", help="Use Mesa CPU rendering")
    args = parser.parse_args()
    if args.steps < 1 or args.fps < 1 or not np.isfinite(args.speed) or args.speed <= 0:
        parser.error("steps, fps and speed must be positive")
    if args.output.suffix.lower() != ".mp4":
        parser.error("output must have an .mp4 extension")

    audit_path = args.audit or args.recording.with_suffix(".audit.json")
    audit = json.loads(audit_path.read_text())
    if Path(audit["recording"]).resolve() != args.recording.resolve():
        raise ValueError("Audit belongs to a different recording")
    landings = audit["landings"][:args.steps]
    if (len(landings) != args.steps or any(
            row["step"] != i + 1 or not row["prefix_fidelity_passed"]
            for i, row in enumerate(landings))):
        raise ValueError("Requested prefix has not passed the tracking audit")
    end_time = float(landings[-1]["time_s"])
    with np.load(args.recording, allow_pickle=False) as source:
        poses = np.asarray(source["history_q"], dtype=np.float64).reshape(-1, 36)
        times = np.asarray(source["history_time"], dtype=np.float64)
        joint_names = source["final_joint_order"].tolist()
        model_name = str(source["fabric_robot_name"].item())
    if (len(poses) != len(times) or len(times) < 2
            or not np.isfinite(poses).all() or not np.isfinite(times).all()
            or not np.all(np.diff(times) > 0)
            or not times[0] < end_time <= times[-1]):
        raise ValueError("Invalid measured history or prefix time range")
    if len(joint_names) != 29 or len(set(joint_names)) != 29:
        raise ValueError("Expected 29 unique recorded joint names")

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    if args.software:
        os.environ["__EGL_VENDOR_LIBRARY_FILENAMES"] = (
            "/usr/share/glvnd/egl_vendor.d/50_mesa.json")
        os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
        os.environ["EGL_PLATFORM"] = "surfaceless"
    # Older PyOpenGL versions lack the opaque EGL_EXT_device_base handle alias.
    if os.environ["MUJOCO_GL"] == "egl":
        from OpenGL import EGL
        if not hasattr(EGL, "EGLDeviceEXT"):
            EGL.EGLDeviceEXT = ctypes.c_void_p
    import mujoco
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw, ImageFont
    from scipy.spatial.transform import Rotation, Slerp

    root = Path(__file__).resolve().parents[1]
    model_dir = root / "src/fabrics_sim/models/robots/urdf/g1"
    model_xml = ET.parse(model_dir / f"{model_name}.xml").getroot()
    model_xml.find("compiler").set("meshdir", str(model_dir / "meshes"))
    visual = ET.SubElement(model_xml, "visual")
    ET.SubElement(visual, "global", offwidth="1280", offheight="720")
    ET.SubElement(visual, "headlight", ambient="0.4 0.4 0.4", diffuse="0.7 0.7 0.7")
    asset = model_xml.find("asset")
    ET.SubElement(asset, "texture", type="skybox", builtin="gradient",
                  rgb1="0.12 0.17 0.24", rgb2="0.35 0.42 0.50", width="512", height="3072")
    ET.SubElement(asset, "texture", name="floor_tex", type="2d", builtin="checker",
                  rgb1="0.19 0.23 0.29", rgb2="0.25 0.29 0.35", width="512", height="512")
    ET.SubElement(asset, "material", name="floor_mat", texture="floor_tex",
                  texrepeat="1 1", texuniform="true", reflectance="0.1")
    world = model_xml.find("worldbody")
    ET.SubElement(world, "geom", name="replay_floor", type="plane", size="200 200 0.1",
                  material="floor_mat", group="1")
    ET.SubElement(world, "light", pos="0 -3 6", dir="0 0 -1", directional="true",
                  diffuse="0.8 0.8 0.8")
    model = mujoco.MjModel.from_xml_string(ET.tostring(model_xml, encoding="unicode"))
    if model.nq != 36:
        raise ValueError("Expected a free base and 29 joints")
    addresses = [model.joint(name).qposadr.item() for name in joint_names]
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance, camera.azimuth, camera.elevation = 2.8, 135, -17
    options = mujoco.MjvOption()
    # Only visual meshes and floor; omit duplicate collision meshes and sites.
    options.geomgroup[:] = 0
    options.geomgroup[1] = 1
    options.sitegroup[:] = 0
    slerp = Slerp(times, Rotation.from_quat(poses[:, 3:7]))
    landing_times = np.array([row["time_s"] for row in landings])
    duration = (end_time - times[0]) / args.speed
    # Include the exact endpoint, then hold it for one second.
    frame_times = np.minimum(times[0] + np.arange(int(np.ceil(duration * args.fps))
                            + args.fps + 1) * args.speed / args.fps, end_time)
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font = ImageFont.truetype(font_path, 22)
    title_font = ImageFont.truetype(font_path, 30)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Rendering {len(frame_times)} frames: {args.steps} verified landings, "
          f"{times[0]:.3f}..{end_time:.3f} s", flush=True)
    with mujoco.Renderer(model, height=720, width=1280) as renderer, imageio.get_writer(
            str(args.output), fps=args.fps, codec="libx264", quality=8,
            macro_block_size=1, pixelformat="yuv420p", ffmpeg_params=["-movflags", "+faststart"]) as writer:
        for i, t in enumerate(frame_times):
            pose = np.array([np.interp(t, times, poses[:, j]) for j in range(36)])
            pose[3:7] = slerp(t).as_quat()
            data.qpos[:3] = pose[:3]
            data.qpos[3:7] = pose[[6, 3, 4, 5]]
            data.qpos[addresses] = pose[7:]
            mujoco.mj_forward(model, data)  # FK only: never step or rerun dynamics.
            camera.lookat[:] = [pose[0], pose[1], 0.75]
            renderer.update_scene(data, camera=camera, scene_option=options)
            frame = Image.fromarray(renderer.render())
            draw = ImageDraw.Draw(frame)
            draw.rectangle((0, 0, 1280, 112), fill=(18, 25, 35))
            draw.text((28, 15), "G1 | Verified tracking prefix", font=title_font, fill="white")
            count = int(np.searchsorted(landing_times, t, side="right"))
            draw.text((28, 62), f"Measured trajectory replay   |   Landings {count:02d}/{args.steps}"
                      f"   |   t = {t:.3f} s   |   {args.speed:g}x", font=font, fill=(168, 219, 245))
            draw.rectangle((0, 680, 1280, 720), fill=(18, 25, 35))
            draw.text((28, 687), "Recorded Isaac Sim states | Visualization only | Prefix ends at last verified landing",
                      font=font, fill=(200, 208, 220))
            writer.append_data(np.asarray(frame))
            if i == len(frame_times) // 3:
                frame.save(args.output.with_suffix(".preview.png"))
            if i % args.fps == 0:
                print(f"Frame {i + 1}/{len(frame_times)}", flush=True)
    metadata = dict(recording=str(args.recording.resolve()), audit=str(audit_path.resolve()),
                    output=str(args.output.resolve()), verified_landings=args.steps,
                    measured_start_s=float(times[0]), measured_end_s=end_time,
                    fps=args.fps, frames=len(frame_times), speed=args.speed,
                    video_duration_s=len(frame_times) / args.fps,
                    rendering="MuJoCo forward kinematics of recorded Isaac Sim states; no physics stepping",
                    software_rendering=args.software, model=model_name)
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
