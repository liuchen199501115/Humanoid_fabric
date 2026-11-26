"""Loop a recorded G1 measured-state history in Isaac Sim, without rerunning control."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path)
    parser.add_argument("--speed", type=float, default=1.0)
    args = parser.parse_args()
    if not np.isfinite(args.speed) or args.speed <= 0:
        parser.error("--speed must be finite and positive")
    with np.load(args.recording, allow_pickle=False) as data:
        poses = np.asarray(data["history_q"], dtype=np.float32)
        times = np.asarray(data["history_time"], dtype=np.float64)
    if poses.ndim == 3 and poses.shape[1] == 1:
        poses = poses[:, 0]
    if (poses.shape != (len(times), 36) or len(times) < 2
            or not np.isfinite(poses).all() or not np.isfinite(times).all()
            or not np.all(np.diff(times) > 0)):
        raise ValueError("Expected finite history_q [N,1,36] and increasing history_time [N].")
    times = times - times[0]

    from test_g1_two_tasks import G1_JOINT_NAMES, G1_USD_PATH
    from isaacsim import SimulationApp

    if not G1_USD_PATH.is_file():
        raise FileNotFoundError(G1_USD_PATH)
    app = SimulationApp({"headless": False, "width": 1280, "height": 800})
    try:
        from isaacsim.core.api import World
        from isaacsim.core.api.objects import GroundPlane
        from isaacsim.core.prims import Articulation
        from isaacsim.core.utils.stage import add_reference_to_stage
        from isaacsim.core.utils.viewports import set_camera_view
        import omni.ui as ui
        import omni.usd
        from omni.physx import get_physx_interface
        from pxr import UsdLux

        world = World(physics_dt=1 / 60, rendering_dt=1 / 60)
        world.scene.add(GroundPlane(prim_path="/World/GroundPlane", size=30.0))
        add_reference_to_stage(str(G1_USD_PATH), "/World/G1")
        robot = world.scene.add(Articulation(prim_paths_expr="/World/G1", name="g1"))
        light = UsdLux.DomeLight.Define(omni.usd.get_context().get_stage(), "/World/Light")
        light.CreateIntensityAttr(1000.0)
        world.reset()
        physx = get_physx_interface()
        omni.usd.get_context().get_selection().clear_selected_prim_paths()
        order = np.asarray([G1_JOINT_NAMES.index(name) for name in robot.dof_names])
        zeros = np.zeros((1, len(order)), dtype=np.float32)
        robot.set_gains(kps=zeros, kds=zeros)
        center = (poses[:, :3].min(axis=0) + poses[:, :3].max(axis=0)) / 2
        set_camera_view(eye=center + [3.0, -6.0, 3.0], target=center,
                        camera_prim_path="/OmniverseKit_Persp")
        controls = ui.Window("Recorded tracking playback", width=480, height=145)
        state = {"paused": False, "restart": False}

        def toggle_pause():
            state["paused"] = not state["paused"]

        def restart():
            state["restart"] = True

        with controls.frame:
            with ui.VStack(spacing=6):
                ui.Label(args.recording.name)
                ui.Label("Measured trajectory replay (no controller or physics stepping)")
                status = ui.Label("")
                with ui.HStack():
                    ui.Button("Pause / Resume", clicked_fn=toggle_pause)
                    ui.Button("Restart", clicked_fn=restart)
        play_time = 0.0
        previous = time.monotonic()
        loop_count = 0
        print(f"[replay] {args.recording.resolve()} | {len(poses)} poses | "
              f"{times[-1]:.3f}s | speed={args.speed:g}x", flush=True)
        while app.is_running():
            started = time.monotonic()
            if not state["paused"]:
                play_time += (started - previous) * args.speed
            previous = started
            if state["restart"]:
                play_time = 0.0
                state["restart"] = False
            # Hold the final recorded pose for a second before each repeat.
            if play_time >= times[-1] + args.speed:
                play_time = 0.0
                loop_count += 1
                print(f"[replay] loop {loop_count} complete", flush=True)
            index = min(np.searchsorted(times, play_time, side="right") - 1, len(times) - 1)
            pose = poses[max(index, 0)]
            robot.set_world_poses(positions=pose[None, :3],
                                  orientations=pose[None, [6, 3, 4, 5]])
            robot.set_joint_positions(pose[None, 7:][:, order])
            robot.set_joint_velocities(zeros)
            robot.set_velocities(np.zeros((1, 6), dtype=np.float32))
            world.physics_sim_view.update_articulations_kinematic()
            physx.update_transformations(False, True)
            set_camera_view(eye=pose[:3] + [2.4, -3.2, 1.5],
                            target=pose[:3] + [0, 0, -0.1],
                            camera_prim_path="/OmniverseKit_Persp")
            status.text = (f"{min(play_time, times[-1]):.2f} / {times[-1]:.2f} s   "
                           f"{args.speed:g}x   {'Paused' if state['paused'] else 'Looping'}")
            world.render()
            time.sleep(max(0, 1 / 60 - (time.monotonic() - started)))
    finally:
        app.close()


if __name__ == "__main__":
    main()
