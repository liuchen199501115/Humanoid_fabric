"""Preview raw posed_joints as points, without assuming a skeleton topology."""
import pickle
import time
from pathlib import Path
import numpy as np


def load_points(path, up_axis):
    with Path(path).open('rb') as handle:
        payload = pickle.load(handle)
    points = np.asarray(payload['posed_joints'], dtype=np.float64)
    if points.ndim != 3 or points.shape[-1] != 3 or min(points.shape[:2]) < 1 or not np.isfinite(points).all():
        raise ValueError('posed_joints must contain finite [frames, joints, 3] positions')
    if up_axis == 'y':
        # Right-handed Y-up -> MuJoCo Z-up: (x, y, z) -> (x, -z, y).
        points = points[..., [0, 2, 1]].copy()
        points[..., 1] *= -1
    elif up_axis != 'z':
        raise ValueError('up_axis must be y or z')
    return points, payload.get('fps')


def run_points(args):
    import mujoco
    from view_motion_follow import _resolve_video_fps
    points, stored_fps = load_points(args.motion_file, args.up_axis)
    if args.fps is None and stored_fps is None:
        raise ValueError('Raw point motion has no FPS metadata; provide --fps')
    fps = _resolve_video_fps(stored_fps, args.fps)
    bodies = ''.join(f'<body mocap="true" name="joint_{j}"><geom type="sphere" size="0.015" rgba="{0.2+0.6*j/max(1,points.shape[1]-1)} 0.5 0.9 1" contype="0" conaffinity="0"/></body>' for j in range(points.shape[1]))
    xml = f'<mujoco><visual><global offwidth="{args.width}" offheight="{args.height}"/></visual><worldbody><light pos="0 0 5"/><geom type="plane" size="200 200 .1" rgba=".22 .24 .27 1"/>{bodies}</worldbody></mujoco>'
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    def update(i, camera):
        data.mocap_pos[:] = points[i]
        mujoco.mj_forward(model, data)
        camera.lookat[:] = points[i].mean(0)
        camera.distance = args.camera_distance
        camera.azimuth = args.camera_azimuth
        camera.elevation = args.camera_elevation
    if args.no_video:
        import mujoco.viewer
        with mujoco.viewer.launch_passive(model, data) as viewer:
            started = time.monotonic()
            count = len(points)*max(1,args.loops)
            while viewer.is_running():
                frame = int((time.monotonic()-started)*fps)
                if frame >= count: break
                with viewer.lock(): update(frame % len(points), viewer.cam)
                viewer.sync()
                time.sleep(min(1/fps, .01))
    else:
        import imageio.v2 as imageio
        output = Path(args.video_path) if args.video_path else Path(args.motion_file).with_name(Path(args.motion_file).stem+'_points.mp4')
        output.parent.mkdir(parents=True,exist_ok=True)
        camera = mujoco.MjvCamera()
        with mujoco.Renderer(model,height=args.height,width=args.width) as renderer:
            with imageio.get_writer(str(output),fps=fps,macro_block_size=1) as writer:
                for i in range(len(points)*max(1,args.loops)):
                    update(i % len(points),camera)
                    renderer.update_scene(data,camera=camera)
                    writer.append_data(renderer.render())
        print(f'[done] {output}: {len(points)} frames, {fps:g} FPS, {len(points)/fps:g}s per loop',flush=True)
