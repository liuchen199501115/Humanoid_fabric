"""Inspect source stance FK; this is not a dynamics or tracking certificate."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import warp as wp
from scipy.spatial.transform import Rotation

from fabrics_sim.fabrics.g1_pose_fabric import G1PoseFabric
from test_g1_two_tasks import load_motion_trajectory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--motion', type=Path, default=Path('retarget/lafan/walk1_g1.pkl'))
    parser.add_argument('--record', type=Path, default=Path('acc_data/walk1_settled_unload_20261021.npz'))
    parser.add_argument('--start', type=int, default=114)
    parser.add_argument('--end', type=int, default=129)
    parser.add_argument('--support', choices=['left', 'right'], default='left')
    parser.add_argument('--output', type=Path, default=Path('acc_data/walk1_stance_motion_20261025.json'))
    args = parser.parse_args()
    wp.config.kernel_cache_dir = '/tmp/g1_warp_cache'
    wp.init()
    torch.set_grad_enabled(False)
    source, fps = load_motion_trajectory(args.motion)
    if not 0 <= args.start < args.end < len(source):
        parser.error('Require 0 <= start < end < motion length')
    with np.load(args.record) as data:
        shift = float(data['motion_ground_shift'])
        initial_q = torch.tensor(data['history_q'][-1])
    source[:, 2] += shift
    fabric = G1PoseFabric(1, 'cpu', robot_name='g1_29dof_rev_1_0', task_profile='dynamics-aware')
    model = fabric.dynamics_tasks.model
    index = model.kinematics.get_link_index(args.support+'_ankle_roll_link')
    anchor = model.snapshot(initial_q)['transforms'][0, index].cpu().numpy()
    # Keep model batch size one, as in online control.
    poses = np.stack([model.snapshot(source[i:i+1])['transforms'][0, index].cpu().numpy()
                      for i in range(args.start, args.end+1)])
    rotation = Rotation.from_quat(poses[:, 3:])
    relative = (rotation * rotation[0].inv()).as_rotvec()
    anchor_rotation = (rotation * Rotation.from_quat(anchor[3:]).inv()).as_rotvec()
    corners = np.array([[-.05, -.025, -.035], [-.05, .025, -.035],
                        [.12, -.025, -.035], [.12, .025, -.035]])
    world = poses[:, None, :3] + np.einsum('nij,kj->nki', rotation.as_matrix(), corners)
    times = np.arange(len(poses))/fps
    speed = np.linalg.norm(np.gradient(world, times, axis=0), axis=-1)
    report = dict(scope='Source FK only; geometric corner motion does not prove loaded contact or slip',
                  simulation_validated=False, motion=str(args.motion), record=str(args.record),
                  support=args.support, frames=list(range(args.start, args.end+1)), fps=fps,
                  ground_shift_m=shift, quaternion_order='xyzw',
                  speed_timebase='Original PKL fps; not the retimed 2-second candidate',
                  max_ankle_translation_from_first_m=float(np.linalg.norm(poses[:, :3]-poses[0, :3], axis=-1).max()),
                  max_rotation_from_first_rad=float(np.linalg.norm(relative, axis=-1).max()),
                  max_translation_from_measured_anchor_m=float(np.linalg.norm(poses[:, :3]-anchor[:3], axis=-1).max()),
                  max_rotation_from_measured_anchor_rad=float(np.linalg.norm(anchor_rotation, axis=-1).max()),
                  corner_height_range_m=[float(world[..., 2].min()), float(world[..., 2].max())],
                  max_corner_speed_m_s=float(speed.max()),
                  ankle_poses=poses.tolist(), sole_corner_positions_m=world.tolist(),
                  corner_order=['heel_y_negative', 'heel_y_positive', 'toe_y_negative', 'toe_y_positive'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    frames = report['frames']
    for k, axis in enumerate('xyz'):
        axes[0].plot(frames, poses[:, k]-poses[0, k], label=axis)
        axes[1].plot(frames, relative[:, k], label=axis)
    for k, name in enumerate(report['corner_order']):
        axes[2].plot(frames, world[:, k, 2], label=name)
    for ax, label in zip(axes, ['Ankle displacement [m]', 'Relative rotation vector [rad]', 'Sole corner height [m]']):
        ax.set_ylabel(label)
        ax.legend()
        ax.grid(True)
    axes[2].axhline(0, color='black', linestyle=':')
    axes[2].set_xlabel('Original PKL frame')
    fig.suptitle(f'{args.support} source stance FK (not measured contact)')
    fig.tight_layout()
    fig.savefig(args.output.with_suffix('.png'))
    plt.close(fig)
    print(json.dumps({k: v for k, v in report.items() if k not in ('ankle_poses', 'sole_corner_positions_m')}, indent=2))


if __name__ == '__main__':
    main()
