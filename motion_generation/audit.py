"""Kinematic audit of generated PKL; no dynamic feasibility claim."""
import argparse
import json
import pickle
from pathlib import Path

import mujoco
import numpy as np
from .data import quat_matrix


def stats(x):
    x = np.asarray(x)
    return dict(mean=float(x.mean()), p95=float(np.percentile(x, 95)), maximum=float(x.max())) if x.size else None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--motion', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--commit-frames', type=int, default=12, help='Assumed sampling stride; old PKLs do not record it')
    args = p.parse_args()
    if args.commit_frames < 1:
        p.error('commit-frames must be positive')
    with args.motion.open('rb') as f:
        motion = pickle.load(f)
    fps = float(motion['fps'])
    root, quat, joints = [np.asarray(motion[k], dtype=float) for k in ('root_pos', 'root_rot', 'dof_pos')]
    if len(root) < 3 or not np.isfinite(fps) or fps <= 0 or not all(np.isfinite(x).all() for x in (root, quat, joints)):
        raise ValueError('Need at least three finite frames and positive fps')
    rotation = quat_matrix(quat)
    model = mujoco.MjModel.from_xml_path(str(Path(__file__).resolve().parents[1]/'retarget/assets/g1_description/g1_29dof.xml'))
    data = mujoco.MjData(model)
    names = motion['dof_names']
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in names]
    if any(i < 0 for i in ids):
        raise ValueError('Unknown joint names')
    addresses = model.jnt_qposadr[ids]
    limits = model.jnt_range[ids]
    feet = []
    for side in ('left', 'right'):
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f'{side}_ankle_roll_link')
        geoms = np.flatnonzero((model.geom_bodyid == body) & (model.geom_type == mujoco.mjtGeom.mjGEOM_SPHERE))
        if len(geoms) != 4:
            raise ValueError('Expected four foot sole spheres')
        feet.append(geoms)
    feet = np.asarray(feet)
    points = []
    for pos, q, joint in zip(root, quat, joints):
        data.qpos[:3] = pos
        data.qpos[3:7] = q[[3, 0, 1, 2]]/np.linalg.norm(q)
        data.qpos[addresses] = joint
        mujoco.mj_forward(model, data)
        points.append(data.geom_xpos[feet].copy())
    points = np.asarray(points)
    height = points[..., 2]-model.geom_size[feet, 0]
    joint_velocity = np.diff(joints, axis=0)*fps
    root_speed = np.linalg.norm(np.diff(root, axis=0)*fps, axis=1)
    delta = rotation[:-1].transpose(0, 2, 1) @ rotation[1:]
    angular_speed = np.arccos(np.clip((np.trace(delta, axis1=1, axis2=2)-1)/2, -1, 1))*fps
    transitions = np.arange(1, len(root))
    seam = (transitions >= 2) & ((transitions-2) % args.commit_frames == 0)
    inside = (transitions >= 2) & ~seam
    max_joint_speed = np.abs(joint_velocity).max(axis=1)
    acceleration = np.abs(np.diff(joint_velocity, axis=0)*fps).max(axis=1)
    near_ground = (np.abs(height[:-1]) <= .02) & (np.abs(height[1:]) <= .02)
    vertical_speed = np.abs(np.diff(points[..., 2], axis=0)*fps)
    candidate = near_ground & (vertical_speed < .1)
    horizontal_speed = np.linalg.norm(np.diff(points[..., :2], axis=0)*fps, axis=-1)
    report = dict(source=str(args.motion), frames=len(root), fps=fps,
        assumed_commit_frames=args.commit_frames,
        boundary_note='Assumes two seed frames; tests initial seed seam and each committed chunk, not a 25-frame period.',
        root_height_m=[float(root[:, 2].min()), float(root[:, 2].max())],
        root_tilt_degrees=stats(np.rad2deg(np.arccos(np.clip(rotation[:, 2, 2], -1, 1)))),
        joint_limit_violation_fraction=float(((joints < limits[:, 0]-1e-5) | (joints > limits[:, 1]+1e-5)).mean()),
        joint_speed_rad_s=stats(max_joint_speed), root_speed_m_s=stats(root_speed),
        root_angular_speed_rad_s=stats(angular_speed), max_joint_acceleration_rad_s2=stats(acceleration),
        seam_joint_speed_rad_s=stats(max_joint_speed[seam]), inside_joint_speed_rad_s=stats(max_joint_speed[inside]),
        seam_root_speed_m_s=stats(root_speed[seam]), inside_root_speed_m_s=stats(root_speed[inside]),
        seam_joint_acceleration_rad_s2=stats(acceleration[seam[1:]]),
        inside_joint_acceleration_rad_s2=stats(acceleration[inside[1:]]),
        worst_joint_transition_frame=int(np.argmax(max_joint_speed)+1),
        worst_joint_transition_time_s=float((np.argmax(max_joint_speed)+1)/fps),
        foot_checks={}, physical_tracking_validated=False)
    for i, side in enumerate(('left', 'right')):
        selected = horizontal_speed[:, i][candidate[:, i]]
        report['foot_checks'][side] = dict(min_sole_height_m=float(height[:, i].min()),
            frames_penetrating_over_1cm_fraction=float((height[:, i].min(axis=-1) < -.01).mean()),
            candidate_point_intervals=int(candidate[:, i].sum()),
            candidate_horizontal_speed_m_s=stats(selected),
            candidate_speed_over_0_1m_s_fraction=float((selected > .1).mean()) if selected.size else None)
    report['foot_check_note'] = 'Heuristic only: same sole sphere within +/-2cm ground on both frames and vertical speed <0.1m/s. No contact force; includes possible near-ground swing.'
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'report.json').write_text(json.dumps(report, indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    time = np.arange(len(root))/fps
    fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True)
    axes[0].plot(time, root[:, 2], label='Root height (m)')
    axes[1].plot(time[1:], max_joint_speed, label='Max joint speed (rad/s)')
    axes[1].scatter(time[1:][seam], max_joint_speed[seam], s=10, color='red', label='Assumed chunk seam')
    axes[2].plot(time[2:], acceleration, label='Max joint acceleration (rad/s²)')
    for i, side in enumerate(('left', 'right')):
        axes[3].plot(time, height[:, i].min(axis=-1), label=f'{side} lowest sole sphere (m)')
    axes[3].axhline(0, color='black', linewidth=.8)
    for ax in axes:
        ax.legend(loc='upper right'); ax.grid(alpha=.25)
    axes[-1].set_xlabel('Time (s)')
    fig.tight_layout()
    fig.savefig(args.output/'diagnostics.png', dpi=150)
    plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
