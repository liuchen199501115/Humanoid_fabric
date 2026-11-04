"""Generate a world-heading-conditioned motion PKL from two seed frames."""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from .data import load_clip, localize, worldize, matrix6d, matrix_quat, rotation6d
from .model import MotionDenoiser, sample, schedule


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--seed-motion', type=Path, required=True)
    p.add_argument('--seed-frame', type=int, default=0, help='Index after resampling to checkpoint fps')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seconds', type=float, default=5.)
    p.add_argument('--heading-deg', type=float, default=0., help='Fixed WORLD direction: 0=+X, 90=+Y')
    p.add_argument('--commit-frames', type=int, default=12)
    p.add_argument('--overlap-frames', type=int, default=None, help='Default 4 for inpainting checkpoints, 0 for legacy')
    p.add_argument('--sampling-steps', type=int, default=50)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    if not np.isfinite(args.seconds) or args.seconds <= 0 or not np.isfinite(args.heading_deg) or args.seed_frame < 0 or args.threads < 1:
        p.error('Invalid duration, heading, seed frame, or thread count')
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
    if ckpt.get('representation') == 'g1-paper99':
        p.error('For paper99 checkpoints use python -m motion_generation.paper.sample')
    model = MotionDenoiser(**ckpt['model_config']).to(args.device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    if not 1 <= args.commit_frames <= model.horizon:
        p.error('commit-frames must be within [1,horizon]')
    overlap = args.overlap_frames if args.overlap_frames is not None else (4 if model.inpainting else 0)
    if overlap < 0 or overlap > model.horizon-args.commit_frames or (overlap and not model.inpainting):
        p.error('Overlap requires inpainting checkpoint and 0 <= overlap <= horizon-commit-frames')
    tail = None
    fps = ckpt['fps']
    motion = load_clip(args.seed_motion, fps, ckpt['dof_names'])
    history = motion[args.seed_frame:args.seed_frame+2]
    if len(history) != 2:
        p.error('seed-frame must leave two available frames')
    mean, std = ckpt['mean'].to(args.device), ckpt['std'].to(args.device)
    alpha = schedule(ckpt['diffusion_steps'], args.device)
    angle = np.deg2rad(args.heading_deg)
    direction = np.array([np.cos(angle), np.sin(angle), 0.], dtype=np.float32)
    count = max(2, int(round(args.seconds*fps)))
    result = [history.copy()]
    remaining = count-2
    limits = ckpt['joint_limits'].cpu().numpy()
    clipped, total = 0, 0
    while remaining:
        local, origin, frame = localize(history)
        normalized = (torch.from_numpy(local).to(args.device)-mean)/std
        heading = torch.from_numpy((direction @ frame)[:2].copy()).to(args.device)
        known = mask = None
        if overlap and tail is not None:
            reference = tail[:overlap].copy()
            reference[:, :3] = (reference[:, :3]-origin) @ frame
            reference[:, 3:9] = rotation6d(frame.T @ matrix6d(reference[:, 3:9]))
            known = torch.zeros(1, model.horizon, 38, device=args.device)
            mask = torch.zeros(1, model.horizon, 1, device=args.device)
            known[0, :len(reference)] = (torch.from_numpy(reference).to(args.device)-mean)/std
            mask[:, :len(reference)] = 1
        generated = sample(model, normalized[None], heading[None], alpha, args.sampling_steps, known, mask)[0]
        generated = (generated*std+mean).cpu().numpy()
        if not np.isfinite(generated).all():
            raise RuntimeError('Nonfinite generated motion')
        generated = worldize(generated, origin, frame)
        n = min(args.commit_frames, remaining)
        committed = generated[:n].copy()
        clipped += int(((committed[:, 9:] < limits[:, 0]) | (committed[:, 9:] > limits[:, 1])).sum())
        total += n*29
        committed[:, 9:] = np.clip(committed[:, 9:], limits[:, 0], limits[:, 1])
        tail = generated[n:].copy()
        result.append(committed)
        history = np.concatenate((history, committed))[-2:]
        remaining -= n
    states = np.concatenate(result)
    quaternion = matrix_quat(matrix6d(states[:, 3:9]))
    for i in range(1, len(quaternion)):
        if quaternion[i-1] @ quaternion[i] < 0:
            quaternion[i] *= -1
    payload = dict(fps=fps, root_pos=states[:, :3], root_rot=quaternion,
                   dof_pos=states[:, 9:], dof_names=ckpt['dof_names'],
                   generation=dict(commit_frames=args.commit_frames, overlap_frames=overlap, seed_frames=2))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('wb') as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    report = dict(frames=len(states), fps=fps, duration_seconds=len(states)/fps,
                  checkpoint_step=ckpt['step'], world_heading_deg=args.heading_deg,
                  commit_frames=args.commit_frames, overlap_frames=overlap,
                  generated_joint_clamp_fraction=clipped/max(total, 1),
                  max_root_speed=float(np.linalg.norm(np.diff(states[:, :3], axis=0)*fps, axis=1).max()),
                  max_joint_speed=float(np.abs(np.diff(states[:, 9:], axis=0)*fps).max()),
                  root_height_range=[float(states[:, 2].min()), float(states[:, 2].max())],
                  physical_tracking_validated=False)
    args.output.with_suffix('.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
