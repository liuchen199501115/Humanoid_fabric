"""Evaluate generated future windows on held-out clips, without simulation."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from .data import MotionWindows
from .paper.data import PaperWindows
from .paper.kinematics import PaperFK, normalize_quaternion
from .model import MotionDenoiser, sample, schedule


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--windows', type=int, default=128)
    p.add_argument('--sampling-steps', type=int, default=50)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--threads', type=int, default=4)
    args = p.parse_args()
    if args.windows < 1 or args.threads < 1:
        p.error('windows and threads must be positive')
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
    model = MotionDenoiser(**ckpt['model_config']).to(args.device).eval()
    model.load_state_dict(ckpt['model'])
    with np.load(args.dataset, allow_pickle=False) as f:
        archive = dict(f)
    if archive['dof_names'].tolist() != ckpt['dof_names'] or float(archive['fps']) != ckpt['fps']:
        raise ValueError('Dataset and checkpoint joint order/fps differ')
    paper = ckpt.get('representation') == 'g1-paper99'
    dataset = (PaperWindows if paper else MotionWindows)(archive, 1, model.horizon)
    joints = slice(7,30) if paper else slice(9,None)
    indices = np.random.default_rng(args.seed).choice(len(dataset), min(args.windows, len(dataset)), replace=False)
    mean, std = ckpt['mean'].to(args.device), ckpt['std'].to(args.device)
    alpha = schedule(ckpt['diffusion_steps'], args.device)
    predictions, targets, histories, directions = [], [], [], []
    for offset in range(0, len(indices), 32):
        batch = [dataset[int(i)] for i in indices[offset:offset+32]]
        h, target, direction = [torch.from_numpy(np.stack(x)).to(args.device) for x in zip(*batch)]
        pred = sample(model, (h-mean)/std, direction, alpha, args.sampling_steps)*std+mean
        if paper:
            pred[...,3:7] = normalize_quaternion(pred[...,3:7])
            pred[...,3:7] = torch.where((pred[...,3:7]*target[...,3:7]).sum(-1,keepdim=True)<0,-pred[...,3:7],pred[...,3:7])
        predictions.append(pred.cpu().numpy())
        targets.append(target.cpu().numpy())
        histories.append(h.cpu().numpy())
        directions.append(direction.cpu().numpy())
    pred, target, history, direction = map(np.concatenate, (predictions, targets, histories, directions))
    if not np.isfinite(pred).all():
        raise RuntimeError('Nonfinite prediction')
    std = std.cpu().numpy()
    displacement = pred[:, -1, :2]-history[:, -1, :2]
    distance = np.linalg.norm(displacement, axis=-1)
    moving = distance > .02
    cosines = np.sum(displacement*direction, axis=-1)/np.maximum(distance, 1e-8)
    limits = ckpt['joint_limits'].cpu().numpy()
    report = dict(
        windows=len(indices), checkpoint_step=ckpt['step'], sampling_steps=args.sampling_steps,
        normalized_future_mse=float(np.mean(((pred-target)/std)**2)),
        hold_last_frame_mse=float(np.mean(((history[:, -1:]-target)/std)**2)),
        joint_rmse_radians=float(np.sqrt(np.mean((pred[:, :, joints]-target[:, :, joints])**2))),
        root_position_rmse_m=float(np.sqrt(np.mean((pred[:, :, :3]-target[:, :, :3])**2))),
        moving_prediction_fraction=float(moving.mean()),
        mean_heading_cosine_moving=float(cosines[moving].mean()) if moving.any() else None,
        mean_seam_joint_speed_rad_s=float(np.abs(pred[:, 0, joints]-history[:, -1, joints]).mean()*ckpt['fps']),
        joint_limit_violation_fraction=float(((pred[:, :, joints] < limits[:, 0]) | (pred[:, :, joints] > limits[:, 1])).mean()),
        physical_tracking_validated=False,
        note='Single stochastic sample per window; multiple futures can be valid. Heading cosine excludes <2cm displacement.'
    )
    if paper:
        with torch.no_grad():
            links = PaperFK()(torch.from_numpy(pred)).numpy()
        report['link_head_fk_rmse_m'] = float(np.sqrt(np.mean((pred[...,30:].reshape(*pred.shape[:2],23,3)-links)**2)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
