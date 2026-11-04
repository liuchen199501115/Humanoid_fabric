"""Train heading-conditioned diffusion on prepared G1 motion windows."""
# Bootstrap before importing native Torch/NumPy extensions.
if __name__ == '__main__':
    from .runtime import maybe_reexec
    maybe_reexec()

import sys
import platform
import faulthandler
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from .data import MotionWindows
from .model import MotionDenoiser, schedule
from .losses import MotionGeometryLoss
from .paper.data import PaperWindows
from .paper.losses import PaperMotionLoss, objective as paper_objective


def objective(model, batch, mean, std, alpha, velocity_weight, condition_noise=0., amp_dtype=None, geometry=None, geometry_weights=None):
    if isinstance(geometry, PaperMotionLoss):
        return paper_objective(model, batch, mean, std, alpha, velocity_weight, condition_noise, amp_dtype, geometry, geometry_weights)
    history, future, heading = [x.to(mean.device) for x in batch[:3]]
    physical_history = history
    history = (history-mean)/std
    target = (future-mean)/std
    if condition_noise:
        history = history+torch.randn_like(history)*condition_noise
    step = torch.randint(len(alpha), (len(target),), device=mean.device)
    a = alpha[step, None, None]
    noisy = a.sqrt()*target+(1-a).sqrt()*torch.randn_like(target)
    mask = torch.zeros_like(target[..., :1])
    if getattr(model, 'inpainting', False):
        count = torch.randint(0, min(13, target.shape[1]-1)+1, (len(target), 1, 1), device=mean.device)
        mask = (torch.arange(target.shape[1], device=mean.device)[None, :, None] < count).to(target.dtype)
    with torch.autocast(device_type=mean.device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
        if getattr(model, 'inpainting', False):
            predicted = model(noisy, step, history, heading, target*mask, mask)
        else:
            predicted = model(noisy, step, history, heading)
    predicted = predicted.float()
    unknown = 1-mask
    reconstruction = ((predicted-target).square()*unknown).sum()/(unknown.sum()*target.shape[-1]).clamp_min(1)
    predicted = predicted*unknown+target*mask
    # Frame-increment loss includes the seam to the last history frame.
    pred_velocity = torch.diff(torch.cat((history[:, -1:], predicted), 1), dim=1)
    ref_velocity = torch.diff(torch.cat((history[:, -1:], target), 1), dim=1)
    velocity = (pred_velocity-ref_velocity).square().mean()
    loss = reconstruction+velocity_weight*velocity
    if geometry is not None:
        terms = geometry(predicted*std+mean, future, physical_history, batch[3].to(mean.device), mask)
        loss = loss + sum(geometry_weights[k]*v for k, v in terms.items())
        model.last_geometry_losses = {k: float(v.detach()) for k, v in terms.items()}
    return loss, reconstruction.detach(), velocity.detach()


def atomic_save(checkpoint, path):
    temporary = path.with_suffix(".pt.tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def main():
    faulthandler.enable(all_threads=True)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--profile', choices=['legacy', 'mdm', 'paper'], default='legacy')
    p.add_argument('--resume', type=Path, help='Resume an epoch-boundary checkpoint; duration is total, not additional')
    duration = p.add_mutually_exclusive_group()
    duration.add_argument('--steps', type=int)
    duration.add_argument('--epochs', type=int)
    p.add_argument('--checkpoint-every-epochs', type=int, default=10)
    p.add_argument('--amp', choices=['off', 'bf16', 'fp16'], default='off')
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--ffn-dim', type=int, default=None)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--horizon', type=int, default=25)
    p.add_argument('--stride', type=int, default=5)
    p.add_argument('--width', type=int, default=128)
    p.add_argument('--layers', type=int, default=4)
    p.add_argument('--heads', type=int, default=4)
    p.add_argument('--diffusion-steps', type=int, default=1000)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--velocity-weight', type=float, default=.1)
    for name, weight in {'velocity': .01, 'acceleration': .01, 'boundary': .02, 'skating': .1, 'contact': 10., 'penetration': 10., 'fk': 1., 'limits': 1.}.items():
        p.add_argument(f'--geometry-{name}', type=float, default=weight)
    p.add_argument('--condition-noise', type=float, default=.01)
    p.add_argument('--eval-every', type=int, default=500)
    p.add_argument('--stats-windows', type=int, default=4096)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    if args.steps is None and args.epochs is None:
        args.steps = 10000
    if min(args.steps if args.steps is not None else args.epochs, args.checkpoint_every_epochs, args.batch_size, args.eval_every, args.stats_windows, args.threads, args.layers, args.width, args.heads) < 1:
        p.error('Counts must be positive')
    if args.lr <= 0 or args.velocity_weight < 0 or args.condition_noise < 0:
        p.error('Invalid learning rate or loss/noise weights')
    if args.weight_decay < 0 or not np.isfinite(args.weight_decay) or (args.ffn_dim is not None and args.ffn_dim < 1):
        p.error('Invalid weight decay or FFN dimension')
    if args.amp != 'off' and args.device != 'cuda':
        p.error('AMP training requires --device cuda')
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)
    amp_dtype = {'off': None, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.amp]
    if args.amp == 'bf16' and not torch.cuda.is_bf16_supported():
        p.error('CUDA device does not support BF16; use --amp fp16')
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp == 'fp16')
    with np.load(args.dataset, allow_pickle=False) as f:
        archive = dict(f)
    if (str(archive.get('representation', '')) == 'g1-paper99') != (args.profile == 'paper'):
        p.error('paper99 datasets require --profile paper; other profiles require 38D data')
    windows = PaperWindows if args.profile == 'paper' else MotionWindows
    train = windows(archive, 0, args.horizon, args.stride, with_contacts=args.profile == 'mdm')
    valid = windows(archive, 1, args.horizon, args.stride, with_contacts=args.profile == 'mdm')
    geometry_weights = {k.removeprefix('geometry_'): v for k,v in vars(args).items() if k.startswith('geometry_')}
    if any(not np.isfinite(v) or v < 0 for v in geometry_weights.values()):
        p.error('Geometry weights must be finite and nonnegative')
    geometry = MotionGeometryLoss(archive['dof_names'].tolist(), archive['joint_limits'], float(archive['fps']), device) if args.profile == 'mdm' else None
    if args.profile == 'paper':
        geometry = PaperMotionLoss(device)
        if archive['dof_names'].tolist() != geometry.fk.names or archive['body_names'].tolist() != geometry.fk.links:
            p.error('Paper joint/body ordering must match the 23-DOF URDF')
    rng = np.random.default_rng(args.seed)
    indices = rng.choice(len(train), min(args.stats_windows, len(train)), replace=False)
    observations = np.concatenate([np.concatenate(train[int(i)][:2]) for i in indices])
    mean = torch.tensor(observations.mean(0), device=device)
    std = torch.tensor(np.maximum(observations.std(0), .05), device=device)
    config = dict(dim=99 if args.profile == 'paper' else 38, horizon=args.horizon, width=args.width, layers=args.layers, heads=args.heads, ffn_dim=args.ffn_dim)
    if args.profile == 'mdm':
        config['inpainting'] = True
    model = MotionDenoiser(**config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    alpha = schedule(args.diffusion_steps, device)
    loader = DataLoader(train, batch_size=args.batch_size, shuffle=True)
    steps_per_epoch = len(loader)
    total_steps = args.epochs*steps_per_epoch if args.epochs is not None else args.steps
    # Cover the validation corpus instead of only the first long clip.
    validation_indices = rng.choice(len(valid), min(512, len(valid)), replace=False).tolist()
    validation = DataLoader(Subset(valid, validation_indices), batch_size=args.batch_size, shuffle=False)
    start_step = 0
    resume_best = float('inf')
    if args.resume:
        saved = torch.load(args.resume, map_location=device, weights_only=True)
        if saved['model_config'] != config or saved['diffusion_steps'] != args.diffusion_steps:
            p.error('Resume requires matching model and diffusion configuration')
        old = saved['train_config']
        for key in ('batch_size', 'stride', 'seed', 'stats_windows'):
            if old[key] != getattr(args, key):
                p.error(f'Resume requires matching {key}')
        if saved['dof_names'] != archive['dof_names'].tolist() or saved['fps'] != float(archive['fps']) or old['train_windows'] != len(train):
            p.error('Resume dataset metadata mismatch')
        start_step = int(saved['step'])
        if start_step % steps_per_epoch or start_step >= total_steps:
            p.error('Resume requires an epoch boundary and a larger total training duration')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        if args.amp == 'fp16' and saved.get('scaler'):
            scaler.load_state_dict(saved['scaler'])
        mean, std = saved['mean'].to(device), saved['std'].to(device)
        resume_best = saved.get('best_validation_loss', saved['validation_loss'])
        print(f'Resuming step {start_step}, epoch {start_step//steps_per_epoch}; data/noise RNG is restarted, not bit-exact replay.', flush=True)
    iterator = iter(loader)
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    metadata.update(train_windows=len(train), validation_windows=len(valid), parameters=sum(p.numel() for p in model.parameters()),
                    steps_per_epoch=steps_per_epoch, total_steps=total_steps, fps=float(archive['fps']),
                    motion_dim=config['dim'], terrain_condition=False, past_frames=2, effective_ffn_dim=args.ffn_dim or args.width*4,
                    runtime=dict(python=sys.version, executable=sys.executable, torch=str(torch.__version__),
                                 cuda=torch.version.cuda, platform=platform.platform()))
    if args.profile == 'paper':
        metadata['active_loss_weights'] = dict(reconstruction=1., velocity=args.velocity_weight,
                                             joint_consistency=args.geometry_fk, flat_penetration=args.geometry_penetration)
    (args.output/'config.json').write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata), flush=True)
    started = time.monotonic()
    best = resume_best
    for step in range(start_step+1, total_steps+1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        model.train()
        loss, reconstruction, velocity = objective(model, batch, mean, std, alpha, args.velocity_weight, args.condition_noise, amp_dtype, geometry, geometry_weights)
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite training loss')
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        scaler.step(optimizer)
        scaler.update()
        epoch = step // steps_per_epoch
        epoch_end = step % steps_per_epoch == 0
        evaluate_now = epoch_end if args.epochs is not None else step % args.eval_every == 0
        snapshot = epoch_end and epoch % args.checkpoint_every_epochs == 0
        if step == 1 or evaluate_now or snapshot or step == total_steps:
            model.eval()
            values = []
            # Fixed noise/steps for comparable validation; preserve training RNG.
            with torch.random.fork_rng(devices=[torch.cuda.current_device()] if device.type == 'cuda' else []):
                torch.manual_seed(args.seed+100)
                with torch.no_grad():
                    for batch in validation:
                        val, _, _ = objective(model, batch, mean, std, alpha, args.velocity_weight, amp_dtype=amp_dtype, geometry=geometry, geometry_weights=geometry_weights)
                        values.append((val.item(), len(batch[0])))
            val_loss = float(sum(v*n for v, n in values)/sum(n for _, n in values))
            metrics = dict(step=step, epoch=step/steps_per_epoch, train_loss=loss.item(), reconstruction=reconstruction.item(), velocity=velocity.item(),
                           validation_loss=val_loss, seconds=time.monotonic()-started)
            if geometry is not None:
                metrics['geometry_last_validation_batch'] = model.last_geometry_losses
            print(json.dumps(metrics), flush=True)
            with (args.output/'metrics.jsonl').open('a') as f:
                f.write(json.dumps(metrics)+'\n')
            checkpoint = dict(model=model.state_dict(), model_config=config, mean=mean.cpu(), std=std.cpu(),
                              diffusion_steps=args.diffusion_steps, fps=float(archive['fps']),
                              dof_names=archive['dof_names'].tolist(), joint_limits=torch.from_numpy(archive['joint_limits']),
                              step=step, epoch=epoch, validation_loss=val_loss, train_config=metadata,
                              scaler=scaler.state_dict(), best_validation_loss=min(best, val_loss),
                              optimizer=optimizer.state_dict(), format_version=3 if args.profile == 'paper' else (2 if args.profile == 'mdm' else 1))
            if args.profile == 'paper':
                checkpoint.update(representation='g1-paper99', body_names=geometry.fk.links)
            atomic_save(checkpoint, args.output/'last.pt')
            if snapshot:
                atomic_save(checkpoint, args.output/f'epoch_{epoch:04d}.pt')
            if val_loss < best:
                best = val_loss
                atomic_save(checkpoint, args.output/'best.pt')


if __name__ == '__main__':
    main()
