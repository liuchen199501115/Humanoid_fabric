"""Generate paper99 motion; export 29-DOF PKL with six omitted joints fixed at zero."""
import argparse
import json
import pickle
from pathlib import Path
import numpy as np
import torch
from ..data import joint_spec, load_clip
from ..model import MotionDenoiser, sample, schedule
from .data import encode_legacy, localize, worldize, canonicalize_quaternions
from .kinematics import PaperFK, normalize_quaternion


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--seed-motion', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seconds', type=float, default=5.)
    p.add_argument('--heading-deg', type=float, default=0., help='World direction: 0=+X, 90=+Y')
    p.add_argument('--commit-frames', type=int, default=12)
    p.add_argument('--sampling-steps', type=int, default=50)
    p.add_argument('--device', choices=['cpu','cuda'], default='cpu')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    if not np.isfinite(args.seconds) or args.seconds<=0 or not np.isfinite(args.heading_deg) or not 1<=args.commit_frames<=25 or args.sampling_steps<1:
        p.error('Invalid duration, heading, commit frames or sampling steps')
    torch.set_num_threads(4); torch.manual_seed(args.seed)
    saved = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
    if saved.get('representation')!='g1-paper99': p.error('Requires a paper99 checkpoint')
    fk = PaperFK()
    if saved['dof_names']!=fk.names or saved['body_names']!=fk.links: p.error('Checkpoint robot ordering mismatch')
    names,_ = joint_spec()
    seed = load_clip(args.seed_motion, saved['fps'], names)
    if len(seed)<2:p.error('Seed must contain two frames')
    history = encode_legacy(seed[:2],names,fk)
    fk = fk.to(args.device)
    model = MotionDenoiser(**saved['model_config']).to(args.device)
    model.load_state_dict(saved['model']); model.eval()
    mean,std = saved['mean'].to(args.device),saved['std'].to(args.device)
    alpha = schedule(saved['diffusion_steps'],args.device)
    angle = np.deg2rad(args.heading_deg)
    direction = np.array([np.cos(angle),np.sin(angle),0],dtype=np.float32)
    result=[history.copy()]; remaining=max(2,round(args.seconds*saved['fps']))-2
    residuals=[]
    with torch.no_grad():
        while remaining:
            local,origin,frame,yaw = localize(history)
            condition=(torch.from_numpy(local).to(args.device)-mean)/std
            heading=torch.from_numpy((direction@frame)[:2].copy()).to(args.device)
            generated=sample(model,condition[None],heading[None],alpha,args.sampling_steps)[0]*std+mean
            if not torch.isfinite(generated).all():raise RuntimeError('Nonfinite generated motion')
            generated[:,3:7]=normalize_quaternion(generated[:,3:7])
            generated[:,7:30]=generated[:,7:30].clamp(fk.limits[:,0],fk.limits[:,1])
            links=fk(generated).flatten(-2)
            residuals.append(float((generated[:,30:]-links).square().mean().sqrt()))
            # Rollout history must describe one consistent robot configuration.
            generated[:,30:]=links
            world=worldize(generated.cpu().numpy(),origin,frame,yaw)
            n=min(args.commit_frames,remaining); committed=world[:n]
            result.append(committed);history=np.concatenate((history,committed))[-2:];remaining-=n
    states=np.concatenate(result)
    angles=np.zeros((len(states),len(names)),dtype=np.float32)
    for j,name in enumerate(fk.names):angles[:,names.index(name)]=states[:,7+j]
    payload=dict(fps=saved['fps'],root_pos=states[:,:3],root_rot=canonicalize_quaternions(states[:,3:7]),
                 dof_pos=angles,dof_names=names,
                 generation=dict(representation='g1-paper99',commit_frames=args.commit_frames,seed_frames=2,
                                 fixed_zero_joints=[n for n in names if n not in fk.names],terrain_condition=False))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('wb') as f:pickle.dump(payload,f,pickle.HIGHEST_PROTOCOL)
    report=dict(frames=len(states),fps=saved['fps'],world_heading_deg=args.heading_deg,
                mean_link_head_fk_rmse_m=float(np.mean(residuals)) if residuals else 0.,physical_tracking_validated=False)
    args.output.with_suffix('.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
