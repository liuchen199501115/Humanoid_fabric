from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from .bvh import save_bvh
from .export import export_robot_pkl
from .h5_motion import load_motion_h5, load_stats
from .rotation import to_matrix
from .skeleton_aware.integrate import get_skeleton_aware_generator


def _namespace_from_dict(data):
    return SimpleNamespace(**data)


def _load_opt(checkpoint_dir: Path):
    opt_path = checkpoint_dir / "opt.npy"
    if not opt_path.exists():
        raise FileNotFoundError(f"Cannot find checkpoint options: {opt_path}")
    data = np.load(opt_path, allow_pickle=True).item()
    opt = _namespace_from_dict(data)
    opt.phase = "test"
    opt.isTrain = False
    return opt


def _pad_time_to_multiple(pose, multiple, minimum=1):
    if multiple < 1:
        raise ValueError("multiple must be positive")
    time_len = pose.shape[-1]
    if time_len < 1:
        raise ValueError("motion must contain at least one frame")
    target_len = max(time_len, minimum)
    target_len = ((target_len + multiple - 1) // multiple) * multiple
    pad = target_len - time_len
    if pad == 0:
        return pose, 0
    repeats = [1] * pose.ndim
    repeats[-1] = pad
    last = pose[..., -1:].repeat(*repeats)
    return torch.cat([pose, last], dim=-1), pad


def _crop_time(pose, pad):
    if pad <= 0:
        return pose
    return pose[..., :-pad]


def _normalize(pose, mean, var):
    return (pose - mean.cpu()) / var.cpu()


def _denormalize(pose, mean, var):
    return pose * var.to(pose.device) + mean.to(pose.device)


def _resolve_output_fps(requested_fps, source_info, fallback=30.0):
    fps = requested_fps
    if fps is None:
        fps = source_info.get("fps")
    if fps is None:
        fps = fallback
    fps = float(fps)
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    return fps


class Retargeter:
    def __init__(self, checkpoint_dir, stats_path=None, device="auto"):
        self.checkpoint_dir = Path(checkpoint_dir)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.opt = _load_opt(self.checkpoint_dir)
        if stats_path is None:
            stats_path = self.checkpoint_dir / "stats.npz"
        self.stats = load_stats(stats_path, self.device) if Path(stats_path).exists() else None
        if getattr(self.opt, "normalize_all", 0) and self.stats is None:
            raise FileNotFoundError(
                "This checkpoint was trained with normalize_all=1, but stats.npz was not found. "
                f"Expected: {stats_path}"
            )
        self.net_g_b = None

    def _load_pair(self, input_h5, target_h5):
        src = load_motion_h5(
            input_h5,
            velocity_virtual_node=bool(self.opt.velocity_virtual_node),
            use_global_y=bool(self.opt.use_global_y),
            rotation_data=self.opt.rotation_data,
            domain="B",
            mode="test",
        )
        tgt = load_motion_h5(
            target_h5,
            velocity_virtual_node=bool(self.opt.velocity_virtual_node),
            use_global_y=bool(self.opt.use_global_y),
            rotation_data=self.opt.rotation_data,
            domain="A",
            mode="test",
        )
        return src, tgt

    def _build_model(self, src_info, tgt_info):
        args = deepcopy(self.opt.skeleton_aware_args)
        args.topology_a = np.asarray(tgt_info["joint_parents_idx"])
        args.topology_b = np.asarray(src_info["joint_parents_idx"])
        args.fix_virtual_bones = self.opt.fix_virtual_bones
        args.rotation_data = self.opt.rotation_data
        args.rotation = self.opt.rotation
        args.num_layers = self.opt.skeleton_num_layers
        args.kernel_size = self.opt.skeleton_kernel_size
        args.window_size = self.opt.window_size
        args.with_root = self.opt.with_root
        args.connect_end_site = self.opt.connect_end_site
        args.velocity_virtual_node = self.opt.velocity_virtual_node
        args.acGAN = self.opt.acGAN
        args.v_connect_all_joints = self.opt.v_connect_all_joints
        args.last_sigmoid = self.opt.last_sigmoid
        args.scale_factor = self.opt.scale_factor
        args.skeleton_dist = self.opt.skeleton_dist
        args.end_sites_a = tgt_info["end_sites"] if args.connect_end_site else None
        args.end_sites_b = src_info["end_sites"] if args.connect_end_site else None

        _, net_g_b = get_skeleton_aware_generator(args)
        net_g_b.to(self.device)
        state_path = self.checkpoint_dir / "latest_net_G_B.pth"
        if not state_path.exists():
            raise FileNotFoundError(f"Cannot find generator weights: {state_path}")
        state = torch.load(state_path, map_location=self.device)
        net_g_b.load_state_dict(state, strict=True)
        net_g_b.eval()
        net_g_b.update(tgt_info)
        self.net_g_b = net_g_b

    def retarget(self, input_h5, target_h5):
        src, tgt = self._load_pair(input_h5, target_h5)
        self._build_model(src.info, tgt.info)

        pose_b = src.pose
        if getattr(self.opt, "normalize_all", 0):
            pose_b = _normalize(pose_b, self.stats["mean_B"], self.stats["var_B"])

        if self.opt.window_size > 0:
            skeleton_b = pose_b.reshape(1, pose_b.shape[0], pose_b.shape[1], pose_b.shape[2]).permute(0, 2, 3, 1)
        else:
            skeleton_b = pose_b
        pad_multiple = 2 ** self.opt.skeleton_num_layers
        # Reflection-padded temporal convolutions in the shipped checkpoint need
        # at least one full kernel.  Repeat the final frame for shorter clips and
        # crop the generated padding before export.
        minimum_time = self.opt.skeleton_kernel_size + 1
        skeleton_b, pad_len = _pad_time_to_multiple(
            skeleton_b, pad_multiple, minimum=minimum_time
        )

        with torch.no_grad():
            skeleton_a = self.net_g_b(skeleton_b.to(self.device).clone(), input_type="matrot").detach()
        skeleton_a = _crop_time(skeleton_a, pad_len)

        if self.opt.window_size > 0:
            joint_count = skeleton_a.shape[1]
            rot_dim = skeleton_a.shape[2]
            skeleton_a = skeleton_a.permute(0, 3, 1, 2).reshape(-1, joint_count, rot_dim)

        if getattr(self.opt, "normalize_all", 0):
            skeleton_a = _denormalize(skeleton_a, self.stats["mean_A"], self.stats["var_A"])

        root_a = None
        if self.opt.velocity_virtual_node:
            root_a = skeleton_a[:, -1, :3].clone()
            identity6 = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], device=skeleton_a.device)
            skeleton_a[:, -1, :] = identity6
            if self.opt.use_global_y:
                root_a[:, [0, 2]] = root_a[:, [0, 2]].cumsum(dim=0)
            else:
                root_a = root_a.cumsum(dim=0)
            root_a = root_a.detach().cpu().numpy()

        rot_a = to_matrix(skeleton_a).detach().cpu().numpy()
        return {
            "rotation_matrix": rot_a,
            "root_pos": root_a,
            "target_info": tgt.info,
            "source_info": src.info,
        }


def run_retarget(
    input_h5,
    target_h5,
    checkpoint_dir,
    output_dir,
    *,
    urdf_path,
    tag="demo",
    fps=None,
    stats_path=None,
    device="auto",
    export_bvh=True,
    clamp_limits=True,
    contact_velocity_threshold=0.08,
    contact_minimum_duration=0.10,
    contact_smoothing_duration=0.10,
    contact_height_threshold=0.04,
    contact_support_fallback_height=0.08,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    retargeter = Retargeter(checkpoint_dir=checkpoint_dir, stats_path=stats_path, device=device)
    result = retargeter.retarget(input_h5=input_h5, target_h5=target_h5)
    output_fps = _resolve_output_fps(fps, result["source_info"])

    pkl_path = output_dir / f"B2A_{tag}.pkl"
    export_robot_pkl(
        result["rotation_matrix"],
        result["root_pos"],
        result["target_info"]["joint_names"],
        urdf_path,
        output_fps,
        pkl_path,
        info=result["target_info"],
        align_ground=True,
        clamp_limits=clamp_limits,
        contact_velocity_threshold=contact_velocity_threshold,
        contact_minimum_duration=contact_minimum_duration,
        contact_smoothing_duration=contact_smoothing_duration,
        contact_height_threshold=contact_height_threshold,
        contact_support_fallback_height=contact_support_fallback_height,
    )

    bvh_path = None
    if export_bvh:
        bvh_path = output_dir / f"B2A_{tag}.bvh"
        save_bvh(
            result["rotation_matrix"],
            result["target_info"],
            bvh_path,
            translation=result["root_pos"],
            normalize=True,
            fps=output_fps,
        )

    return {"pkl": pkl_path, "bvh": bvh_path, "fps": output_fps}
