from pathlib import Path
import pickle

import torch


def _normalize_quat(q):
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def _slerp(q0, q1, blend):
    q0 = _normalize_quat(q0)
    q1 = _normalize_quat(q1)
    dot = (q0 * q1).sum(dim=-1, keepdim=True)
    q1 = torch.where(dot < 0.0, -q1, q1)
    dot = torch.abs(dot).clamp(-1.0, 1.0)

    blend = blend.unsqueeze(-1)
    linear = _normalize_quat((1.0 - blend) * q0 + blend * q1)
    theta_0 = torch.acos(dot)
    sin_theta_0 = torch.sin(theta_0)
    theta = theta_0 * blend
    s0 = torch.sin(theta_0 - theta) / sin_theta_0.clamp_min(1e-8)
    s1 = torch.sin(theta) / sin_theta_0.clamp_min(1e-8)
    spherical = s0 * q0 + s1 * q1
    return torch.where(dot > 0.9995, linear, spherical)


class _MotionUnpickler(pickle.Unpickler):
    """Read NumPy 2 array pickle paths with this NumPy 1 environment too."""
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except ModuleNotFoundError as exc:
            if module.startswith("numpy._core.") and exc.name in ("numpy._core", module):
                return super().find_class(module.replace("numpy._core.", "numpy.core.", 1), name)
            raise


class MotionLib:
    """Minimal pkl motion loader for MuJoCo visualization.

    The pkl format is produced by retargeting.export.export_robot_pkl:
    fps/root_pos/root_rot/dof_pos and optional dof_names.
    """

    def __init__(self, motion_file, device="cpu"):
        self.device = torch.device(device)
        self.motion_file = Path(motion_file)
        self._load_motion()

    def _load_motion(self):
        with self.motion_file.open("rb") as f:
            data = _MotionUnpickler(f).load()
        self.fps = float(data["fps"])
        self.dt = 1.0 / self.fps
        self.root_pos = torch.as_tensor(data["root_pos"], dtype=torch.float32, device=self.device)
        self.root_rot = _normalize_quat(torch.as_tensor(data["root_rot"], dtype=torch.float32, device=self.device))
        self.dof_pos = torch.as_tensor(data["dof_pos"], dtype=torch.float32, device=self.device)
        self.dof_names = [str(name) for name in data.get("dof_names", [])] or None
        self.num_frames = self.root_pos.shape[0]
        self.length = torch.tensor(max(self.num_frames - 1, 1) * self.dt, dtype=torch.float32, device=self.device)

    def get_motion_length(self, motion_ids):
        return self.length.repeat(motion_ids.shape[0])

    def get_dof_names(self):
        return self.dof_names

    def build_dof_index_map(self, target_dof_names):
        if self.dof_names is None:
            raise ValueError("Motion file does not contain dof_names.")
        missing = [name for name in target_dof_names if name not in self.dof_names]
        if missing:
            raise KeyError(f"Missing joints in motion file: {missing}")
        return torch.tensor([self.dof_names.index(name) for name in target_dof_names], dtype=torch.long, device=self.device)

    def calc_motion_frame(self, motion_ids, motion_times):
        del motion_ids
        times = torch.remainder(motion_times.to(self.device), self.length)
        frame = (times / self.dt).clamp(0, self.num_frames - 1)
        idx0 = torch.floor(frame).long()
        idx1 = torch.clamp(idx0 + 1, max=self.num_frames - 1)
        blend = frame - idx0.float()
        blend_u = blend.unsqueeze(-1)

        root_pos = (1.0 - blend_u) * self.root_pos[idx0] + blend_u * self.root_pos[idx1]
        root_rot = _slerp(self.root_rot[idx0], self.root_rot[idx1], blend)
        dof_pos = (1.0 - blend_u) * self.dof_pos[idx0] + blend_u * self.dof_pos[idx1]
        root_vel = torch.zeros_like(root_pos)
        root_ang_vel = torch.zeros_like(root_pos)
        dof_vel = torch.zeros_like(dof_pos)
        return root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel
