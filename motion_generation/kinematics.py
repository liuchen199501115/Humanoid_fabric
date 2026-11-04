"""Differentiable G1 sole-sphere FK from the repository URDF (metres)."""
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def rotation_6d(x):
    a = F.normalize(x[..., :3], dim=-1, eps=1e-6)
    b = F.normalize(x[..., 3:]-(a*x[..., 3:]).sum(-1, keepdim=True)*a, dim=-1, eps=1e-6)
    return torch.stack((a, b, torch.cross(a, b, dim=-1)), -1)


def rpy_matrix(rpy):
    r, p, y = rpy
    cr, cp, cy = np.cos([r, p, y]); sr, sp, sy = np.sin([r, p, y])
    return np.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                     [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr], [-sp, cp*sr, cp*cr]])


class G1FeetFK(nn.Module):
    def __init__(self, names):
        super().__init__()
        path = Path(__file__).resolve().parents[1]/'src/fabrics_sim/models/robots/urdf/g1/g1_29dof.urdf'
        joints = ET.parse(path).getroot().findall('joint')
        by_child = {j.find('child').attrib['link']: j for j in joints}
        self.chains = []
        for side in ('left', 'right'):
            chain, link = [], f'{side}_ankle_roll_link'
            while link in by_child:
                j = by_child[link]
                chain.append(j)
                link = j.find('parent').attrib['link']
            entries = []
            for k, j in enumerate(reversed(chain)):
                prefix = f'{side}_{k}'
                origin = j.find('origin')
                xyz = np.fromstring(origin.get('xyz', '0 0 0') if origin is not None else '0 0 0', sep=' ')
                rpy = np.fromstring(origin.get('rpy', '0 0 0') if origin is not None else '0 0 0', sep=' ')
                axis = j.find('axis')
                axis = np.fromstring(axis.get('xyz') if axis is not None else '1 0 0', sep=' ')
                axis = axis/np.linalg.norm(axis)
                x, y, z = axis
                skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
                for name, value in [('xyz', xyz), ('rotation', rpy_matrix(rpy)), ('skew', skew)]:
                    self.register_buffer(prefix+'_'+name, torch.tensor(value, dtype=torch.float32))
                index = -1 if j.get('type') == 'fixed' else names.index(j.get('name'))
                entries.append((prefix, index))
            self.chains.append(entries)
        self.register_buffer('centres', torch.tensor([[-.05,.025,-.03],[-.05,-.025,-.03],[.12,.03,-.03],[.12,-.03,-.03]]))

    def forward(self, states):
        feet = []
        identity = torch.eye(3, device=states.device, dtype=states.dtype)
        for chain in self.chains:
            p, r = states[..., :3], rotation_6d(states[..., 3:9])
            for prefix, index in chain:
                p = p + (r @ getattr(self, prefix+'_xyz')[..., None]).squeeze(-1)
                r = r @ getattr(self, prefix+'_rotation')
                if index >= 0:
                    angle = states[..., 9+index, None, None]
                    k = getattr(self, prefix+'_skew')
                    r = r @ (identity+angle.sin()*k+(1-angle.cos())*(k@k))
            feet.append(p[..., None, :] + (r[..., None, :, :] @ self.centres[..., None]).squeeze(-1))
        return torch.stack(feet, dim=-3)  # [..., left/right, four spheres, xyz]
