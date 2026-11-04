"""Motion IO, rotation geometry, and clip-safe heading-conditioned windows."""
from pathlib import Path
import pickle
import xml.etree.ElementTree as ET

import numpy as np
from torch.utils.data import Dataset


def joint_spec():
    path = Path(__file__).resolve().parents[1] / 'src/fabrics_sim/models/robots/urdf/g1/g1_29dof.urdf'
    joints = [j for j in ET.parse(path).getroot().findall('joint') if j.attrib['type'] == 'revolute']
    names = [j.attrib['name'] for j in joints]
    if len(names) != 29:
        raise ValueError('Expected G1 29-DOF URDF')
    limits = np.array([[float(j.find('limit').attrib[k]) for k in ('lower', 'upper')] for j in joints])
    return names, limits.astype(np.float32)


def quat_matrix(q):
    q = np.asarray(q)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(norm < 1e-8):
        raise ValueError('Zero quaternion')
    x, y, z, w = np.moveaxis(q / norm, -1, 0)
    return np.stack((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                     2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                     2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), -1).reshape(q.shape[:-1]+(3, 3))


def rotation6d(matrix):
    return np.concatenate((matrix[..., :, 0], matrix[..., :, 1]), axis=-1)


def matrix6d(x):
    a, b = x[..., :3], x[..., 3:6]
    norm = np.linalg.norm(a, axis=-1, keepdims=True)
    a = np.where(norm > 1e-7, a / np.maximum(norm, 1e-7), np.array([1., 0., 0.]))
    b = b - (a*b).sum(-1, keepdims=True)*a
    norm = np.linalg.norm(b, axis=-1, keepdims=True)
    fallback = np.eye(3)[np.argmin(np.abs(a), axis=-1)]
    fallback -= (fallback*a).sum(-1, keepdims=True)*a
    b = np.where(norm > 1e-7, b, fallback)
    b /= np.linalg.norm(b, axis=-1, keepdims=True)
    return np.stack((a, b, np.cross(a, b)), axis=-1)


def matrix_quat(m):
    # Symmetric eigensystem avoids trace singularities near a pi rotation.
    result = []
    for r in np.asarray(m).reshape(-1, 3, 3):
        xx, xy, xz, yx, yy, yz, zx, zy, zz = r.ravel()
        k = np.array([[xx-yy-zz, xy+yx, xz+zx, zy-yz],
                      [xy+yx, yy-xx-zz, yz+zy, xz-zx],
                      [xz+zx, yz+zy, zz-xx-yy, yx-xy],
                      [zy-yz, xz-zx, yx-xy, xx+yy+zz]])/3
        _, vec = np.linalg.eigh(k)
        result.append(vec[:, -1])
    q = np.asarray(result).reshape(np.asarray(m).shape[:-2]+(4,))
    return q.astype(np.float32)


def load_clip(path, fps, names):
    # Only load trusted local pickle files.
    with Path(path).open('rb') as f:
        d = pickle.load(f)
    p, q, j = [np.asarray(d[k], dtype=np.float64) for k in ('root_pos', 'root_rot', 'dof_pos')]
    if p.ndim != 2 or p.shape[1] != 3 or q.shape != (len(p), 4) or j.shape != (len(p), 29) or len(p) < 2:
        raise ValueError('Invalid root/joint array shapes or fewer than two frames')
    if not all(np.isfinite(a).all() for a in (p, q, j)):
        raise ValueError('Nonfinite motion')
    supplied = list(d['dof_names'])
    if len(supplied) != 29 or len(set(supplied)) != 29 or set(supplied) != set(names):
        raise ValueError('Joint names do not match G1')
    j = j[:, [supplied.index(n) for n in names]]
    source_fps = float(d['fps'])
    if not np.isfinite(source_fps) or source_fps <= 0 or not np.isfinite(fps) or fps <= 0:
        raise ValueError('Invalid fps')
    norms = np.linalg.norm(q, axis=1, keepdims=True)
    if np.any(norms < 1e-8):
        raise ValueError('Zero quaternion')
    q /= norms
    for i in range(1, len(q)):
        if q[i-1] @ q[i] < 0:
            q[i] *= -1
    times = np.arange(int(np.floor((len(p)-1)*fps/source_fps))+1)/fps
    source = np.arange(len(p))/source_fps
    interp = lambda a: np.stack([np.interp(times, source, a[:, i]) for i in range(a.shape[1])], -1)
    index = np.minimum(np.floor(times*source_fps).astype(int), len(q)-2)
    weight = (times*source_fps-index)[:, None]
    a, b = q[index], q[index+1]
    dot = np.clip((a*b).sum(-1, keepdims=True), -1, 1)
    theta = np.arccos(dot)
    den = np.maximum(np.sin(theta), 1e-8)
    slerp = np.sin((1-weight)*theta)/den*a + np.sin(weight*theta)/den*b
    q = np.where(dot > .9995, (1-weight)*a+weight*b, slerp)
    return np.concatenate((interp(p), rotation6d(quat_matrix(q)), interp(j)), -1).astype(np.float32)


def localize(states):
    """One fixed heading frame for the full window, anchored at history frame 1."""
    r = matrix6d(states[1, 3:9])
    yaw = np.arctan2(r[1, 0], r[0, 0])
    c, s = np.cos(yaw), np.sin(yaw)
    frame = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
    origin = states[1, :3].copy()
    origin[2] = 0  # Preserve absolute root height above the flat ground.
    local = states.copy()
    local[:, :3] = (states[:, :3]-origin) @ frame
    local[:, 3:9] = rotation6d(frame.T @ matrix6d(states[:, 3:9]))
    return local, origin, frame


def worldize(local, origin, frame):
    result = local.copy()
    result[:, :3] = local[:, :3] @ frame.T + origin
    result[:, 3:9] = rotation6d(frame @ matrix6d(local[:, 3:9]))
    return result


class MotionWindows(Dataset):
    def __init__(self, archive, split, horizon=25, stride=5, with_contacts=False):
        if horizon < 1 or stride < 1:
            raise ValueError('horizon and stride must be positive')
        self.states = archive['states']
        self.contacts = archive.get('contacts') if with_contacts else None
        if with_contacts and (self.contacts is None or self.contacts.shape != (len(self.states), 2)):
            raise ValueError('Rebuild dataset with --with-contacts; expected contacts [frames,2]')
        self.length = horizon + 2
        self.starts = []
        for i, flag in enumerate(archive['split']):
            if flag == split:
                begin, end = archive['offsets'][i:i+2]
                self.starts.extend(range(int(begin), int(end)-self.length+1, stride))
        if not self.starts:
            raise ValueError(f'No windows in split {split}')
        self.starts = np.asarray(self.starts)

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, i):
        start = self.starts[i]
        x, _, _ = localize(self.states[start:start+self.length])
        heading = x[-1, :2]-x[1, :2]
        norm = np.linalg.norm(heading)
        # For near-stationary clips use final facing direction, rather than NaN.
        if norm < .02:
            heading = matrix6d(x[-1, 3:9])[:2, 0]
            norm = np.linalg.norm(heading)
        heading = heading / norm if norm > 1e-6 else np.array([1., 0.])
        result = (x[:2].copy(), x[2:].copy(), heading.astype(np.float32))
        return result + (self.contacts[start:start+self.length].copy(),) if self.contacts is not None else result


def load_contacts(path, fps, frames):
    with Path(path).open('rb') as f:
        d = pickle.load(f)
    c = np.asarray(d['contact_sequence'], dtype=np.float32)
    if c.shape != (len(d['root_pos']), 2) or not np.isfinite(c).all() or not np.isin(c, [0, 1]).all():
        raise ValueError('Expected binary contact_sequence [T,2]')
    names = [n.replace('_foot', '') for n in d.get('contact_names', ['left', 'right'])]
    if names != ['left', 'right']:
        if set(names) == {'left', 'right'}:
            c = c[:, [names.index('left'), names.index('right')]]
        else:
            raise ValueError(f'Unknown contact names: {names}')
    indices = np.minimum(np.rint(np.arange(frames)/fps*float(d['fps'])).astype(int), len(c)-1)
    return c[indices]
