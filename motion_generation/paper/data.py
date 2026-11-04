"""99D paper features with fixed two-frame history and heading conditioning."""
import numpy as np
import torch
from torch.utils.data import Dataset
from ..data import matrix6d, matrix_quat, quat_matrix
from .kinematics import PaperFK, DIM


def quat_multiply(a, b):
    xyz=a[...,3:4]*b[...,:3]+b[...,3:4]*a[...,:3]+np.cross(a[...,:3],b[...,:3])
    w=a[...,3:4]*b[...,3:4]-(a[...,:3]*b[...,:3]).sum(-1,keepdims=True)
    return np.concatenate((xyz,w),-1)


def canonicalize_quaternions(q):
    q=q.copy()
    if len(q)>1:
        flip=np.where((q[1:]*q[:-1]).sum(-1)<0,-1.,1.)
        q[1:]*=np.cumprod(flip)[:,None]
    return q


def localize(states):
    q=states[1,3:7]
    r=quat_matrix(q)
    yaw=np.arctan2(r[1,0],r[0,0])
    c,s=np.cos(yaw),np.sin(yaw)
    frame=np.array([[c,-s,0],[s,c,0],[0,0,1]],dtype=np.float32)
    origin=states[1,:3].copy();origin[2]=0
    local=states.copy()
    local[:,:3]=(states[:,:3]-origin)@frame
    local[:,3:7]=canonicalize_quaternions(quat_multiply(np.array([0,0,-np.sin(yaw/2),np.cos(yaw/2)],dtype=np.float32),states[:,3:7]))
    # Anchor the quaternion sign, so random file sign conventions do not affect statistics.
    if local[1,6]<0:local[:,3:7]*=-1
    local[:,30:]=((states[:,30:].reshape(-1,23,3)-origin)@frame).reshape(-1,69)
    return local,origin,frame,yaw


def worldize(states, origin, frame, yaw):
    world=states.copy()
    world[:,:3]=states[:,:3]@frame.T+origin
    world[:,3:7]=quat_multiply(np.array([0,0,np.sin(yaw/2),np.cos(yaw/2)],dtype=np.float32),states[:,3:7])
    world[:,30:]=(states[:,30:].reshape(-1,23,3)@frame.T+origin).reshape(-1,69)
    return world


def encode_legacy(states, source_names, fk):
    indices=[list(source_names).index(n) for n in fk.names]
    p=states[:,:3]
    q=matrix_quat(matrix6d(states[:,3:9]))
    angles=states[:,9:][:,indices]
    config=np.concatenate((p,q,angles),-1).astype(np.float32)
    with torch.no_grad():links=fk(torch.from_numpy(config)).numpy()
    return np.concatenate((config,links.reshape(-1,69)),-1)


class PaperWindows(Dataset):
    def __init__(self, archive, split, horizon=25, stride=5, **_):
        if horizon!=25:
            raise ValueError('Paper profile predicts exactly 25 future frames at 50 FPS')
        if stride<1:raise ValueError('stride must be positive')
        if str(archive.get('representation',''))!='g1-paper99' or float(archive['fps'])!=50.:
            raise ValueError('Use motion_generation.paper.prepare to create 99D, 50 FPS data')
        self.states=archive['states']
        if self.states.ndim!=2 or self.states.shape[1]!=DIM:raise ValueError('Expected [T,99]')
        self.starts=[]
        for i,flag in enumerate(archive['split']):
            if flag==split:
                begin,end=archive['offsets'][i:i+2]
                self.starts.extend(range(int(begin),int(end)-27+1,stride))
        if not self.starts:raise ValueError('No complete windows in requested split')
        self.starts=np.asarray(self.starts)

    def __len__(self):return len(self.starts)

    def __getitem__(self,index):
        start=self.starts[index]
        x,*_=localize(self.states[start:start+27])
        heading=x[-1,:2]-x[1,:2]
        norm=np.linalg.norm(heading)
        if norm<.02:
            heading=quat_matrix(x[-1,3:7])[:2,0];norm=np.linalg.norm(heading)
        heading=heading/max(norm,1e-6) if norm>1e-6 else np.array([1.,0.])
        return x[:2].copy(),x[2:].copy(),heading.astype(np.float32)

    def __getitems__(self, indices):
        """Batch fetch without the per-window NumPy rotation hot path."""
        from .kinematics import quat_matrix as torch_matrix
        starts = torch.as_tensor(self.starts[indices])
        source = torch.from_numpy(self.states)
        x = source[starts[:, None]+torch.arange(27)].clone()
        rotation = torch_matrix(x[:, 1, 3:7])
        yaw = torch.atan2(rotation[:, 1, 0], rotation[:, 0, 0])
        c, s = yaw.cos(), yaw.sin()
        frame = torch.zeros(len(indices), 3, 3)
        frame[:, 0, 0] = c; frame[:, 0, 1] = -s
        frame[:, 1, 0] = s; frame[:, 1, 1] = c; frame[:, 2, 2] = 1
        origin = x[:, 1, :3].clone(); origin[:, 2] = 0
        x[:, :, :3] = (x[:, :, :3]-origin[:, None])@frame
        links = x[:, :, 30:].reshape(len(indices), -1, 3)
        x[:, :, 30:] = ((links-origin[:, None])@frame).reshape(len(indices), 27, 69)
        q = x[:, :, 3:7].clone()
        a = torch.zeros(len(indices), 1, 4)
        a[:, 0, 2] = -(yaw/2).sin(); a[:, 0, 3] = (yaw/2).cos()
        xyz = a[:, :, 3:]*q[:, :, :3]+q[:, :, 3:]*a[:, :, :3]+torch.cross(a[:, :, :3].expand_as(q[:, :, :3]), q[:, :, :3], dim=-1)
        w = a[:, :, 3:]*q[:, :, 3:]-(a[:, :, :3]*q[:, :, :3]).sum(-1, keepdim=True)
        q = torch.cat((xyz, w), -1)
        sign = torch.where((q[:, 1:]*q[:, :-1]).sum(-1)<0, -1., 1.).cumprod(1)
        q[:, 1:] *= sign[:, :, None]
        q *= torch.where(q[:, 1, 3]<0, -1., 1.)[:, None, None]
        x[:, :, 3:7] = q
        heading = x[:, -1, :2]-x[:, 1, :2]
        norm = heading.norm(dim=-1, keepdim=True)
        fallback = torch_matrix(q[:, -1])[:, :2, 0]
        heading = torch.where(norm<.02, fallback, heading)
        norm = heading.norm(dim=-1, keepdim=True)
        heading = torch.where(norm>1e-6, heading/norm.clamp_min(1e-6), torch.tensor([1.,0.]))
        return [(row[:2], row[2:], direction) for row, direction in zip(x, heading)]
