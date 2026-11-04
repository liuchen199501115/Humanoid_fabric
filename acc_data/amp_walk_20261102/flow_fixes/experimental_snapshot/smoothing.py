"""Optional offline smoothing for control references; keep source poses for audit."""
import math
import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d


def smooth_motion_reference(trajectory, fps, sigma_seconds):
    if (trajectory.ndim != 2 or trajectory.shape[1] != 36 or len(trajectory)<2
            or not torch.isfinite(trajectory).all() or not trajectory.is_floating_point()
            or not all(math.isfinite(v) and v>0 for v in (fps,sigma_seconds))):
        raise ValueError('Expected finite floating poses [T,36] and positive smoothing parameters')
    source = trajectory.detach().cpu().double().numpy().copy()
    norms = np.linalg.norm(source[:,3:7],axis=1)
    if (norms<1e-8).any():raise ValueError('Quaternion must be nonzero')
    source[:,3:7] /= norms[:,None]
    for i in range(1,len(source)):
        if source[i-1,3:7]@source[i,3:7]<0:source[i,3:7]*=-1
    sigma = fps*sigma_seconds
    pad = max(1,int(math.ceil(4*sigma)))
    # Linear extension retains initial/final motion; reflection would invent a
    # stationary boundary for a clip that starts halfway through a stride.
    before = source[:1]-np.arange(pad,0,-1)[:,None]*(source[1]-source[0])
    after = source[-1:]+np.arange(1,pad+1)[:,None]*(source[-1]-source[-2])
    result = gaussian_filter1d(np.concatenate((before,source,after)),sigma,axis=0)[pad:-pad]
    norm = np.linalg.norm(result[:,3:7],axis=1)
    if (norm<1e-8).any():raise ValueError('Smoothing crossed a singular quaternion')
    result[:,3:7] /= norm[:,None]
    result[:,7:] = np.clip(result[:,7:],source[:,7:].min(axis=0),source[:,7:].max(axis=0))
    return torch.as_tensor(result,device=trajectory.device,dtype=trajectory.dtype)
