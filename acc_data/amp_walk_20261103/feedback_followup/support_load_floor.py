"""Instantaneous WBC sole-support load constraint; no prediction horizon."""
import math
import torch

def support_load_floor(mode, measured_normal_force, minimum, maximum=800., preferred_mode=None):
    if not math.isfinite(minimum) or not 0. <= minimum < maximum:
        raise ValueError('Minimum support load must be finite, nonnegative and below the force limit')
    if mode.ndim != 1 or measured_normal_force.shape != (len(mode),2):
        raise ValueError('Expected modes [B] and measured normal forces [B,2]')
    if not bool(torch.isfinite(measured_normal_force).all()):
        raise ValueError('Measured support forces must be finite')
    if preferred_mode is not None and preferred_mode.shape != mode.shape:
        raise ValueError('Preferred support modes must match modes [B]')
    G=measured_normal_force.new_zeros((len(mode),1,12));b=measured_normal_force.new_zeros((len(mode),1))
    for side in (0,1):
        selected=mode==(1<<side)
        if preferred_mode is not None:
            selected=selected | ((mode==3) & (preferred_mode==(1<<side)))
        active=selected & (measured_normal_force[:,side]>=20.)
        G[:,0,6*side+2]=-active.to(G.dtype)
        b[:,0]-=active.to(b.dtype)*minimum
    return G,b
