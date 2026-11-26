"""Optional root XY tracking barrier, without a vertical recovery direction."""
import math
import torch


def root_xy_tube(q,v,reference_q,reference_v,reference_a,*,radius=.18,bandwidth=10.):
    if not math.isfinite(radius) or not 0<radius<=.2 or not math.isfinite(bandwidth) or bandwidth<=0:
        raise ValueError('Invalid root XY tube radius or bandwidth')
    error=q[:,:2]-reference_q[:,:2];velocity=v[:,:2]-reference_v[:,:2]
    h=radius**2-error.square().sum(-1);hd=-2*(error*velocity).sum(-1)
    C=torch.zeros_like(v);C[:,:2]=2*error
    d=2*(error*reference_a[:,:2]).sum(-1)-2*velocity.square().sum(-1)+2*bandwidth*hd+bandwidth**2*h
    return C[:,None],d[:,None],torch.stack((h,hd,hd+bandwidth*h),-1)
