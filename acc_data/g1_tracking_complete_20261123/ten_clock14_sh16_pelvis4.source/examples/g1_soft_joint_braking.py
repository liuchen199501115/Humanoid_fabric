"""One-sided optional stop braking rows; stronger inward braking is unpenalized."""
import math
import torch


def soft_joint_braking_rows(position, velocity, lower, upper, indices, bandwidth=8.,
                            margin=.05, acceleration_limit=50.):
    if (position.ndim != 2 or position.shape[1] != 29
            or any(x.shape != position.shape for x in (velocity, lower, upper))
            or not all(torch.isfinite(x).all() for x in (position, velocity, lower, upper))
            or (upper <= lower).any() or not indices or len(set(indices)) != len(indices)
            or any(type(i) is not int or not 0 <= i < 29 for i in indices)
            or not all(math.isfinite(x) and x > 0 for x in (bandwidth, margin, acceleration_limit))):
        raise ValueError('Invalid G1 soft braking inputs')
    p,v,lo,hi=(x[:,indices] for x in (position,velocity,lower,upper))
    room=torch.minimum(torch.full_like(lo,margin),.25*(hi-lo))
    inward_low=(-2*bandwidth*v-bandwidth**2*(p-lo-room)).clamp(0.,acceleration_limit)
    inward_high=(-2*bandwidth*v+bandwidth**2*(hi-room-p)).clamp(-acceleration_limit,0.)
    count=len(indices);C=position.new_zeros(len(position),2*count,35);d=position.new_zeros(len(position),2*count)
    for j,index in enumerate(indices):
        C[:,2*j,6+index]=-(inward_low[:,j]>0).to(C)
        C[:,2*j+1,6+index]=(inward_high[:,j]<0).to(C)
        d[:,2*j]=-inward_low[:,j];d[:,2*j+1]=inward_high[:,j]
    return C,d
