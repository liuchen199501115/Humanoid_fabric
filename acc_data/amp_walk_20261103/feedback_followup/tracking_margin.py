"""Instantaneous, bounded task allocation from measured tracking margins.

The short stopping-time term is a velocity error, not a horizon optimization.
This changes only soft objective weights; reference q/v/a, contact rows, effort
bounds, and independent source acceptance limits are untouched.
"""
import math
import torch


def margin_weights(q, v, reference_q, reference_v, reaction_time=.08, maximum=16.):
    if not math.isfinite(reaction_time) or reaction_time < 0:
        raise ValueError('Reaction time must be finite and nonnegative')
    if not math.isfinite(maximum) or maximum < 1:
        raise ValueError('Maximum weight must be finite and at least one')
    if (q.ndim != 2 or q.shape[1] != 36 or v.shape != (len(q),35)
            or reference_q.shape != q.shape or reference_v.shape != v.shape
            or not all(torch.isfinite(x).all() for x in (q,v,reference_q,reference_v))):
        raise ValueError('Expected finite G1 states')
    def error_norm(error, velocity, radius):
        return torch.maximum(error.norm(dim=-1), (error+reaction_time*velocity).norm(dim=-1))/radius
    position=error_norm(q[:,:3]-reference_q[:,:3],v[:,:3]-reference_v[:,:3],.2)
    joint=error_norm(q[:,7:]-reference_q[:,7:],v[:,6:]-reference_v[:,6:],.2*math.sqrt(29))
    actual=q[:,3:7]/q[:,3:7].norm(dim=-1,keepdim=True).clamp_min(1e-12)
    target=reference_q[:,3:7]/reference_q[:,3:7].norm(dim=-1,keepdim=True).clamp_min(1e-12)
    xyz=target[:,3:]*actual[:,:3]-actual[:,3:]*target[:,:3]-torch.linalg.cross(actual[:,:3],target[:,:3])
    w=(actual*target).sum(-1,keepdim=True)
    xyz=torch.where(w<0,-xyz,xyz);w=w.abs()
    length=xyz.norm(dim=-1,keepdim=True)
    rotation=xyz*(2*torch.atan2(length,w)/length.clamp_min(1e-12))
    orientation=error_norm(rotation,v[:,3:6]-reference_v[:,3:6],.35)
    risk=torch.stack((position,orientation,joint),-1)
    # Smoothly amplify only shrinking margins, with bounded conditioning.
    weights=(1-risk.square()).clamp_min(1/math.sqrt(maximum)).pow(-2)
    return weights,risk


def allocate_margin_weights(requests, weights, blend):
    if not 0 <= blend <= 1:
        raise ValueError('Blend must be in [0,1]')
    names={'center_of_mass':0,'body_orientation':1,'source_pelvis_orientation':1,'joint_posture':2}
    for request in requests:
        if request.name in names:
            scale=1+blend*(weights[:,names[request.name]]-1)
            request.metric*=scale[:,None,None]
