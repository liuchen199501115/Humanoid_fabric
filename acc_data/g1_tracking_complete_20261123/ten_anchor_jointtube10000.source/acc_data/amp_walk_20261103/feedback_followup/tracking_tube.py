"""Acceleration inequalities for a moving Euclidean tracking tube.

These constrain predicted physical acceleration; no reference/state reset.
The source-pose auditor remains independent and retains its original limits.
"""
import math
import torch

def tracking_tube_constraints(q,v,reference_q,reference_v,reference_a,*,joint_radius=.195,position_radius=.195,orientation_radius=.34,bandwidth=10.,joint_bandwidth=None,orientation_bandwidth=None):
    rows=[];bounds=[];diagnostics=[]
    for qs,vs,radius,divisor in [(slice(7,36),slice(6,35),joint_radius,29.),(slice(0,3),slice(0,3),position_radius,1.)]:
        error=q[:,qs]-reference_q[:,qs]
        velocity=v[:,vs]-reference_v[:,vs]
        rate=(joint_bandwidth if divisor==29. and joint_bandwidth is not None else bandwidth)
        h=radius**2-error.square().sum(-1)/divisor
        hdot=-2*(error*velocity).sum(-1)/divisor
        row=torch.zeros_like(v);row[:,vs]=2*error/divisor
        bound=(2*(error*reference_a[:,vs]).sum(-1)-2*velocity.square().sum(-1))/divisor+2*rate*hdot+rate**2*h
        rows.append(row);bounds.append(bound)
        diagnostics.append(torch.stack((h,hdot,hdot+rate*h),-1))
    def product(a,b):
        return torch.cat((a[:,3:]*b[:,:3]+b[:,3:]*a[:,:3]+torch.linalg.cross(a[:,:3],b[:,:3]),a[:,3:]*b[:,3:]-(a[:,:3]*b[:,:3]).sum(-1,keepdim=True)),-1)
    def pure(a):return torch.cat((a,torch.zeros_like(a[:,:1])),-1)
    actual=q[:,3:7]/q[:,3:7].norm(dim=-1,keepdim=True)
    target=reference_q[:,3:7]/reference_q[:,3:7].norm(dim=-1,keepdim=True)
    actual_v=.5*product(pure(v[:,3:6]),actual)
    target_v=.5*product(pure(reference_v[:,3:6]),target)
    actual_a_bias=.5*product(pure(v[:,3:6]),actual_v)
    target_a=.5*product(pure(reference_a[:,3:6]),target)+.5*product(pure(reference_v[:,3:6]),target_v)
    c=(actual*target).sum(-1)
    cd=(actual_v*target+actual*target_v).sum(-1)
    cb=(actual_a_bias*target+2*actual_v*target_v+actual*target_a).sum(-1)
    columns=[]
    for axis in range(3):
        unit=torch.zeros_like(v[:,:3]);unit[:,axis]=1
        columns.append((.5*product(pure(unit),actual)*target).sum(-1))
    g=torch.stack(columns,-1)
    h=c.square()-math.cos(orientation_radius/2)**2
    hd=2*c*cd
    row=torch.zeros_like(v);row[:,3:6]=-2*c[:,None]*g
    rotation_rate=bandwidth if orientation_bandwidth is None else orientation_bandwidth
    bound=2*cd.square()+2*c*cb+2*rotation_rate*hd+rotation_rate**2*h
    rows.append(row);bounds.append(bound);diagnostics.append(torch.stack((h,hd,hd+rotation_rate*h),-1))
    return torch.stack(rows,1),torch.stack(bounds,1),torch.stack(diagnostics,1)
