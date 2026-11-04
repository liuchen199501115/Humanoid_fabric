"""Explicit arm-only adaptation; the original motion remains the audit target."""
import math
import torch


def arm_clearance_reference(source, names, offset, lower, upper):
    if (source.ndim!=2 or source.shape[1]!=36 or len(names)!=29
            or not torch.isfinite(source).all() or not math.isfinite(offset) or offset<0
            or lower.shape!=(29,) or upper.shape!=(29,)
            or not torch.isfinite(lower).all() or not torch.isfinite(upper).all()
            or (lower>upper).any()):
        raise ValueError('Arm adaptation requires finite G1 motion, limits and nonnegative offset')
    result=source.clone()
    for side,sign in (('left',1.),('right',-1.)):
        index=names.index(side+'_shoulder_roll_joint')
        result[:,7+index]+=sign*offset
    if ((result[:,7:]<lower-1e-6)|(result[:,7:]>upper+1e-6)).any():
        raise ValueError('Arm clearance offset exceeds joint limits in the source clip')
    return result
