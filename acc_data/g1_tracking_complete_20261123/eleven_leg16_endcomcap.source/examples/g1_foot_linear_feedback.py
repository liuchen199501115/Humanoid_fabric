"""Bound foot translation feedback while preserving analytic feedforward."""
import math
import torch


def linear_tracking_target(position, target, velocity, reference_velocity,
                           reference_acceleration, stiffness, damping, limit):
    if not math.isfinite(limit) or limit<=0:
        raise ValueError('Foot linear feedback limit must be positive and finite')
    feedback=-stiffness*(position-target)-damping*(velocity-reference_velocity)
    scale=(limit/torch.linalg.vector_norm(feedback,dim=-1,keepdim=True).clamp_min(1e-8)).clamp(max=1)
    return reference_acceleration+scale*feedback
