"""Bound angular tracking feedback separately from reference acceleration."""
import math
import torch
from fabrics_sim.fabrics.dynamics_aware import rotation_error


def angular_tracking_target(rotation, target_rotation, angular_velocity,
                            reference_velocity, reference_acceleration,
                            stiffness, damping, limit):
    if not math.isfinite(limit) or limit <= 0:
        raise ValueError('Angular feedback limit must be finite and positive')
    error = rotation_error(rotation, target_rotation)
    feedback = -stiffness*error-damping*(angular_velocity-reference_velocity)
    scale = (limit/torch.linalg.vector_norm(feedback, dim=-1, keepdim=True).clamp_min(1e-8)).clamp(max=1.)
    return reference_acceleration+feedback*scale
