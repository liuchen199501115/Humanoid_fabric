"""Physical angular-momentum-rate feedback in world coordinates."""
import torch


def momentum_rate(momentum, reference, reference_rate, damping, maximum):
    return torch.clamp(reference_rate - damping * (momentum - reference), -maximum, maximum)
