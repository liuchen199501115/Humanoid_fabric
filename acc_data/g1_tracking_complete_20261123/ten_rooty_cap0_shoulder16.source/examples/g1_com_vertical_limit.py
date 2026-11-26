"""Optional soft COM task policy; leaves all physical QP constraints intact."""
import math


def limit_upward_target(desired, maximum, blend):
    """Blend only the vertical desired acceleration toward an upper cap."""
    if not math.isfinite(maximum) or maximum < 0:
        raise ValueError('Maximum upward COM target must be finite and nonnegative')
    if not math.isfinite(blend) or not 0 <= blend <= 1:
        raise ValueError('COM cap blend must lie in [0,1]')
    result = desired.clone()
    if blend:
        vertical = desired[..., 2]
        result[..., 2] = vertical + blend * (vertical.clamp_max(maximum) - vertical)
    return result
