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


def limit_upward_speed_target(desired, velocity_z, speed_limit, bandwidth, minimum, blend):
    """Soft velocity feedback ceiling, allowing upward braking during descent."""
    if (not math.isfinite(speed_limit) or speed_limit < 0
            or not math.isfinite(bandwidth) or bandwidth <= 0
            or not math.isfinite(minimum) or not -9.81 < minimum <= 0
            or not math.isfinite(blend) or not 0 <= blend <= 1
            or velocity_z.shape != desired[..., 2].shape
            or not bool(velocity_z.isfinite().all())):
        raise ValueError('Invalid COM upward-speed target parameters')
    result = desired.clone()
    if blend:
        ceiling = (bandwidth*(speed_limit-velocity_z)).clamp_min(minimum)
        vertical = desired[..., 2]
        limited = vertical.minimum(ceiling)
        result[..., 2] = vertical + blend*(limited-vertical)
    return result
