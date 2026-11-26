"""Shared acceleration constraints for the runtime wrist/body collision covers."""
import math
import torch


def wrist_collision_barrier(distance, jacobian, bias, speed, first_pair, pair_count,
                            safe_distance, stable_rows=False, bandwidth=8.):
    if not math.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError("Collision barrier bandwidth must be finite and positive")
    # h'' + 2*w*h' + w*w*h >= 0, h = cover distance - safety margin.
    active = distance < .4
    active[:, :first_pair] = False
    active[:, pair_count:] = False
    bound = bias + (2*bandwidth)*speed + bandwidth**2*(distance-safe_distance)
    if stable_rows:
        selected = slice(first_pair, pair_count)
        return (torch.where(active[:, selected, None], -jacobian[:, selected],
                            torch.zeros_like(jacobian[:, selected])),
                torch.where(active[:, selected], bound[:, selected],
                            torch.zeros_like(bound[:, selected])))
    rows = active.any(dim=0)
    bound = torch.where(active, bound, torch.full_like(bound, 1e6))
    return -jacobian[:, rows], bound[:, rows]
