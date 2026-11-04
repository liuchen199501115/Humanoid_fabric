"""Shared acceleration constraints for the runtime wrist/body collision covers."""
import torch


def wrist_collision_barrier(distance, jacobian, bias, speed, first_pair, pair_count,
                            safe_distance, stable_rows=False):
    # h'' + 2*w*h' + w*w*h >= 0, h = cover distance - safety margin.
    active = distance < .4
    active[:, :first_pair] = False
    active[:, pair_count:] = False
    bound = bias + 16*speed + 64*(distance-safe_distance)
    if stable_rows:
        selected = slice(first_pair, pair_count)
        return (torch.where(active[:, selected, None], -jacobian[:, selected],
                            torch.zeros_like(jacobian[:, selected])),
                torch.where(active[:, selected], bound[:, selected],
                            torch.zeros_like(bound[:, selected])))
    rows = active.any(dim=0)
    bound = torch.where(active, bound, torch.full_like(bound, 1e6))
    return -jacobian[:, rows], bound[:, rows]
