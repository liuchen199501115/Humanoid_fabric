"""Bounded touchdown search inside the existing foot acceleration task."""
import math
import torch


def contact_search_target(desired, position, target, normal_force, planned_mode,
                          active_rows, acceleration, blend=1.):
    """Descend near an intended landing until measured contact takes over.

    This modifies only a soft Cartesian acceleration request. Contact forces,
    physical constraints and the original-source acceptance remain external.
    """
    if (not math.isfinite(acceleration) or not 0 < acceleration <= 10
            or not math.isfinite(blend) or not 0 <= blend <= 1):
        raise ValueError('Expected bounded downward acceleration and unit blend')
    batch = len(desired)
    if (desired.shape != (batch, 12) or position.shape != (batch, 2, 3)
            or target.shape != position.shape or normal_force.shape != (batch, 2)
            or planned_mode.shape != (batch,) or active_rows.shape != desired.shape
            or active_rows.dtype != torch.bool or planned_mode.dtype.is_floating_point
            or bool(((planned_mode < 0) | (planned_mode > 3)).any())
            or not all(torch.isfinite(x).all() for x in (desired, position, target, normal_force))):
        raise ValueError('Expected finite batched G1 foot tasks and integer contact modes')
    gap = position-target
    intended = torch.stack(((planned_mode & 1) != 0, (planned_mode & 2) != 0), dim=1)
    # Do not chase a remote landing or continue descending below its target.
    near = (gap[:, :, :2].norm(dim=-1) <= .04) & (gap[:, :, 2] <= .02) & (gap[:, :, 2] >= -.005)
    free = active_rows.reshape(batch, 2, 6)[:, :, :3].all(dim=-1)
    selected = intended & free & (normal_force < 20.) & near
    result = desired.clone()
    result.reshape(batch, 2, 6)[:, :, 2] -= selected.to(result)*acceleration*blend
    return result, selected & (blend > 0)
