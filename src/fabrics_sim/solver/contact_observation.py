"""Ground-filtered force observations shared by support detection and servo."""
import numpy as np


def ground_support_forces(contact_force_matrix):
    """Sum configured ground filters for the left/right foot, in world axes.

    The caller must configure ONLY ground colliders as filters. Net rigid-body
    contact forces also contain self collisions and cannot establish support.
    Missing/malformed filtered reports must never fall back to those net forces.
    """
    forces = np.asarray(contact_force_matrix)
    if (forces.ndim != 3 or forces.shape[0] != 2 or forces.shape[1] < 1
            or forces.shape[2] != 3 or not np.isfinite(forces).all()):
        raise ValueError('Expected finite ground-filtered forces [2, filters, 3]')
    return forces.sum(axis=1)
