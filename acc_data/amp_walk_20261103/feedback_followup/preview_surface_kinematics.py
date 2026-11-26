"""Instantaneous no-slip velocity at native sphere/point ground contacts.

The world ground normal is fixed. A sphere's current contact offset is
R p - radius * normal, and its time derivative is omega x (R p).
"""
import numpy as np
from centroidal_pose_linearization import skew

def surface_jacobian(rotation, jacobian, points, radii):
    rotated = np.asarray(points) @ rotation.T
    offsets = rotated - np.asarray(radii)[:, None] * np.array([0., 0., 1.])
    return np.vstack([jacobian[:3] - skew(offset) @ jacobian[3:] for offset in offsets])

def surface_bias(rotation, jacobian, foot_bias, velocity, points, radii):
    rotated = np.asarray(points) @ rotation.T
    offsets = rotated - np.asarray(radii)[:, None] * np.array([0., 0., 1.])
    omega = jacobian[3:] @ velocity
    return np.concatenate([
        foot_bias[:3] - skew(offset) @ foot_bias[3:]
        + np.cross(omega, np.cross(omega, center))
        for center, offset in zip(rotated, offsets)])
