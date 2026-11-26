"""First variation of a world-left rotation correction Exp(delta) R(t).

The correction changes angular velocity and acceleration as well as pose:
dw = delta_dot - [w]x delta;
da = delta_ddot - [w]x delta_dot - [a]x delta.
Used only by offline reference preparation, not a runtime controller.
"""
import numpy as np


def skew(value):
    x,y,z=value
    return np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])


def rotation_variation_maps(position,velocity,acceleration,omega,alpha):
    """Return full tangent maps, preserving translation and joint rows."""
    v=velocity.copy();a=acceleration.copy()
    v[3:6]-=skew(omega)@position[3:6]
    a[3:6]-=skew(omega)@velocity[3:6]+skew(alpha)@position[3:6]
    return position.copy(),v,a
