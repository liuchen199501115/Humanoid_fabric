"""Offline first variation of a normalized quaternion coefficient spline.

Rotating each coefficient and then interpolating differs from rotating the
interpolated curve. These maps follow the actual exported-reference operation.
Quaternions use xyzw; velocities and accelerations are in world coordinates.
"""
import numpy as np


def quaternion_kinematics(c, v, a):
    norm = np.linalg.norm(c, axis=-1, keepdims=True)
    u = c / norm
    speed = np.sum(u*v, axis=-1, keepdims=True)
    du = (v-u*speed)/norm
    curvature = (np.sum(v*v+c*a,axis=-1,keepdims=True)-speed*speed)/norm
    ddu = (a-u*curvature-2*du*speed)/norm
    def angular(d):
        return 2*(u[...,3:]*d[...,:3]-d[...,3:]*u[...,:3]
                  +np.cross(u[...,:3],d[...,:3]))
    return u, angular(du), angular(ddu)


def exported_rotation_maps(c, v, a, coefficients, basis, first, second,
                           scale=1., epsilon=1e-5):
    """Return 3-by-(3 * coefficient count) pose/omega/alpha Jacobians."""
    coefficients=np.asarray(coefficients)
    axes=np.eye(3)[None]
    vector=coefficients[:,None,:3]
    # Derivative of Exp(world_delta) * coefficient quaternion.
    derivative=np.concatenate((coefficients[:,None,3:]*axes+np.cross(axes,vector),
                                -coefficients[:,:3,None]),axis=-1)*(.5*scale)
    dc=(np.asarray(basis)[:,None,None]*derivative).reshape(-1,4)
    dv=(np.asarray(first)[:,None,None]*derivative).reshape(-1,4)
    da=(np.asarray(second)[:,None,None]*derivative).reshape(-1,4)
    u,_,_=quaternion_kinematics(c,v,a)
    du=(dc-u*np.sum(dc*u,axis=-1,keepdims=True))/np.linalg.norm(c)
    rotation=2*(u[3]*du[:,:3]-du[:,3:]*u[:3]+np.cross(u[:3],du[:,:3]))
    plus=quaternion_kinematics(c+epsilon*dc,v+epsilon*dv,a+epsilon*da)
    minus=quaternion_kinematics(c-epsilon*dc,v-epsilon*dv,a-epsilon*da)
    return rotation.T,((plus[1]-minus[1])/(2*epsilon)).T,((plus[2]-minus[2])/(2*epsilon)).T
