"""In-memory spline preparation shared by the online runner and offline audit."""
import math
import xml.etree.ElementTree as ET
import numpy as np
from .interpolation import MotionInterpolator


def scalar_derivative_peaks(coefficients):
    a, b, c, _ = coefficients
    u = np.divide(-b, 3*a, out=np.zeros_like(a), where=np.abs(a)>1e-15)
    vertex = np.where((u>0)&(u<1), np.abs(3*a*u*u+2*b*u+c), 0.)
    velocity = np.maximum.reduce([np.abs(c), np.abs(3*a+2*b+c), vertex]).max(axis=0)
    acceleration = np.maximum(np.abs(2*b), np.abs(6*a+2*b)).max(axis=0)
    return velocity, acceleration


def prepare_reference(trajectory, fps, joint_names, urdf, acceleration_cap, requested_speed=1., *, method="bspline"):
    """Return the actual online spline and a constant source-clock speed cap.

    Limits apply to scalar joint derivatives at constant speed, not the extra
    acceleration caused by online clock ramps, nor WBC feasibility.
    """
    if method not in ('bspline', 'pchip'):
        raise ValueError('Preparation requires a bounded bspline or pchip interpolator')
    if (not all(math.isfinite(x) and x>0 for x in (fps, requested_speed))
            or (acceleration_cap is not None and (not math.isfinite(acceleration_cap) or acceleration_cap <= 0))):
        raise ValueError('fps, acceleration cap and requested speed must be positive and finite')
    if len(trajectory)<4 or len(joint_names)!=29 or len(set(joint_names))!=29:
        raise ValueError('Expected at least four G1 poses and 29 unique joint names')
    limits = {j.attrib['name']: j.find('limit') for j in ET.parse(urdf).getroot().findall('joint')}
    lower = trajectory.new_tensor([float(limits[n].attrib['lower']) for n in joint_names])
    upper = trajectory.new_tensor([float(limits[n].attrib['upper']) for n in joint_names])
    vmax = np.array([float(limits[n].attrib['velocity']) for n in joint_names])
    if ((trajectory[:,7:]<lower-1e-6)|(trajectory[:,7:]>upper+1e-6)).any():
        raise ValueError('Source control points violate joint position limits')
    sampler = MotionInterpolator(trajectory, method=method)
    vp, ap = scalar_derivative_peaks(sampler.coefficients.detach().cpu().double().numpy()[:,:,7:])
    vp *= fps
    ap *= fps**2
    speed = min(requested_speed, float(np.min(vmax/np.maximum(vp,1e-12))),
                math.sqrt(acceleration_cap/max(float(ap.max()),1e-12)) if acceleration_cap is not None else float('inf'))
    report = dict(method=method, frames=len(trajectory), fps=float(fps),
        requested_speed=requested_speed, playback_speed=speed,
        planning_acceleration_cap=acceleration_cap, acceleration_cap_enabled=acceleration_cap is not None,
        constant_rate_peak_joint_velocity=float(vp.max()*speed),
        constant_rate_peak_joint_acceleration=float(ap.max()*speed**2),
        full_tracking_passed=False,
        limitations=['Clock derivatives are applied online; preparation is not an online acceleration guarantee',
                    'Contact, collision and dynamics feasibility require separate controller checks',
                    'Missing source joint-name metadata remains a loader assumption'])
    return sampler, report
