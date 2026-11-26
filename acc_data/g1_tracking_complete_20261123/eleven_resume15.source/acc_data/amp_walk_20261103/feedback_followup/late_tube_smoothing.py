"""C2 transition between source-pose spline scales; no predictive controller."""
import math
from blended_spline import BlendedSplineReference
from fabrics_sim.reference.interpolation import MotionInterpolator

def transition_tube_spline(previous,trajectory,*,stride,start_frame,ramp_frames):
    if not isinstance(previous,BlendedSplineReference):
        raise ValueError('Late smoothing requires the existing source spline')
    if not math.isfinite(start_frame) or start_frame<previous.start+previous.ramp:
        raise ValueError('Late smoothing must begin after the initial spline transition')
    curve=BlendedSplineReference(MotionInterpolator(trajectory),trajectory,stride=stride,
        start_frame=start_frame,ramp_frames=ramp_frames)
    curve.original=previous
    curve.raw=previous.smooth
    return curve
