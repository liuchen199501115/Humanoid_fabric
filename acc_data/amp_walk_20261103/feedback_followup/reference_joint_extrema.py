"""Exact actuated-joint extrema on a constant-speed, fully blended spline."""
import numpy as np
from scipy.interpolate import PPoly

def joint_extrema(spline,start,end,frame_rate,limits,speed_limits):
    if not (np.isfinite([start,end,frame_rate]).all() and end>start and frame_rate>0):
        raise ValueError('Expected a finite interval and positive source rate')
    if np.shape(limits)!=(29,2) or np.shape(speed_limits)!=(29,) or np.any(np.asarray(speed_limits)<=0):
        raise ValueError('Expected G1 position and velocity limits')
    minimum=[];maximum=[];speed=[]
    for j in range(29):
        polynomial=PPoly.from_spline((spline.t,spline.c[:,7+j],spline.k))
        def extrema(function):
            roots=function.derivative().roots(extrapolate=False)
            nodes=np.r_[start,end,roots[np.isfinite(roots)&(roots>=start)&(roots<=end)]]
            values=function(nodes)
            return float(np.min(values)),float(np.max(values))
        lo,hi=extrema(polynomial);vl,vh=extrema(polynomial.derivative())
        minimum.append(lo);maximum.append(hi);speed.append(max(abs(vl),abs(vh))*frame_rate)
    minimum=np.array(minimum);maximum=np.array(maximum);speed=np.array(speed)
    return dict(minimum_continuous_joint_margin=float(np.minimum(minimum-limits[:,0],limits[:,1]-maximum).min()),maximum_continuous_joint_velocity_fraction=float((speed/speed_limits).max()),joint_minimum=minimum.tolist(),joint_maximum=maximum.tolist(),joint_maximum_speed=speed.tolist())
