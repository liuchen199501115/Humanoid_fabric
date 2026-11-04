"""Speed bound for the actual retimed PCHIP foot path plus startup bridge."""
import math
import numpy as np
from numpy.polynomial import Polynomial
from scipy.optimize import brentq

from .bridge import ReferenceBridge
from .phase_clock import monotone_swing_clock


def source_swing_speed_bound(coefficients,fps,start_frame,start_rate,end_frame,
                             duration,position,velocity,height_correction=0.,bridge_duration=None,end_rate=0.):
    """Conservative continuous bound, including between source sample knots.

    Compose the cubic source curve with the quintic clock, terminal height
    correction, and measured-state bridge. Bernstein control points bound the
    velocity norm on each polynomial interval by convexity. Splitting every
    interval in four keeps the bound tight without sampling away sharp peaks.
    ``coefficients`` are MotionInterpolator's cubic coefficients for XYZ.
    """
    c=np.asarray(coefficients,dtype=float)
    if c.ndim!=3 or c.shape[0]!=4 or c.shape[2]!=3 or not np.isfinite(c).all():
        raise ValueError('Expected finite XYZ PCHIP coefficients [4,T-1,3]')
    if not np.isfinite(fps) or fps<=0 or not 0<=start_frame<end_frame<=c.shape[1]:
        raise ValueError('Invalid source frame interval or FPS')
    clock=monotone_swing_clock(start_frame/fps,start_rate,end_frame/fps,duration,end_rate=end_rate)
    # All polynomials below use normalized physical time s in [0,1].
    frame=Polynomial([start_frame,start_rate*fps*duration,0.,
                      *(clock.coefficients*fps*np.array([duration**3,duration**4,duration**5]))])
    index=min(int(start_frame),c.shape[1]-1);fraction=start_frame-index
    initial_position=((c[0,index]*fraction+c[1,index])*fraction+c[2,index])*fraction+c[3,index]
    initial_velocity=((3*c[0,index]*fraction+2*c[1,index])*fraction+c[2,index])*(fps*start_rate)
    initial_acceleration=(6*c[0,index]*fraction+2*c[1,index])*(fps*start_rate)**2
    if bridge_duration is not None and (not np.isfinite(bridge_duration) or bridge_duration<=0):
        raise ValueError('Bridge duration must be positive and finite')
    bridge_time=duration if bridge_duration is None else min(duration,bridge_duration)
    bridge=ReferenceBridge(np.asarray(position)-initial_position,
        np.asarray(velocity)-initial_velocity,-initial_acceleration,bridge_time)
    bridge_end=bridge_time/duration
    bridge_polynomials=[Polynomial(bridge.coefficients[:,j]/bridge_end**np.arange(6)) for j in range(3)]
    ground=height_correction*Polynomial([0,0,0,10,-15,6])
    boundaries=[0.]
    for knot in range(int(math.floor(start_frame))+1,int(math.ceil(end_frame))):
        boundaries.append(brentq(lambda s:frame(s)-knot,boundaries[-1],1.,xtol=1e-13))
    boundaries.append(1.)
    if bridge_end<1.:boundaries=sorted(set(boundaries+[bridge_end]))
    maximum=0.
    for left,right in zip(boundaries[:-1],boundaries[1:]):
        index=min(int(frame((left+right)/2)),c.shape[1]-1)
        for lo,hi in zip(np.linspace(left,right,5)[:-1],np.linspace(left,right,5)[1:]):
            affine=Polynomial([lo,hi-lo]);source=frame(affine)-index
            source_derivative=frame.deriv()(affine)
            velocity_polynomials=[]
            for j in range(3):
                # Differentiate in global normalized time, then restrict the
                # derivative to this interval. A source start within roundoff
                # of a knot can produce hi == lo after brentq/linspace; that
                # interval still has a finite limiting velocity, not 0/0.
                derivative=((3*c[0,index,j]*source+2*c[1,index,j])*source+c[2,index,j])*source_derivative
                if (lo+hi)/2 < bridge_end:
                    derivative+=bridge_polynomials[j].deriv()(affine)
                if j==2:derivative-=ground.deriv()(affine)
                velocity_polynomials.append(derivative/duration)
            degree=max(len(poly.coef)-1 for poly in velocity_polynomials)
            bernstein=np.zeros((degree+1,3))
            for j,poly in enumerate(velocity_polynomials):
                for k in range(degree+1):
                    bernstein[k,j]=sum(poly.coef[i]*math.comb(k,i)/math.comb(degree,i)
                                       for i in range(min(k,len(poly.coef)-1)+1))
            maximum=max(maximum,float(np.linalg.norm(bernstein,axis=1).max()))
    return maximum


def minimum_source_swing_duration(coefficients,fps,start_frame,start_rate,end_frame,
                                  position,velocity,height_correction,minimum,maximum_speed,bridge_duration=None,end_rate=0.):
    if not np.isfinite([minimum,maximum_speed]).all() or min(minimum,maximum_speed)<=0:
        raise ValueError('Expected positive finite duration and speed limit')
    if np.linalg.norm(velocity)>maximum_speed:
        raise RuntimeError('Measured swing velocity already exceeds the planning speed limit')
    duration=minimum
    for _ in range(16):
        try:
            bound=source_swing_speed_bound(coefficients,fps,start_frame,start_rate,end_frame,
                                          duration,position,velocity,height_correction,bridge_duration,end_rate)
        except ValueError as exc:
            raise RuntimeError(f'Cannot plan a monotone source swing within the speed limit: {exc}') from exc
        if bound<=maximum_speed*(1.-1e-4):return duration
        duration*=max(1.02,min(1.5,bound/maximum_speed*1.001))
    raise RuntimeError('No source swing duration satisfies the continuous speed bound')
