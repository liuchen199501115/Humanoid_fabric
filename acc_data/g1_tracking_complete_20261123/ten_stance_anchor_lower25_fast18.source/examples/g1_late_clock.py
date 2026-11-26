"""Optional smooth fourth speed transition, with exact clock derivatives."""
import math


def late_speed_transition(base_state, time, *, start, ramp, previous_speed, speed):
    if (not all(math.isfinite(x) for x in (*base_state,time,start,ramp,previous_speed,speed))
            or start<0 or ramp<=0 or previous_speed<=0 or speed<=0):
        raise ValueError('Invalid late source-clock transition')
    if time<=start:
        return base_state
    delta=speed-previous_speed
    if time>=start+ramp:
        correction=(delta*(time-start-ramp/2),delta,0.)
    else:
        u=(time-start)/ramp
        correction=(delta*ramp*(2.5*u**4-3*u**5+u**6),
                    delta*u**3*(10-15*u+6*u*u),delta*30*u*u*(1-u)**2/ramp)
    return tuple(a+b for a,b in zip(base_state,correction))
