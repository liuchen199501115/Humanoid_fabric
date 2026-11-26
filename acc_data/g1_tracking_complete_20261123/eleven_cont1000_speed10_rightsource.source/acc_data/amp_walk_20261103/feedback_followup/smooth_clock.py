"""Monotone source clock with analytic rate and acceleration."""
import math
def _single_clock(t,start,ramp,speed):
 if not all(math.isfinite(x) for x in (t,start,ramp,speed)) or start<0 or ramp<=0 or speed<=0:raise ValueError('Invalid source clock')
 if t<=start:return t,1.,0.
 if t>=start+ramp:return start+ramp*(1+speed)/2+speed*(t-start-ramp),speed,0.
 u=(t-start)/ramp;h=u**3*(10-15*u+6*u*u)
 return t+(speed-1)*ramp*(2.5*u**4-3*u**5+u**6),1+(speed-1)*h,(speed-1)*30*u*u*(1-u)**2/ramp


def smooth_clock(t,start,ramp,speed,second_start=100.,second_ramp=.3,second_speed=None,third_start=100.,third_ramp=.2,third_speed=None):
 result=_single_clock(t,start,ramp,speed)
 if second_speed is None:
  if third_speed is not None:raise ValueError('Third transition requires a second transition')
  return result
 if (not all(math.isfinite(x) for x in (second_start,second_ramp,second_speed))
     or second_start<start+ramp or second_ramp<=0 or second_speed<=0):
  raise ValueError('Invalid second source-clock transition')
 if t<=second_start:return result
 delta=second_speed-speed
 if t>=second_start+second_ramp:
  correction=(delta*(t-second_start-second_ramp/2),delta,0.)
 else:
  u=(t-second_start)/second_ramp;h=u**3*(10-15*u+6*u*u)
  correction=(delta*second_ramp*(2.5*u**4-3*u**5+u**6),delta*h,delta*30*u*u*(1-u)**2/second_ramp)
 result=tuple(x+y for x,y in zip(result,correction))
 if third_speed is None:return result
 if (not all(math.isfinite(x) for x in (third_start,third_ramp,third_speed))
     or third_start<second_start+second_ramp or third_ramp<=0 or third_speed<=0):
  raise ValueError('Invalid third source-clock transition')
 if t<=third_start:return result
 delta=third_speed-second_speed
 if t>=third_start+third_ramp:
  correction=(delta*(t-third_start-third_ramp/2),delta,0.)
 else:
  u=(t-third_start)/third_ramp;h=u**3*(10-15*u+6*u*u)
  correction=(delta*third_ramp*(2.5*u**4-3*u**5+u**6),delta*h,delta*30*u*u*(1-u)**2/third_ramp)
 return tuple(x+y for x,y in zip(result,correction))
