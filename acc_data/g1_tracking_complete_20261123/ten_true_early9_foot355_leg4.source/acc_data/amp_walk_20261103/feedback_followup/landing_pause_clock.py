"""C2 source phase clock: decelerate to one source landing, verify, resume."""
import math

class LandingPauseClock:
 def __init__(self,start=3.904,end=4.01,resume_ramp=.2,required_landings=10):
  if not all(math.isfinite(x) for x in [start,end,resume_ramp]) or not end>start>=0 or resume_ramp<=0:
   raise ValueError('Invalid landing pause clock boundaries')
  self.start=start;self.end=end;self.down=2*(end-start);self.stop_time=start+self.down;self.up=resume_ramp;self.required=required_landings;self.release=None
 def update(self,time,verified_landings):
  if self.release is None and time>=self.stop_time and verified_landings>=self.required:self.release=float(time)
 def sample(self,time):
  if time<=self.start:return float(time),1.,0.
  if time<self.stop_time:
   u=(time-self.start)/self.down;h=u**3*(10-15*u+6*u*u);integral=2.5*u**4-3*u**5+u**6
   return self.start+self.down*(u-integral),1-h,-30*u*u*(1-u)**2/self.down
  if self.release is None or time<=self.release:return self.end,0.,0.
  elapsed=time-self.release
  if elapsed>=self.up:return self.end+elapsed-.5*self.up,1.,0.
  u=elapsed/self.up;h=u**3*(10-15*u+6*u*u);integral=2.5*u**4-3*u**5+u**6
  return self.end+self.up*integral,h,30*u*u*(1-u)**2/self.up
