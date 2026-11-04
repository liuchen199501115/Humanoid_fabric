"""Smooth source-time interpolation between measured step transitions."""
import numpy as np


def monotone_swing_clock(start,rate,end,duration,end_rate=0.):
    """Retain terminal source speed as far as a monotone quintic permits."""
    if not np.isfinite(end_rate) or end_rate<0:
        raise ValueError('Terminal rate must be finite and nonnegative')
    for fraction in (1.,.75,.5,0.):
        try:return SwingClock(start,rate,end,duration,end_rate*fraction)
        except ValueError:
            if fraction==0.:raise


class SwingClock:
    def __init__(self,start,rate,end,duration,end_rate=0.):
        if not np.isfinite([start,rate,end,duration,end_rate]).all() or end<=start or min(rate,end_rate)<0 or duration<=0:
            raise ValueError('Invalid swing clock boundary conditions')
        self.start=float(start);self.rate=float(rate);self.end=float(end);self.duration=float(duration)
        self.end_rate=float(end_rate)
        t=self.duration
        self.coefficients=np.linalg.solve(
            [[t**3,t**4,t**5],[3*t*t,4*t**3,5*t**4],[6*t,12*t*t,20*t**3]],
            [end-start-rate*t,end_rate-rate,0.])
        # Check derivative extrema analytically, including the endpoints.
        c3,c4,c5=self.coefficients
        critical=np.roots([20*c5,12*c4,6*c3])
        times=[0.,t]+[float(r.real) for r in critical if abs(r.imag)<1e-9 and 0<r.real<t]
        if min(self.sample(at)[1] for at in times)<-1e-8:
            raise ValueError('Swing clock would reverse source progress')

    def sample(self,elapsed):
        if elapsed>=self.duration:return self.end,self.end_rate,0.
        t=max(0.,float(elapsed));c3,c4,c5=self.coefficients
        return (self.start+self.rate*t+c3*t**3+c4*t**4+c5*t**5,
                self.rate+3*c3*t*t+4*c4*t**3+5*c5*t**4,
                6*c3*t+12*c4*t*t+20*c5*t**3)
