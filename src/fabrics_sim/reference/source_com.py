"""World-space source COM preferences on the same retimed swing clock."""
import numpy as np
from .phase_clock import monotone_swing_clock


class SourceComReference:
    def __init__(self, interpolator, fps, start_frame, rate, landing_frame, end_rate,
                 momentum_interpolator=None):
        self.interpolator = interpolator
        self.fps = fps
        self.start = start_frame / fps
        self.rate = rate
        self.end = landing_frame / fps
        self.end_rate = end_rate
        self.momentum_interpolator = momentum_interpolator

    def __call__(self, duration, fractions):
        try:
            clock = monotone_swing_clock(self.start, self.rate, self.end, duration, self.end_rate)
        except ValueError:
            return None
        frames = np.array([clock.sample(float(s)*duration)[0]*self.fps for s in fractions])
        positions = self.interpolator(frames)[:,:2]
        velocity = self.interpolator(self.end*self.fps, 1)[:2]*self.fps*clock.end_rate
        return positions, velocity

    def momentum(self, duration, fractions):
        """Source angular momentum scales with the actual source-clock rate."""
        if self.momentum_interpolator is None:
            raise ValueError('A source angular momentum interpolator is required')
        try:
            clock = monotone_swing_clock(self.start, self.rate, self.end, duration, self.end_rate)
        except ValueError:
            return None
        values = np.asarray([clock.sample(float(s)*duration) for s in fractions])
        return self.momentum_interpolator(values[:,0]*self.fps)*values[:,1,None]


class SourceComContinuation:
    """Remaining source preferences on an existing clock, without retiming it."""
    def __init__(self,reference,clock,elapsed):
        if not np.isfinite(elapsed) or not 0<=elapsed<clock.duration:
            raise ValueError('Continuation must begin inside the existing swing')
        self.interpolator=reference.interpolator
        self.momentum_interpolator=reference.momentum_interpolator
        self.fps=reference.fps;self.clock=clock;self.elapsed=float(elapsed)

    def _samples(self,duration,fractions):
        if (not np.isfinite(duration) or duration<=0
                or self.elapsed+duration>self.clock.duration+1e-8):
            return None
        return np.asarray([self.clock.sample(self.elapsed+float(s)*duration) for s in fractions])

    def __call__(self,duration,fractions):
        samples=self._samples(duration,fractions)
        if samples is None:return None
        position=self.interpolator(samples[:,0]*self.fps)[:,:2]
        end,rate,_=self.clock.sample(self.elapsed+duration)
        velocity=self.interpolator(end*self.fps,1)[:2]*self.fps*rate
        return position,velocity

    def momentum(self,duration,fractions):
        samples=self._samples(duration,fractions)
        if samples is None:return None
        if self.momentum_interpolator is None:raise ValueError('Source momentum is required')
        return self.momentum_interpolator(samples[:,0]*self.fps)*samples[:,1,None]
