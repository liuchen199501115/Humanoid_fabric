"""Retiming of a full-body reference with analytic chain-rule derivatives."""
import math
from .whole_body import WholeBodyReference


class LandingRetimedReference:
    """Preserve the spatial path and initial q/v/a while slowing at touchdown.

    The quintic clock has initial rate 1, terminal rate in [0,1], and zero
    clock acceleration at both boundaries. This is not a feasibility proof.
    """
    def __init__(self, source, terminal_rate):
        if (not math.isfinite(terminal_rate) or not 0 <= terminal_rate <= 1
                or not math.isfinite(source.duration) or source.duration <= 0):
            raise ValueError('Retiming requires positive duration and terminal rate in [0,1]')
        self.source = source
        self.terminal_rate = float(terminal_rate)
        self.duration = source.duration

    def __getattr__(self, name):
        return getattr(self.source, name)

    @property
    def q(self):
        return self.source.q

    @q.setter
    def q(self, value):
        self.source.q = value

    def clock(self, time):
        if not math.isfinite(time) or not 0 <= time <= self.duration:
            raise ValueError('Retimed sample is outside its interval')
        u = time/self.duration
        d = self.terminal_rate-1.
        phase = u+d*(-4*u**3+7*u**4-3*u**5)
        rate = 1+d*(-12*u**2+28*u**3-15*u**4)
        acceleration = d*(-24*u+84*u**2-60*u**3)/self.duration
        # Avoid floating-point overshoot at the two exact endpoints.
        if time == 0.:return 0., 1., 0.
        if time == self.duration:return self.duration, self.terminal_rate, 0.
        return phase*self.duration, rate, acceleration

    def sample(self, time):
        source_time, rate, acceleration = self.clock(time)
        reference = self.source.sample(source_time)
        return WholeBodyReference(reference.q, reference.v*rate,
                                  reference.a*rate**2+reference.v*acceleration)

    def progress(self, time):
        source_time, rate, acceleration = self.clock(time)
        phase, first, second, auxiliary = self.source.progress(source_time)
        return phase, first*rate, second*rate**2+first*acceleration, auxiliary
