"""Vertical reference with a separate, bounded initial-velocity braking phase."""
import math


class BrakedHeightProfile:
    """C2 height path: stop measured vertical speed, then reach landing height.

    The first segment bounds acceleration by ``braking_acceleration``. The
    remaining segment is a rest-to-rest quintic; its acceleration must also fit
    that bound. This is a reference profile, not a whole-body feasibility test.
    """
    def __init__(self, height, velocity, goal, duration, braking_acceleration):
        values = (height, velocity, goal, duration, braking_acceleration)
        if (not all(math.isfinite(x) for x in values) or min(height, goal) <= .1
                or duration <= 0 or braking_acceleration <= 0):
            raise ValueError('Invalid bounded height transition')
        self.height, self.velocity, self.goal, self.duration = map(float, values[:4])
        self.brake_time = 1.5*abs(velocity)/braking_acceleration
        if self.brake_time >= duration:
            raise ValueError('Not enough time to brake vertical velocity before landing')
        self.stopped_height = height + .5*velocity*self.brake_time
        self.remaining = duration-self.brake_time
        # max |d2 smoothstep5 / ds2| = 10/sqrt(3).
        peak = (10/math.sqrt(3))*abs(goal-self.stopped_height)/self.remaining**2
        if self.stopped_height <= .1 or peak > braking_acceleration+1e-12:
            raise ValueError('Landing height cannot be reached within the acceleration bound')

    def sample(self, time):
        if not math.isfinite(time):
            raise ValueError('Height sample time must be finite')
        t = max(0., float(time))
        if t >= self.duration:
            return self.goal, 0., 0.
        if t < self.brake_time:
            s = t/self.brake_time
            return (self.height+self.velocity*self.brake_time*(s-s**3+.5*s**4),
                    self.velocity*(1-3*s*s+2*s**3),
                    self.velocity*(-6*s+6*s*s)/self.brake_time)
        s = (t-self.brake_time)/self.remaining
        delta = self.goal-self.stopped_height
        return (self.stopped_height+delta*(10*s**3-15*s**4+6*s**5),
                delta*(30*s*s-60*s**3+30*s**4)/self.remaining,
                delta*(60*s-180*s*s+120*s**3)/self.remaining**2)
