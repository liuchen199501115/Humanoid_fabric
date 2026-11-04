"""Measured-state gates for one quasi-static lift and touchdown experiment."""
from dataclasses import dataclass


def smooth_height(time, duration, start, end):
    s = min(1., max(0., time / duration))
    delta = end - start
    return (start + delta * s**3 * (10 - 15*s + 6*s*s),
            delta * 30*s*s*(1-s)**2 / duration,
            delta * 60*s*(1-s)*(1-2*s) / duration**2)


@dataclass
class ContactCycle:
    phase: str = "transfer"
    phase_start: float = 0.
    confirmed: int = 0
    height: float = .04
    lift_duration: float = 2.
    lower_duration: float = 3.
    ever_clear: bool = False
    landing_time: float | None = None

    def update(self, time, *, ready, clearance, swing_load, swing_speed, settled):
        age = time - self.phase_start
        condition = False
        following = self.phase
        if self.phase == "transfer":
            condition, following = ready, "lift"
        elif self.phase == "lift":
            condition = age >= self.lift_duration and clearance >= .025 and swing_load < 10 and swing_speed < .03
            following = "hold"
        elif self.phase == "hold":
            condition = age >= 1. and clearance >= .025 and swing_load < 10 and swing_speed < .03
            following = "lower"
        elif self.phase == "lower":
            condition = self.ever_clear and swing_load >= 20 and swing_speed < .05
            following = "settle"
        elif self.phase == "settle":
            condition, following = age >= 1. and settled, "done"
        self.confirmed = self.confirmed + 1 if condition else 0
        if self.confirmed >= 10 and self.phase != "done":
            if following == "hold":
                self.ever_clear = True
            if following == "settle":
                self.landing_time = time
            self.phase, self.phase_start, self.confirmed = following, time, 0
        return self.phase

    def reference(self, time):
        age = time - self.phase_start
        if self.phase == "lift":
            return smooth_height(age, self.lift_duration, 0., self.height)
        if self.phase == "hold":
            return self.height, 0., 0.
        if self.phase == "lower":
            return smooth_height(age, self.lower_duration, self.height, 0.)
        return 0., 0., 0.
