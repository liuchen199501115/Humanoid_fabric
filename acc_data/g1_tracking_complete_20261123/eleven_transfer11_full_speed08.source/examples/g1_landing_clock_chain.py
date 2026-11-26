"""Compose causal landing pauses with exact source-clock derivatives."""
import math


class LandingClockChain:
    def __init__(self, schedules, resume_ramp=.1):
        from early_resume_clock import EarlyResumeLandingClock
        self.clocks = []
        previous_end, previous_count = -1., 0
        for start, end, count in schedules:
            if (not all(math.isfinite(x) for x in (start, end, count))
                    or start < previous_end or count != int(count) or count <= previous_count):
                raise ValueError('Landing pauses need ordered source intervals and increasing positive landing counts')
            self.clocks.append(EarlyResumeLandingClock(start, end, resume_ramp, int(count)))
            previous_end, previous_count = end, int(count)

    @staticmethod
    def compose(clock, state):
        phase, rate, acceleration = state
        position, phase_rate, phase_acceleration = clock.sample(phase)
        return position, phase_rate*rate, phase_acceleration*rate**2+phase_rate*acceleration

    def update(self, state, verified_landings):
        for clock in self.clocks:
            clock.update(state[0], verified_landings)
            state = self.compose(clock, state)

    def sample(self, state):
        for clock in self.clocks:
            state = self.compose(clock, state)
        return state
