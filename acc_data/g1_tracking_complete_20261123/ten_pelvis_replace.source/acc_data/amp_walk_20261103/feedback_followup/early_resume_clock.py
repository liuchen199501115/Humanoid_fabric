"""Causal C2 phase recovery after a physically certified landing.

Remove the current deceleration without reversing phase, then restore the
nominal speed over the configured interval. No dynamics rollout or horizon
optimization is used; the existing measured landing audit opens the gate.
"""
from landing_pause_clock import LandingPauseClock

class EarlyResumeLandingClock(LandingPauseClock):
    def update(self,time,verified_landings):
        if self.release is not None or time < self.start or verified_landings < self.required:
            return
        phase,rate,acceleration=super().sample(time)
        self.release=float(time)
        self.release_state=(phase,rate,acceleration)
        self.brake_duration=min(.25*self.up,1.5*rate/(-acceleration)) if acceleration<0 else 0.
        self.bridge_duration=self.brake_duration+self.up

    def sample(self,time):
        if self.release is None or time < self.release:
            return super().sample(time)
        phase,rate,acceleration=self.release_state
        elapsed=time-self.release;d=self.brake_duration;T=self.up
        if d>0:
            if elapsed < d:
                return (phase+rate*elapsed+.5*acceleration*elapsed**2-acceleration*elapsed**3/(6*d),
                        rate+acceleration*elapsed-.5*acceleration*elapsed**2/d,
                        acceleration*(1-elapsed/d))
            phase+=rate*d+acceleration*d*d/3
            rate+=.5*acceleration*d
            elapsed-=d
        if elapsed>=T:
            return phase+.5*T*(rate+1)+(elapsed-T),1.,0.
        u=elapsed/T
        return (phase+T*(rate*u+(1-rate)*(u**3-.5*u**4)),
                rate+(1-rate)*(3*u*u-2*u**3),
                (1-rate)*(6*u-6*u*u)/T)
