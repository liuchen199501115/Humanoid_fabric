"""Continuous source phase with measured landing gates and bounded braking.

This retimes one shared whole-body reference. It does not change source poses
or manufacture contacts, and is not a dynamics feasibility certificate.
"""
import math
import numpy as np


class ContactMotionClock:
    def __init__(self, events, fps, speed=1., braking_time=.25, confirm_seconds=.006):
        if not events or not all(math.isfinite(x) and x>0 for x in (fps,speed,braking_time,confirm_seconds)):
            raise ValueError('Expected landing events and positive finite clock parameters')
        self.events = events
        self.frame = 0.
        self.nominal_rate = fps*speed
        self.rate = self.nominal_rate
        self.acceleration = 0.
        self.max_acceleration = self.nominal_rate/braking_time
        self.confirm_seconds = confirm_seconds
        self.loaded_seconds = 0.
        self.ready = False
        self.airborne = False
        self.index = 0
        self.waiting = False
        if self.rate**2/(2*self.max_acceleration) > events[0]['end_frame']:
            raise ValueError('Initial landing is too near for the specified phase braking time')

    def update(self, forces, dt):
        forces = np.asarray(forces)
        if forces.shape != (2,) or not np.isfinite(forces).all() or not math.isfinite(dt) or dt<=0:
            raise ValueError('Invalid clock observation')
        self.waiting = False
        if self.index < len(self.events):
            event = self.events[self.index]
            incoming = 1-event['support']
            self.airborne |= forces[incoming]<10.
            self.loaded_seconds = self.loaded_seconds+dt if self.airborne and forces[incoming]>=20. else 0.
            self.ready |= self.loaded_seconds+1e-9 >= self.confirm_seconds
            if self.ready and self.frame >= event['end_frame']-1e-8:
                self.index += 1
                self.ready = False
                self.airborne = False
                self.loaded_seconds = 0.
        v, a = self.rate, self.max_acceleration
        u = min(self.nominal_rate, v+a*dt)
        if self.index < len(self.events) and not self.ready:
            end = float(self.events[self.index]['end_frame'])
            distance = max(0.,end-self.frame)
            # Next position plus its stopping distance must remain before the
            # landing gate: (v+u)*dt/2 + u*u/(2*a) <= distance.
            discriminant = (a*dt)**2 - 4*a*dt*v + 8*a*distance
            cap = (-a*dt + math.sqrt(max(0.,discriminant)))/2
            u = min(u, max(0.,cap))
            if u < v-a*dt-1e-6:
                raise RuntimeError('Source phase cannot brake before the unconfirmed landing')
            if u <= 1e-10:
                self.frame = min(end, self.frame+v*v/(2*a))
                self.rate = self.acceleration = 0.
                self.waiting = True
                return self.frame, self.rate, self.acceleration
        self.acceleration = (u-v)/dt
        self.frame += (v+u)*dt/2
        self.rate = u
        return self.frame, self.rate, self.acceleration
