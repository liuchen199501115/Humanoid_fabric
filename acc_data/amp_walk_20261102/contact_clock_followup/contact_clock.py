"""Experimental smooth source-time clock driven by measured support loads.

Only source timing changes. Two cascaded first-order filters give continuous
source speed and acceleration despite discrete contact decisions. Kinematics
must use frame rate and acceleration returned by this clock (chain rule).
"""
import math
import numpy as np


class ContactTrackingClock:
    def __init__(self, frame=0., fps=200., rate=1., minimum_rate=.45, tau=.06, start=1.82, include_landing=False):
        if not (0 < minimum_rate <= rate and tau > 0 and fps > 0):
            raise ValueError('Invalid clock parameters')
        self.frame, self.fps = float(frame), float(fps)
        self.nominal_rate = self.rate = self.filtered_target = float(rate)
        self.minimum_rate, self.tau, self.start = minimum_rate, tau, start
        self.include_landing = include_landing
        self.acceleration = 0.
        self.target = rate
        self.mismatch = False

    def advance(self, dt, wall_time, planned_mode, forces):
        if dt < 0 or not np.isfinite([dt, wall_time, *forces]).all():
            raise ValueError('Invalid clock observation')
        forces = np.maximum(forces, 0.)
        self.mismatch = (wall_time >= self.start and planned_mode in (1, 2)
                         and (forces[planned_mode-1] < 100.
                              or forces[2-planned_mode] > max(10., .1*forces.sum())))
        if self.include_landing and wall_time >= self.start and planned_mode == 3:
            self.mismatch = bool(forces.min() < 20.)
        target = self.minimum_rate if self.mismatch else self.nominal_rate
        d, e = self.rate-target, self.filtered_target-target
        u = dt/self.tau
        decay = math.exp(-u)
        # Exact integration while this interval's measured target is constant.
        integral = target*dt + d*self.tau*(-math.expm1(-u))
        integral += e*self.tau*(1.-(1.+u)*decay)
        self.frame += self.fps*integral
        self.rate = target+(d+e*u)*decay
        self.filtered_target = target+e*decay
        self.acceleration = (self.filtered_target-self.rate)/self.tau
        self.target = target
        return self.frame, self.fps*self.rate, self.fps*self.acceleration
