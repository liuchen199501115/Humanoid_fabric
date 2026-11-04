"""Bounded source steps: exact source endpoints, measured flight/landing evidence."""
import numpy as np
from .source_sequence import next_source_event


def select_steps(modes, count, start=0):
    if type(count) is not int or count < 1:
        raise ValueError('Step count must be a positive integer')
    modes = np.asarray(modes)
    if not 0 <= start < len(modes) or modes[start] != 3:
        raise ValueError('Bounded steps must start in double support')
    events = []
    boundary = start
    for _ in range(count):
        event = next_source_event(modes, boundary)
        if event['kind'] != 'swing' or event['inserted_double_support']:
            raise ValueError('Requested steps need source double-support landings')
        if events and event['support'] == events[-1]['support']:
            raise ValueError('Requested source steps do not alternate')
        events.append(event)
        boundary = event['end_frame']
    return events


class MeasuredSteps:
    def __init__(self, events, goals, tolerance=.04):
        self.events = events
        self.goals = np.asarray(goals, dtype=float)
        if self.goals.shape != (len(events), 3) or not np.isfinite(self.goals).all():
            raise ValueError('Expected one finite world-frame landing goal per step')
        self.tolerance = tolerance
        self.airborne = False
        self.loaded_seconds = 0.
        self.landings = []

    @property
    def complete(self):
        return len(self.landings) == len(self.events)

    def update(self, frame, time, mode, forces, clearances, feet, dt):
        forces, clearances, feet = map(np.asarray, (forces, clearances, feet))
        if (forces.shape != (2,) or clearances.shape != (2,) or feet.shape != (2, 3)
                or not all(np.isfinite(x).all() for x in (forces, clearances, feet))
                or not np.isfinite([frame,time,dt]).all() or dt <= 0):
            raise ValueError('Invalid measured step observation')
        if self.complete:
            return True
        event = self.events[len(self.landings)]
        swing, support = 1-event['support'], event['support']
        if frame < event['takeoff_frame']-1:
            return False
        if (mode == support+1 and forces[support]>=100. and forces[swing]<10.
                and clearances[swing]>=.02):
            self.airborne = True
        error = float(np.linalg.norm(feet[swing]-self.goals[len(self.landings)]))
        ready = (self.airborne and frame>=event['end_frame']-1. and mode==3
                 and (forces>=20.).all() and error<=self.tolerance)
        self.loaded_seconds = self.loaded_seconds+dt if ready else 0.
        if self.loaded_seconds >= .05-1e-9:
            self.landings.append(dict(step=len(self.landings)+1, swing='left' if swing==0 else 'right',
                source_frame=float(frame), time=float(time), landing_error_m=error,
                goal=self.goals[len(self.landings)].tolist(), measured_load_n=float(forces[swing])))
            self.airborne = False
            self.loaded_seconds = 0.
        return self.complete

    def report(self):
        return dict(scope='bounded source steps, unchanged landing targets', expected_steps=len(self.events),
            measured_steps=len(self.landings), landings=self.landings,
            measured_steps_complete=self.complete, full_pkl_tracking=False)
