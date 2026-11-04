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


def select_dynamic_steps(modes, count):
    """Select source landings, including an initial partial swing.

    Direct support switches are retained; no fictitious double-support phase
    is inserted and the moving reference is not truncated at a landing.
    """
    if type(count) is not int or count < 1:
        raise ValueError('Step count must be a positive integer')
    modes = np.asarray(modes)
    events, boundary = [], 0
    for _ in range(count):
        event = next_source_event(modes, boundary)
        if event['kind'] != 'swing':
            raise ValueError('Not enough source landing events')
        if events and event['support'] == events[-1]['support']:
            raise ValueError('Requested source steps do not alternate')
        event['direct_support_switch'] = event.pop('inserted_double_support')
        event['starts_mid_swing'] = bool(not events and modes[0] != 3)
        # A recovery after this foot begins its next swing cannot count as the
        # intended landing. The incoming foot swings in the old support mode.
        later = np.flatnonzero(modes[event['end_frame']:] == event['support']+1)
        event['landing_window_end_frame'] = (int(event['end_frame'] + later[0])
                                             if len(later) else len(modes))
        events.append(event)
        boundary = event['end_frame']
    return events


def select_all_dynamic_steps(modes):
    """Select every observed source landing, without inventing a final landing.

    A final single-support segment is an unfinished swing and stays outside
    the landing plan. Its original samples still belong to full-clip tracking.
    """
    modes = np.asarray(modes)
    if modes.ndim != 1 or not len(modes) or not np.isin(modes, [1, 2, 3]).all():
        raise ValueError('Source events require non-flight contact modes')
    ends = np.flatnonzero(np.diff(modes)) + 1
    count = int(np.isin(modes[ends-1], [1, 2]).sum())
    return select_dynamic_steps(modes, count) if count else []


class MeasuredSteps:
    def __init__(self, events, goals, tolerance=.04, continuous=False, track_source_pose=False):
        self.events = events
        self.goals = np.asarray(goals, dtype=float)
        if self.goals.shape != (len(events), 3) or not np.isfinite(self.goals).all():
            raise ValueError('Expected one finite world-frame landing goal per step')
        self.tolerance = tolerance
        self.continuous = continuous
        if track_source_pose and not continuous:
            raise ValueError('Source-pose landing goals require continuous tracking')
        self.track_source_pose = bool(track_source_pose)
        self.airborne = False
        self.loaded_seconds = 0.
        self.landings = []

    @property
    def complete(self):
        return len(self.landings) == len(self.events)

    def update(self, frame, time, mode, forces, clearances, feet, dt, reference_feet=None):
        forces, clearances, feet = map(np.asarray, (forces, clearances, feet))
        if (forces.shape != (2,) or clearances.shape != (2,) or feet.shape != (2, 3)
                or not all(np.isfinite(x).all() for x in (forces, clearances, feet))
                or not np.isfinite([frame,time,dt]).all() or dt <= 0):
            raise ValueError('Invalid measured step observation')
        if self.track_source_pose:
            reference_feet = np.asarray(reference_feet, dtype=float)
            if reference_feet.shape != (2, 3) or not np.isfinite(reference_feet).all():
                raise ValueError('Source-pose tracking requires finite same-frame original reference feet [2,3]')
        if self.complete:
            return True
        event = self.events[len(self.landings)]
        swing, support = 1-event['support'], event['support']
        if frame < event['takeoff_frame']-1:
            return False
        if self.continuous and frame >= event['landing_window_end_frame']:
            self.loaded_seconds = 0.
            return False
        if (mode == support+1 and forces[support]>=100. and forces[swing]<10.
                and clearances[swing]>=.02):
            self.airborne = True
        # A foot LINK translates during heel-to-flat rolling even when its
        # material contact does not slip. Continuous PKL tracking follows the
        # original same-frame link pose, not its contact-onset coordinates.
        goal = reference_feet[swing] if self.track_source_pose else self.goals[len(self.landings)]
        error = float(np.linalg.norm(feet[swing]-goal))
        bearing = ((mode in (3, swing+1) and forces[swing]>=20.) if self.continuous
                   else (mode==3 and (forces>=20.).all()))
        ready = (self.airborne and frame>=event['end_frame']-1. and bearing
                 and error<=self.tolerance)
        self.loaded_seconds = self.loaded_seconds+dt if ready else 0.
        if self.loaded_seconds >= .05-1e-9:
            self.landings.append(dict(step=len(self.landings)+1, swing='left' if swing==0 else 'right',
                source_frame=float(frame), time=float(time), landing_error_m=error,
                goal=goal.tolist(), measured_load_n=float(forces[swing]),
                source_contact_onset_goal=self.goals[len(self.landings)].tolist()))
            self.airborne = False
            self.loaded_seconds = 0.
        return self.complete

    def report(self):
        return dict(scope=('continuous source steps, same-frame original foot-link targets' if self.track_source_pose
                           else 'bounded source steps, unchanged landing targets'),
            landing_goal_policy='source_pose' if self.track_source_pose else 'fixed_contact_onset',
            expected_steps=len(self.events),
            measured_steps=len(self.landings), landings=self.landings,
            continuous=self.continuous,
            initial_partial_step=bool(self.events and self.events[0].get('starts_mid_swing',False)),
            measured_steps_complete=self.complete, full_pkl_tracking=False)
