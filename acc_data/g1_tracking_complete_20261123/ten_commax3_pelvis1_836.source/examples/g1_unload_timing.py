"""Fit a measured load-transfer ramp inside the next source swing boundary."""
import math


def unload_ramp_for_source(events, frame, frame_rate, outgoing, nominal, dt):
    """Shorten a force-objective ramp, without releasing contact constraints.

    Call once when a measured transfer starts. Completed swings are excluded;
    an overdue ongoing swing gets one control period. A paused clock retains
    the ordinary ramp, since its next source boundary has no wall-time deadline.
    """
    if (outgoing not in (0, 1) or not all(math.isfinite(x) for x in
            (frame, frame_rate, nominal, dt)) or frame_rate < 0
            or not 0 < dt <= nominal):
        raise ValueError('Invalid source-clock load-transfer timing')
    if frame_rate == 0:
        return nominal
    pending = [event for event in events
               if 1-event['support'] == outgoing and event['end_frame'] > frame]
    if not pending:
        return nominal
    event = min(pending, key=lambda item: item['takeoff_frame'])
    return max(dt, min(nominal, (event['takeoff_frame']-frame)/frame_rate))
