"""Retain audited failure times when refining an offline reference search."""
import numpy as np


def refined_sample_times(duration, samples, report=None):
    """Uniform grid plus prior failure times and their dense-grid neighbours.

    No time rescaling is implicit: reports from another duration are rejected.
    This is a search grid, never a replacement for independent dense validation.
    """
    if (not np.isfinite(duration) or duration <= 0
            or type(samples) is not int or samples < 3):
        raise ValueError('Positive finite duration and at least three samples required')
    times = list(np.linspace(0., duration, samples))
    if report is not None:
        previous_duration = float(report['options']['duration'])
        if not np.isfinite(previous_duration) or abs(previous_duration-duration) > 1e-9:
            raise ValueError('Refinement report duration must match the candidate')
        best = report['best']
        centers = list(best.get('worst_constraint_times', {}).values())
        centers += [best.get('worst_sample', {}).get('time'),
                    best.get('collision', {}).get('worst_time')]
        centers += best.get('collision', {}).get('violation_times', [])
        previous = list(report.get('optimization_sample_times', []))
        centers = [float(t) for t in centers if t is not None]
        if any(not np.isfinite(t) or not 0 <= t <= duration for t in centers+previous):
            raise ValueError('Report times must be finite and within the candidate')
        times.extend(previous)
        # At least as fine as the established 601-point validation grid.
        step = duration/max(600, int(best.get('sample_count', 601))-1)
        for center in centers:
            times.extend(np.clip(center+step*np.arange(-2, 3), 0., duration))
    # Merge floating-point duplicates to avoid nearly coincident IK samples.
    return np.unique(np.round(times, 12))
