"""A fixed, C2 source clock prepared offline; no runtime optimization."""
import hashlib
from pathlib import Path

import numpy as np
from scipy.interpolate import BPoly


class PreparedSourceClock:
    def __init__(self, path, reference_path, source_sha256, fps, base_clock):
        with np.load(path, allow_pickle=False) as data:
            if data['parent_sha256'].item() != hashlib.sha256(Path(reference_path).read_bytes()).hexdigest():
                raise ValueError('Prepared clock reference mismatch')
            if data['source_sha256'].item() != source_sha256:
                raise ValueError('Prepared clock original source mismatch')
            self.time = np.array(data['time'], dtype=float)
            phase = np.array(data['frame'], dtype=float) / fps
            speed = np.array(data['speed'], dtype=float)
            acceleration = np.array(data['phase_acceleration'], dtype=float)
        arrays = (self.time, phase, speed, acceleration)
        if (not np.isfinite(fps) or fps <= 0 or len(self.time) < 3
                or any(a.shape != self.time.shape or not np.isfinite(a).all() for a in arrays)
                or np.any(np.diff(self.time) <= 0) or np.any(np.diff(phase) <= 0)
                or np.any(speed <= 0)):
            raise ValueError('Invalid prepared source clock')
        # Average adjoining constant-acceleration intervals at shared knots.
        # Quintic Hermite interpolation preserves phase and speed at every knot.
        acceleration[1:-1] = (acceleration[:-2] + acceleration[1:-1]) / 2
        acceleration[-1] = 0.
        prefix = np.asarray(base_clock(self.time[0]), dtype=float)
        prefix_residual = np.array([phase[0],speed[0],acceleration[0]])-prefix
        if not np.allclose(prefix,
                           [phase[0], speed[0], acceleration[0]], atol=1e-6, rtol=0):
            raise ValueError('Prepared clock does not preserve the existing prefix')
        # Remove only solver equality residuals so the actual runtime join is C2.
        phase[0],speed[0],acceleration[0] = prefix
        self.spline = BPoly.from_derivatives(self.time, np.column_stack((phase, speed, acceleration)))
        self.rate = self.spline.derivative()
        self.acceleration = self.spline.derivative(2)
        # Derivative roots give exact extrema within every polynomial interval.
        from scipy.interpolate import PPoly
        critical = PPoly.from_bernstein_basis(self.acceleration).roots(extrapolate=False)
        samples = np.r_[self.time, critical[np.isfinite(critical)]]
        rates = self.rate(samples)
        if np.min(rates) <= 0 or np.max(rates) > 2:
            raise ValueError('Prepared clock loses positive, bounded progression')
        self.final_phase, self.final_rate = phase[-1], speed[-1]
        self.report = dict(path=str(path), sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                           start_time_s=float(self.time[0]), end_time_s=float(self.time[-1]),
                           minimum_speed=float(np.min(rates)), maximum_speed=float(np.max(rates)),
                           prefix_equality_residual=prefix_residual.tolist(),
                           original_source_retained=True, runtime_optimizer=False,
                           full_tracking_passed=False)

    def sample(self, time, base_state):
        if time <= self.time[0]:
            return base_state
        if time >= self.time[-1]:
            return (float(self.final_phase + (time-self.time[-1])*self.final_rate),
                    float(self.final_rate), 0.)
        return tuple(float(f(time)) for f in (self.spline, self.rate, self.acceleration))
