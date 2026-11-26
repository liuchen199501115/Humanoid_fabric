"""Measured stance heading adapted in a whole-body reference, with C2 fades.

The measured heading is frozen once per loaded source stance and faded out
before planned takeoff so the next swing can turn. Only reference
leg coordinates change; the physical contact anchors and raw audit do not.
"""
import numpy as np
from measured_stance_reference import MeasuredStanceReference, smooth
from bounded_leg_ik import solve_leg_pose


class StanceHeadingReference(MeasuredStanceReference):
    def __init__(self, urdf, joint_names, modes, fps, *, ramp=.15, limit=.4):
        super().__init__(urdf, joint_names, modes, fps, ramp=ramp, limit=limit)
        self.headings = {}
        self.peak_heading_correction = 0.

    def observe(self, frame, q, forces, contact_data):
        previous = dict(self.last_interval)
        super().observe(frame, q, forces, contact_data)
        for side in self.anchors:
            if previous.get(side) != self.last_interval[side]:
                _, rotation, _ = self.model.evaluate(q, side)
                self.headings[side] = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
                next(event for event in reversed(self.events) if event['side'] == side)['heading'] = self.headings[side]

    def target(self, frame, side, position, rotation):
        anchor = self.anchors[side]
        weight = smooth((frame-anchor.begin)/self.ramp)*smooth((anchor.end-frame)/self.ramp)
        source_heading = np.arctan2(rotation[1, 0], rotation[0, 0])
        error = self.headings[side]-source_heading
        error = np.arctan2(np.sin(error), np.cos(error))
        delta = weight*self.limit*np.tanh(error/self.limit)
        self.peak_heading_correction = max(self.peak_heading_correction, abs(float(delta)))
        c, s = np.cos(delta), np.sin(delta)
        target_rotation = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])@rotation
        # Preserve the audited foot-link path. Rotating about a sole point
        # can extend a nearly straight reference leg beyond its reachable set.
        target_position = position.copy()
        return target_position, target_rotation

    def correction(self, frame, q):
        out = np.array(q, dtype=float, copy=True)
        for side, anchor in self.anchors.items():
            if not anchor.begin < frame < anchor.end:
                continue
            position, rotation, _ = self.model.evaluate(q, side)
            target_position, target_rotation = self.target(frame, side, position, rotation)
            out, pe, re = solve_leg_pose(
                self.model, out, side, target_position, target_rotation,
                max_iterations=80, position_tolerance=1e-10, orientation_tolerance=1e-9)
            self.peak_position_residual = max(self.peak_position_residual, pe)
            self.peak_orientation_residual = max(self.peak_orientation_residual, re)
            if pe > .002 or re > .01:
                raise RuntimeError(f'Stance heading IK failed at frame {frame:.3f}, foot {side}: {pe:.6f} m, {re:.6f} rad')
        result = out[7:]-np.asarray(q)[7:]
        self.peak_joint_correction = max(self.peak_joint_correction, float(abs(result).max()))
        return result
