"""Measured liftoff bridges for foot task references; source clock is unchanged."""
import math
import numpy as np

from fabrics_sim.reference.bridge import ReferenceBridge, RotationReferenceBridge


class ContactFootReference:
    def __init__(self, duration):
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError('Contact bridge duration must be finite and positive')
        self.duration = duration
        self.previous = None
        self.bridges = {}

    def update(self, time, contacts, measured_pose, measured_twist,
               source_pose, source_twist, source_acceleration):
        contacts = np.asarray(contacts, dtype=bool)
        if contacts.shape != (2,) or not math.isfinite(time):
            raise ValueError('Expected two contacts and finite time')
        pose, twist, acceleration = (np.array(x, dtype=float, copy=True) for x in
                                     (source_pose, source_twist, source_acceleration))
        if self.previous is None:
            self.previous = contacts.copy()
        for foot in range(2):
            if contacts[foot]:
                self.bridges.pop(foot, None)
            elif self.previous[foot]:
                self.bridges[foot] = (time,
                    ReferenceBridge(measured_pose[foot, :3] - pose[foot, :3],
                                    measured_twist[foot, :3] - twist[foot, :3],
                                    -acceleration[foot, :3], self.duration),
                    RotationReferenceBridge(measured_pose[foot, 3:], measured_twist[foot, 3:],
                                            pose[foot, 3:], twist[foot, 3:],
                                            acceleration[foot, 3:], self.duration))
            if foot in self.bridges:
                start, linear, angular = self.bridges[foot]
                elapsed = time - start
                if elapsed >= self.duration:
                    del self.bridges[foot]
                    continue
                offsets = linear.sample(elapsed)
                for target, offset in zip((pose[foot, :3], twist[foot, :3], acceleration[foot, :3]), offsets):
                    target += offset
                pose[foot, 3:], twist[foot, 3:], acceleration[foot, 3:] = angular.sample(
                    elapsed, pose[foot, 3:], twist[foot, 3:], acceleration[foot, 3:])
        self.previous = contacts.copy()
        return pose, twist, acceleration, np.array([foot in self.bridges for foot in range(2)])
