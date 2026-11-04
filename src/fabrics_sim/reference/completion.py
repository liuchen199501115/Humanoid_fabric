"""End-of-clip confirmation, independent of the requested simulation timeout."""
import math
import numpy as np


class MotionCompletion:
    def __init__(self, final_frame, settle_seconds=.5):
        if not math.isfinite(final_frame) or final_frame < 0 or not math.isfinite(settle_seconds) or settle_seconds <= 0:
            raise ValueError('Invalid completion configuration')
        self.final_frame=float(final_frame)
        self.settle_seconds=float(settle_seconds)
        self.stable_seconds=0.
        self.complete=False

    def update(self, frame, mode, settling, forces, com_velocity, dt):
        forces=np.asarray(forces,dtype=float)
        velocity=np.asarray(com_velocity,dtype=float)
        if (forces.shape!=(2,) or velocity.shape!=(3,) or not np.isfinite(forces).all()
                or not np.isfinite(velocity).all() or not math.isfinite(frame)
                or not math.isfinite(dt) or dt<=0):
            raise ValueError('Invalid completion observation')
        ready=(frame>=self.final_frame-1e-6 and mode==3 and not settling
               and (forces>=20.).all() and np.linalg.norm(velocity)<=.06)
        self.stable_seconds=self.stable_seconds+dt if ready else 0.
        self.complete=self.stable_seconds+1e-9>=self.settle_seconds
        return self.complete
