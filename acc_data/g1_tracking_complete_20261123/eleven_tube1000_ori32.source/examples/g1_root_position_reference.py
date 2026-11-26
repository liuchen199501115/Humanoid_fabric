"""Keep original-source root targets separate from prepared control targets."""
from fabrics_sim.reference.interpolation import MotionInterpolator


class RootPositionReference:
    def __init__(self, source_trajectory):
        self.source = MotionInterpolator(source_trajectory, method='pchip')

    def sample(self, frame, rate, acceleration, *, mode, control=None):
        if mode == 'source':
            return self.source.kinematics(frame, rate, acceleration)
        if mode == 'control' and control is not None:
            return control.q, control.v, control.a
        raise ValueError('Root reference requires source or an explicit control q/v/a')
