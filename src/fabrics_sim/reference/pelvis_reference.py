"""Local pelvis-roll adaptation with world angular velocity/acceleration."""
import math
import torch
from numpy.polynomial import Polynomial
from .whole_body import WholeBodyReference


class WindowedPelvisRoll:
    """C2 roll offset on [0.3T, 0.9T]; foot targets are owned by the caller."""
    def __init__(self, duration, amplitude):
        if not math.isfinite(duration) or duration <= 0 or not math.isfinite(amplitude):
            raise ValueError('Finite roll amplitude and positive duration required')
        self.start, self.end = .3*duration, .9*duration
        self.amplitude = amplitude
        bump = Polynomial([0.,0.,0.,64.,-192.,192.,-64.])
        self.derivatives = tuple(bump.deriv(i) for i in range(3))

    def sample(self, reference, time):
        if not math.isfinite(time):
            raise ValueError('Finite sample time required')
        if not self.amplitude or time <= self.start or time >= self.end:
            return reference
        width = self.end-self.start
        u = (time-self.start)/width
        angle, rate, acceleration = [self.amplitude*p(u)/width**i
                                     for i,p in enumerate(self.derivatives)]
        q, v, a = (getattr(reference,key).clone() for key in ('q','v','a'))
        quat = q[:,3:7].clone()
        delta = torch.zeros_like(quat)
        delta[:,0], delta[:,3] = math.sin(angle/2), math.cos(angle/2)
        q[:,3:6] = (quat[:,3:]*delta[:,:3]+delta[:,3:]*quat[:,:3]
                    +torch.linalg.cross(quat[:,:3],delta[:,:3]))
        q[:,6] = quat[:,3]*delta[:,3]-(quat[:,:3]*delta[:,:3]).sum(-1)
        # The source local X axis expressed in world coordinates.
        x,y,z,w = quat.unbind(-1)
        axis = torch.stack((1-2*(y*y+z*z),2*(x*y+w*z),2*(x*z-w*y)),-1)
        omega = v[:,3:6].clone()
        v[:,3:6] += axis*rate
        a[:,3:6] += axis*acceleration+torch.linalg.cross(omega,axis)*rate
        return WholeBodyReference(q,v,a)
