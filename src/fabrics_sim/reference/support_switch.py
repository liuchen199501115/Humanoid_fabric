"""Measured-load gate for switching support after a stopped landing reference."""
import math
import numpy as np
from .contact_transfer import ContactTransferReference


class SupportSwitchReference:
    def __init__(self, fabric, initial, names, stance, support, start_time, dt, joint_bounds,
                 *, com_height_offset=.01, lift_height=.04, pelvis_roll_offset=0.):
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError('Support switch requires positive finite control period')
        self.transfer=ContactTransferReference(fabric,initial,names,stance,support,start_time,
            preflight_dt=dt,joint_bounds=joint_bounds,com_height_offset=com_height_offset,lift_height=lift_height,pelvis_roll_offset=pelvis_roll_offset)
        self.support,self.mode,self.dt=support,3,float(dt)
        self.ready_samples=0
        self.confirmation_samples=max(1,math.ceil(.05/dt))

    def update(self,time,forces,contacts,com,velocity,q):
        forces=np.asarray(forces,dtype=float);contacts=np.asarray(contacts,dtype=bool)
        if (forces.shape!=(2,) or contacts.shape!=(2,) or not np.isfinite(forces).all()
                or not math.isfinite(time)):
            raise ValueError('Support switch requires finite measured forces/contact flags')
        if not contacts.any():
            raise RuntimeError('Support switch lost both measured supports')
        if self.transfer.release_time is not None:
            if not contacts[self.support] or forces[self.support]<20.:
                raise RuntimeError('Support switch lost the new support foot')
            return self.mode
        ready=(contacts[self.support] and forces[self.support]>=100.
               and forces[1-self.support]<=.1*max(float(forces.sum()),1.)
               and self.transfer.ready_for_release(time,com,velocity,q))
        self.ready_samples=self.ready_samples+1 if ready else 0
        if self.ready_samples>=self.confirmation_samples:
            self.mode=self.support+1
        elif time>self.transfer.start_time+self.transfer.duration+5.:
            raise RuntimeError('Support switch timed out waiting for measured load and balance')
        return self.mode

    def sample(self,time):
        return self.transfer.sample(time,self.mode)
