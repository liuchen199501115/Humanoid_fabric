"""Kinematic COM reference from the full robot model, without balance previews."""
import math
import torch
from .bridge import ReferenceBridge
from .interpolation import MotionInterpolator


class FullBodyComPath:
    """Join an initial COM state to source FK with a finite C2 correction.

    This is a reference construction, not a balance or feasibility certificate.
    Source endpoints are unmodified; forces are never predicted here.
    """
    @torch.no_grad()
    def __init__(self, fabric, initial, source, duration):
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError('Full-body COM path requires a positive finite duration')
        self.fabric, self.duration = fabric, float(duration)
        self.source = MotionInterpolator(source, method='pchip')
        self.span = len(source)-1
        initial_values = self._com(initial.q, initial.v, initial.a)
        first_values = self._source(0.)
        self.bridge = ReferenceBridge(*(x-y for x,y in zip(initial_values,first_values)), duration)

    def _com(self, q, v, a):
        model = self.fabric.dynamics_tasks.model
        state, _, _ = model.evaluate(q,v,self.fabric.integrate_root_position,
                                    self.fabric.dynamics_tasks.config['difference_step'])
        return tuple(x[0].detach().cpu().numpy().copy() for x in (
            state['com'], (state['com_jac']@v[...,None]).squeeze(-1),
            (state['com_jac']@a[...,None]).squeeze(-1)+state['com_jac_bias']))

    def _source(self, time):
        # Preserve the running source rates at touchdown. The C2 bridge
        # matches the initial state without forcing a stop at the endpoint.
        rate = self.span/self.duration
        return self._com(*self.source.kinematics(time*rate, rate, 0.))

    @torch.no_grad()
    def sample(self, time):
        if not math.isfinite(time) or not 0 <= time <= self.duration:
            raise ValueError('Full-body COM path sample is outside its interval')
        return tuple(x+y for x,y in zip(self._source(time),self.bridge.sample(time)))


class FullBodySourceFeet:
    """Track source swing FK with a C2 join; retain the actual stance anchor."""
    @torch.no_grad()
    def __init__(self, fabric, initial, source, duration, support, indices, clearance_bump=0.):
        from .bridge import RotationReferenceBridge
        from numpy.polynomial import Polynomial
        if not math.isfinite(clearance_bump) or clearance_bump < 0:
            raise ValueError('Swing clearance bump must be finite and nonnegative')
        bump = Polynomial([0.,0.,0.,64.,-192.,192.,-64.])*clearance_bump
        self.clearance_derivatives = tuple(bump.deriv(i) for i in range(3))
        self.fabric, self.duration, self.support = fabric, duration, support
        self.indices = indices
        self.source = MotionInterpolator(source, method='pchip')
        self.rate = (len(source)-1)/duration
        self.initial = self._feet(initial.q, initial.v, initial.a)
        # Match small measured stance rates without changing the WBC anchor.
        # This bounded reference-only transient returns to the same pose.
        self.stance_entry_duration = min(.2, duration)
        zero = self.initial[1][support,:3]*0.
        self.stance_translation = ReferenceBridge(zero, self.initial[1][support,:3],
            self.initial[2][support,:3], self.stance_entry_duration)
        self.stance_rotation = RotationReferenceBridge(self.initial[0][support,3:],
            self.initial[1][support,3:], self.initial[0][support,3:], zero, zero,
            self.stance_entry_duration, measured_acceleration=self.initial[2][support,3:])
        first = self._source(0.)
        swing = 1-support
        self.translation = ReferenceBridge(*(self.initial[i][swing,:3]-first[i][swing,:3] for i in range(3)),duration)
        self.rotation = RotationReferenceBridge(self.initial[0][swing,3:],self.initial[1][swing,3:],
            first[0][swing,3:],first[1][swing,3:],first[2][swing,3:],duration,
            measured_acceleration=self.initial[2][swing,3:])

    def _feet(self, q, v, a):
        model = self.fabric.dynamics_tasks.model
        state,_,_ = model.evaluate(q,v,self.fabric.integrate_root_position,
                                  self.fabric.dynamics_tasks.config['difference_step'])
        j = torch.cat((state['origin_jac'][:,self.indices],state['angular_jac'][:,self.indices]),2)
        bias = torch.cat((state['origin_jac_bias'][:,self.indices],state['angular_jac_bias'][:,self.indices]),2)
        return tuple(x[0].detach().cpu().numpy().copy() for x in (
            state['transforms'][:,self.indices],(j@v[:,None,:,None]).squeeze(-1),
            (j@a[:,None,:,None]).squeeze(-1)+bias))

    def _source(self,time):
        return self._feet(*self.source.kinematics(time*self.rate,self.rate,0.))

    @torch.no_grad()
    def sample(self,time):
        if not math.isfinite(time) or not 0 <= time <= self.duration:
            raise ValueError('Source foot sample is outside its interval')
        pose,v,a = self._source(time)
        swing = 1-self.support
        if time < self.duration:
            for target,offset in zip((pose,v,a),self.translation.sample(time)):
                target[swing,:3] += offset
            pose[swing,3:],v[swing,3:],a[swing,3:] = self.rotation.sample(
                time,pose[swing,3:],v[swing,3:],a[swing,3:])
        # C2 vertical lift: leave the measured entry and exact PKL landing
        # pose, velocity and acceleration unchanged.
        if 0. < time < self.duration:
            for order, target in enumerate((pose,v,a)):
                target[swing,2] += self.clearance_derivatives[order](time/self.duration)/self.duration**order
        pose[self.support] = self.initial[0][self.support]
        v[self.support] = a[self.support] = 0.
        if time < self.stance_entry_duration:
            from scipy.spatial.transform import Rotation
            import numpy as np
            offset, speed, acceleration = self.stance_translation.sample(time)
            if np.linalg.norm(offset) > .002:
                raise RuntimeError('Stance entry reference exceeds 2 mm; boundary needs further settling')
            pose[self.support,:3] += offset
            v[self.support,:3], a[self.support,:3] = speed, acceleration
            quat, omega, alpha = self.stance_rotation.sample(time,
                self.initial[0][self.support,3:], speed*0., speed*0.)
            angle = (Rotation.from_quat(quat)*Rotation.from_quat(self.initial[0][self.support,3:]).inv()).magnitude()
            if angle > .01:
                raise RuntimeError('Stance entry rotation exceeds 0.01 rad; boundary needs further settling')
            pose[self.support,3:] = quat
            v[self.support,3:], a[self.support,3:] = omega, alpha
        return pose,v,a


class OffsetComPath:
    """Explicit C2 COM adaptation; feasibility is checked by full dynamics.

    The offset changes no foot target. This is a kinematic parameterization,
    not an inverted-pendulum, capture-point or contact-force predictor.
    """
    def __init__(self, source_path, end_offset):
        import numpy as np
        self.source = source_path
        self.duration = source_path.duration
        self.offset = np.asarray(end_offset,dtype=float)
        if self.offset.shape != (3,) or not np.isfinite(self.offset).all():
            raise ValueError('COM offset must be a finite vector [3]')
        self.offset = self.offset.copy()

    def sample(self,time):
        from .contact_transfer import smooth_progress
        values = self.source.sample(time)
        return tuple(value + factor*self.offset for value,factor in
                     zip(values,smooth_progress(time,self.duration)))


class EndpointComPath:
    """C2 offset with independently adjustable terminal position/rates.

    Parameters are optimized against full-body constraints, not a reduced
    dynamics model. Initial position, velocity and acceleration are unchanged.
    """
    def __init__(self, source_path, position, velocity, acceleration):
        import numpy as np
        self.source, self.duration = source_path, source_path.duration
        states = np.asarray([position,velocity,acceleration],dtype=float)
        if states.shape != (3,3) or not np.isfinite(states).all():
            raise ValueError('Endpoint COM offsets require three finite vectors [3]')
        rhs = states*np.array([1.,self.duration,self.duration**2])[:,None]
        self.coefficients = np.linalg.solve(np.array([[1.,1.,1.],[3.,4.,5.],[6.,12.,20.]]),rhs)

    def sample(self,time):
        base = self.source.sample(time)
        u = time/self.duration
        c = self.coefficients
        offsets = (sum(c[k]*u**(k+3) for k in range(3)),
                   sum((k+3)*c[k]*u**(k+2) for k in range(3))/self.duration,
                   sum((k+3)*(k+2)*c[k]*u**(k+1) for k in range(3))/self.duration**2)
        return tuple(value+offset for value,offset in zip(base,offsets))


class InteriorComPath:
    """Two interior polynomial modes; both endpoint q/v/a offsets are zero.

    Coefficients change reference COM only, never foot targets or constraints.
    """
    def __init__(self, source_path, coefficients):
        import numpy as np
        from numpy.polynomial import Polynomial
        self.source,self.duration=source_path,source_path.duration
        values=np.asarray(coefficients,dtype=float)
        if values.shape!=(2,3) or not np.isfinite(values).all():
            raise ValueError('Interior COM coefficients must be finite [2,3]')
        self.coefficients=values.copy()
        bump=Polynomial([0.,0.,0.,64.,-192.,192.,-64.])
        modes=(bump,bump*Polynomial([-1.,2.]))
        self.derivatives=tuple(tuple(mode.deriv(order) for mode in modes) for order in range(3))

    def sample(self,time):
        import numpy as np
        base=self.source.sample(time)
        if time==0. or time==self.duration:return base
        u=time/self.duration
        return tuple(value+np.asarray([mode(u) for mode in modes])@self.coefficients/self.duration**order
                     for order,(value,modes) in enumerate(zip(base,self.derivatives)))


class WindowedComPath:
    """Compact C2 COM offset; leave the reference unchanged outside a window."""
    def __init__(self, source_path, offset, start, end):
        import numpy as np
        from numpy.polynomial import Polynomial
        self.source, self.duration = source_path, source_path.duration
        self.start, self.end = float(start), float(end)
        self.offset = np.asarray(offset, dtype=float).copy()
        if (self.offset.shape != (3,) or not np.isfinite(self.offset).all()
                or not np.isfinite([self.start, self.end]).all()
                or not 0 <= self.start < self.end <= self.duration):
            raise ValueError('Windowed COM requires finite [3] offset and a window inside the path')
        bump = Polynomial([0., 0., 0., 64., -192., 192., -64.])
        self.derivatives = tuple(bump.deriv(order) for order in range(3))

    def sample(self, time):
        base = self.source.sample(time)
        if time <= self.start or time >= self.end:
            return base
        width = self.end-self.start
        u = (time-self.start)/width
        return tuple(value+mode(u)*self.offset/width**order
                     for order, (value, mode) in enumerate(zip(base, self.derivatives)))
