"""Finite C2 offset that joins a running reference without moving the robot."""
import numpy as np


class ReferenceBridge:
    """Cancel an initial position/velocity/acceleration mismatch, then fade out.

    The caller adds the returned offset and its derivatives to the source
    reference. At duration all three are zero, so the original endpoint is
    retained. This ensures continuity, not dynamic feasibility.
    """

    def __init__(self, position_error, velocity_error, acceleration_error, duration):
        p, v, a = (np.asarray(x, dtype=float) for x in
                   (position_error, velocity_error, acceleration_error))
        if (p.ndim != 1 or not p.size or v.shape != p.shape or a.shape != p.shape
                or not all(np.isfinite(x).all() for x in (p, v, a))
                or not np.isfinite(duration) or duration <= 0):
            raise ValueError("Bridge requires finite matching vectors and positive duration")
        self.duration = float(duration)
        # Normalized time keeps the solve independent of the physical duration.
        c0, c1, c2 = p.copy(), v*duration, .5*a*duration**2
        rhs = np.stack((-c0-c1-c2, -c1-2*c2, -2*c2))
        high = np.linalg.solve(np.array([[1., 1., 1.], [3., 4., 5.], [6., 12., 20.]]), rhs)
        self.coefficients = np.concatenate((np.stack((c0, c1, c2)), high))

    def sample(self, elapsed):
        if not np.isfinite(elapsed) or elapsed < 0:
            raise ValueError("Bridge elapsed time must be finite and nonnegative")
        if elapsed >= self.duration:
            return tuple(np.zeros_like(self.coefficients[0]) for _ in range(3))
        s = elapsed/self.duration
        c = self.coefficients
        return (sum(c[k]*s**k for k in range(6)),
                sum(k*c[k]*s**(k-1) for k in range(1, 6))/self.duration,
                sum(k*(k-1)*c[k]*s**(k-2) for k in range(2, 6))/self.duration**2)


class RotationReferenceBridge:
    """Fade a measured-to-source world rotation offset with consistent rates."""

    def __init__(self, measured_quaternion, measured_velocity, source_quaternion,
                 source_velocity, source_acceleration, duration, measured_acceleration=None):
        from scipy.spatial.transform import Rotation
        delta=Rotation.from_quat(measured_quaternion)*Rotation.from_quat(source_quaternion).inv()
        q=delta.as_quat()
        if q[3]<0: q=-q
        transported=delta.apply(source_velocity)
        omega=np.asarray(measured_velocity)-transported
        alpha=(np.zeros(3) if measured_acceleration is None else np.asarray(measured_acceleration))-delta.apply(source_acceleration)-np.cross(omega,transported)
        def multiply(a,b):
            return np.r_[a[3]*b[:3]+b[3]*a[:3]+np.cross(a[:3],b[:3]),
                         a[3]*b[3]-a[:3]@b[:3]]
        dq=.5*multiply(np.r_[omega,0.],q)
        ddq=.5*(multiply(np.r_[alpha,0.],q)+multiply(np.r_[omega,0.],dq))
        self.offset=ReferenceBridge(q-np.array([0.,0.,0.,1.]),dq,ddq,duration)

    def sample(self, elapsed, source_quaternion, source_velocity, source_acceleration):
        from scipy.spatial.transform import Rotation
        p,dp,ddp=self.offset.sample(elapsed)
        raw=p+np.array([0.,0.,0.,1.]);norm=np.linalg.norm(raw)
        if norm<1e-6: raise RuntimeError('Rotation bridge crossed a zero quaternion')
        q=raw/norm;nv=q@dp;dq=(dp-q*nv)/norm
        na=(dp@dp+raw@ddp-nv*nv)/norm
        ddq=(ddp-q*na-2*dq*nv)/norm
        def angular(d):return 2*(q[3]*d[:3]-d[3]*q[:3]+np.cross(q[:3],d[:3]))
        w,a=angular(dq),angular(ddq)
        delta=Rotation.from_quat(q)
        transported=delta.apply(source_velocity)
        return ((delta*Rotation.from_quat(source_quaternion)).as_quat(),
                w+transported,a+delta.apply(source_acceleration)+np.cross(w,transported))


class WholeBodyReferenceBridge:
    """C2 join of a running q/v/a reference, including world angular rates.

    This modifies a reference only; it neither resets a robot nor establishes
    contact or dynamics feasibility. At the endpoint the source is unchanged.
    """
    def __init__(self, initial, source, duration):
        if initial.q.shape != (1,36) or source.q.shape != (1,36):
            raise ValueError('Whole-body bridge requires single G1 reference states')
        self.q_columns = [0,1,2]+list(range(7,36))
        self.v_columns = [0,1,2]+list(range(6,35))
        array=lambda t:t[0].detach().cpu().numpy()
        self.offset = ReferenceBridge(array(initial.q[:,self.q_columns]-source.q[:,self.q_columns]),
            array(initial.v[:,self.v_columns]-source.v[:,self.v_columns]),
            array(initial.a[:,self.v_columns]-source.a[:,self.v_columns]),duration)
        self.rotation = RotationReferenceBridge(array(initial.q[:,3:7]),array(initial.v[:,3:6]),
            array(source.q[:,3:7]),array(source.v[:,3:6]),array(source.a[:,3:6]),duration,
            measured_acceleration=array(initial.a[:,3:6]))

    def sample(self, elapsed, source):
        from .whole_body import WholeBodyReference
        # Validate time even on the unchanged endpoint path.
        offsets=self.offset.sample(elapsed)
        if elapsed >= self.offset.duration:
            return source
        q,v,a=source.q.clone(),source.v.clone(),source.a.clone()
        for target,columns,offset in zip((q,v,a),(self.q_columns,self.v_columns,self.v_columns),offsets):
            target[:,columns] += target.new_tensor(offset)[None]
        array=lambda t:t[0].detach().cpu().numpy()
        quat,omega,alpha=self.rotation.sample(elapsed,array(source.q[:,3:7]),
                                            array(source.v[:,3:6]),array(source.a[:,3:6]))
        q[:,3:7]=q.new_tensor(quat)[None]
        v[:,3:6]=v.new_tensor(omega)[None]
        a[:,3:6]=a.new_tensor(alpha)[None]
        return WholeBodyReference(q,v,a)
