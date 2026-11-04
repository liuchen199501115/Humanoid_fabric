"""Motion interpolation, shape-preserving C1 PCHIP by default.

PCHIP bounds each scalar joint coordinate by the adjacent sample values.
Quaternion signs are unwrapped before interpolation and renormalized afterward.
Default PCHIP removes linear interpolation's velocity jumps; its acceleration is
piecewise continuous, not C2. Optional cubic interpolation is C2 but may overshoot
sampled positions; callers must audit the resulting joint/dynamics bounds.
Optional bspline uses source samples as control points: it is C2, preserves
global scalar joint bounds, but changes intermediate poses and requires a
separate source-fidelity audit. Quaternion normalization is still required.
"""
import math

import numpy as np
import torch


class MotionInterpolator:
    def __init__(self, trajectory, *, pose_dimension=36, method="pchip"):
        from scipy.interpolate import PchipInterpolator, CubicSpline, BSpline
        if method not in ('pchip', 'cubic', 'bspline'):
            raise ValueError('Interpolation method must be pchip, cubic or bspline')
        self.method = method
        if (pose_dimension not in (7,36) or trajectory.ndim != 2 or trajectory.shape[1] != pose_dimension or not len(trajectory)
                or not trajectory.is_floating_point() or not torch.isfinite(trajectory).all()
                or (trajectory[:,3:7].norm(dim=-1) < 1e-8).any()):
            raise ValueError(f'Expected finite poses [T,{pose_dimension}] with nonzero quaternions')
        self.trajectory = trajectory.detach().clone()
        samples = trajectory.detach().cpu().double().numpy().copy()
        samples[:,3:7] /= np.linalg.norm(samples[:,3:7],axis=1,keepdims=True)
        for i in range(1,len(samples)):
            if samples[i-1,3:7] @ samples[i,3:7] < 0:
                samples[i,3:7] *= -1
        self.coefficients = None
        if len(samples) > 1:
            if method == "bspline":
                # Source poses are control points, not interpolation constraints.
                # A clamped cubic B-spline is C2 and preserves joint convex bounds.
                count = len(samples)
                degree = min(3, count-1)
                interior = (np.arange(2, count-2) if degree == 3 else np.array([]))
                knots = np.r_[np.zeros(degree+1), interior,
                              np.full(degree+1, count-1)]
                spline = BSpline(knots, samples, degree, axis=0)
                frames = np.arange(count-1)
                coefficients = np.stack([
                    spline.derivative(order)(frames)/math.factorial(order)
                    if order <= degree else np.zeros((count-1, samples.shape[1]))
                    for order in (3, 2, 1, 0)])
            else:
                interpolate = PchipInterpolator if method == "pchip" else CubicSpline
                coefficients = interpolate(np.arange(len(samples)), samples, axis=0).c
            self.coefficients = torch.as_tensor(coefficients,device=trajectory.device,dtype=trajectory.dtype)

    def __call__(self, frame):
        if not math.isfinite(frame):
            raise ValueError('Motion frame must be finite')
        frame = min(max(float(frame),0.),len(self.trajectory)-1)
        index = int(frame)
        if self.coefficients is None or (frame == index and self.method != "bspline"):
            return self.trajectory[index:index+1].clone()
        index = min(index, len(self.trajectory)-2)
        t = frame-index
        c = self.coefficients[:,index]
        value = ((c[0]*t+c[1])*t+c[2])*t+c[3]
        value[3:7] /= value[3:7].norm().clamp_min(1e-8)
        return value[None]

    def kinematics(self, frame, frame_rate, frame_acceleration=0.):
        """Pose and analytic world-frame generalized velocity/acceleration.

        Rates are in source frames/s and frames/s², including retiming by the
        reference manager. Differentiate the polynomial and normalized
        quaternion directly; no subtraction of nearly equal float32 poses.
        PCHIP remains only C1 at source knots (right-hand acceleration here).
        """
        if not all(math.isfinite(x) for x in (frame, frame_rate, frame_acceleration)):
            raise ValueError('Motion time derivatives must be finite')
        pose = self(frame)
        zero = pose.new_zeros(1, pose.shape[1]-1)
        if self.coefficients is None or not 0 <= frame <= len(self.trajectory)-1:
            return pose, zero, zero.clone()
        index = min(int(frame), len(self.trajectory)-2)
        t = frame-index
        c = self.coefficients[:, index]
        raw = ((c[0]*t+c[1])*t+c[2])*t+c[3]
        first = (3*c[0]*t+2*c[1])*t+c[2]
        second = 6*c[0]*t+2*c[1]
        velocity = first*frame_rate
        acceleration = second*frame_rate**2+first*frame_acceleration
        u, du, ddu = raw[3:7], velocity[3:7], acceleration[3:7]
        norm = u.norm().clamp_min(1e-8)
        quat = u/norm
        norm_v = (quat*du).sum()
        quat_v = (du-quat*norm_v)/norm
        norm_a = (du.square().sum()+(u*ddu).sum()-norm_v.square())/norm
        quat_a = (ddu-quat*norm_a-2*quat_v*norm_v)/norm
        def angular(derivative):
            return 2*(quat[3]*derivative[:3]-derivative[3]*quat[:3]
                      +torch.linalg.cross(quat[:3],derivative[:3]))
        return (pose,
                torch.cat((velocity[:3],angular(quat_v),velocity[7:]))[None],
                torch.cat((acceleration[:3],angular(quat_a),acceleration[7:]))[None])
