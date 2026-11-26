import unittest
import numpy as np
import torch
from scipy.interpolate import BSpline
from reference_smoothing_basis import runtime_smoothing_basis
from blended_spline import BlendedSplineReference
from fabrics_sim.reference.interpolation import MotionInterpolator

class RuntimeBasisTests(unittest.TestCase):
    def test_pose_velocity_acceleration_match_runtime(self):
        knots=np.r_[np.full(4,780),np.arange(788,1100,8),np.full(4,1100)]
        raw=BSpline(knots,np.eye(len(knots)-4),3)
        coefficients=np.random.default_rng(8).normal(0,.02,len(knots)-4)
        coefficients[:3]=0;coefficients[-3:]=0
        q=torch.zeros((2000,36),dtype=torch.float64);q[:,6]=1
        q[780:1101,7]=torch.tensor(raw(np.arange(780,1101))@coefficients)
        runtime=BlendedSplineReference(MotionInterpolator(q),q,stride=6,start_frame=130,ramp_frames=20)
        basis=runtime_smoothing_basis(raw,2000)
        for frame in [775.5,780.,804.25,899.5,1008.75,1099.,1103.5]:
            for rate,acceleration in [(240.,0.),(173.,-80.)]:
                pose,velocity,acc=runtime.kinematics(frame,rate,acceleration)
                self.assertAlmostEqual(float(pose[0,7]),float(basis(frame)@coefficients),places=11)
                self.assertAlmostEqual(float(velocity[0,6]),float(basis(frame,1)@coefficients*rate),places=10)
                expected=(basis(frame,2)*rate**2+basis(frame,1)*acceleration)@coefficients
                self.assertAlmostEqual(float(acc[0,6]),float(expected),places=9)
    def test_rejects_nonlinear_endpoint_sampling(self):
        knots=np.r_[np.full(4,1950),[1962,1974,1986],np.full(4,1999)]
        with self.assertRaises(ValueError):runtime_smoothing_basis(BSpline(knots,np.eye(len(knots)-4),3),2000)

if __name__=='__main__':unittest.main()
