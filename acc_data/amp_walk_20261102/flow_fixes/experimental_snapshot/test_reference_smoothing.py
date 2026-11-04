import unittest
import torch
from fabrics_sim.reference.smoothing import smooth_motion_reference
from fabrics_sim.reference.interpolation import MotionInterpolator

class ReferenceSmoothingTests(unittest.TestCase):
    def trajectory(self):
        q=torch.zeros(101,36,dtype=torch.float64);q[:,6]=1
        t=torch.arange(101,dtype=q.dtype)/200
        q[:,0]=t;q[:,7]=.5*t
        return q

    def test_linear_motion_keeps_boundary_velocity_and_source(self):
        q=self.trajectory();original=q.clone();smooth=smooth_motion_reference(q,200,.02)
        torch.testing.assert_close(q,original)
        torch.testing.assert_close(smooth,q,atol=1e-12,rtol=1e-12)
        _,v,a=MotionInterpolator(smooth).kinematics(0,200)
        self.assertAlmostEqual(float(v[0,0]),1.,places=8)
        self.assertAlmostEqual(float(a[0,0]),0.,places=7)

    def test_quaternion_sign_and_joint_bounds(self):
        q=self.trajectory();q[1::2,6]=-1
        q[:,7]=torch.sin(torch.arange(101,dtype=q.dtype)*2)*.1
        smooth=smooth_motion_reference(q,200,.02)
        torch.testing.assert_close(smooth[:,3:7].norm(dim=-1),torch.ones(101,dtype=q.dtype))
        self.assertGreaterEqual(float(smooth[:,7].min()),float(q[:,7].min()))
        self.assertLessEqual(float(smooth[:,7].max()),float(q[:,7].max()))
        self.assertLess(float(smooth[:,7].std()),float(q[:,7].std()))

if __name__=='__main__':unittest.main()
