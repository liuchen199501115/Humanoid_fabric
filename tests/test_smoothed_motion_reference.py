import unittest
import torch
from fabrics_sim.reference.interpolation import MotionInterpolator

class SmoothedMotionTests(unittest.TestCase):
    def motion(self,n=9):
        q=torch.zeros(n,36,dtype=torch.float64);q[:,6]=1
        q[:,7]=torch.tensor(([0,.2,.8,.1,-.1,.7,.4,.2,0])[:n])
        return q
    def test_bounds_and_endpoints(self):
        q=self.motion();r=MotionInterpolator(q,method='bspline')
        for i in torch.linspace(0,8,201):
            x=r(float(i));self.assertLessEqual(float(x[0,7]),float(q[:,7].max())+1e-10)
            self.assertGreaterEqual(float(x[0,7]),float(q[:,7].min())-1e-10)
        torch.testing.assert_close(r(0),q[:1]);torch.testing.assert_close(r(8),q[-1:])
        self.assertGreater(float(abs(r(2)-q[2:3]).max()),.01)
    def test_c2_and_derivatives(self):
        r=MotionInterpolator(self.motion(),method='bspline')
        for f in range(1,8):
            before=r.kinematics(f-1e-7,1.);after=r.kinematics(f+1e-7,1.)
            for a,b in zip(before,after):torch.testing.assert_close(a,b,atol=1e-5,rtol=1e-5)
        f=3.4;e=1e-4;q,v,a=r.kinematics(f,1.)
        torch.testing.assert_close(v[:,6:],(r(f+e)[:,7:]-r(f-e)[:,7:])/(2*e),atol=1e-7,rtol=1e-6)
        torch.testing.assert_close(a[:,6:],(r(f+e)[:,7:]-2*q[:,7:]+r(f-e)[:,7:])/e**2,atol=1e-7,rtol=1e-6)
    def test_short_clips_and_quaternion_signs(self):
        for n in range(1,5):
            q=self.motion(n);q[1::2,6]=-1;r=MotionInterpolator(q,method='bspline')
            for f in (0.,(n-1)/2,n-1):
                pose,v,a=r.kinematics(f,1.)
                self.assertTrue(torch.isfinite(a).all());self.assertAlmostEqual(float(pose[0,6].abs()),1.)
    def test_retiming_chain_rule(self):
        r=MotionInterpolator(self.motion(),method='bspline')
        _,v,a=r.kinematics(3.2,1);_,slow_v,slow_a=r.kinematics(3.2,.25)
        torch.testing.assert_close(slow_v,.25*v);torch.testing.assert_close(slow_a,.0625*a)

    def test_scalar_peak_search_includes_interior_extremum(self):
        import numpy as np
        from prepare_mocap_reference import scalar_derivative_peaks
        c=np.array([1.,-1.5,0.,0.]).reshape(4,1,1)
        v,a=scalar_derivative_peaks(c)
        self.assertAlmostEqual(float(v[0]),.75)
        self.assertAlmostEqual(float(a[0]),3.)
