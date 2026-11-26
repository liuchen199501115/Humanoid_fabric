import unittest,numpy as np,torch
from blended_spline import BlendedSplineReference
from late_tube_smoothing import transition_tube_spline
from fabrics_sim.reference.interpolation import MotionInterpolator
class LateTubeSmoothingTests(unittest.TestCase):
    def setUp(self):
        t=np.arange(180.)/200.;q=np.zeros((180,36));q[:,6]=1.;q[:,0]=t+.003*np.sin(60*t);q[:,7]=.2*np.sin(30*t)
        angle=.03*np.sin(40*t);q[:,3]=np.sin(angle/2);q[:,6]=np.cos(angle/2)
        self.q=torch.tensor(q,dtype=torch.float64)
        self.old=BlendedSplineReference(MotionInterpolator(self.q),self.q,stride=2,start_frame=10,ramp_frames=10)
        self.new=transition_tube_spline(self.old,self.q,stride=6,start_frame=80,ramp_frames=20)
    def test_prior_motion_exactly_preserved(self):
        for f in [0.,20.,50.,79.9,80.]:
            for a,b in zip(self.old.kinematics(f,240.),self.new.kinematics(f,240.)):
                torch.testing.assert_close(a,b,rtol=0,atol=0)
    def test_qva_are_derivatives_of_one_curve(self):
        for f in [81.17,90.37,99.23,110.41]:
            h=1e-6;q,v,a=self.new.kinematics(f,240.)
            qm,vm,_=self.new.kinematics(f-h*240.,240.);qp,vp,_=self.new.kinematics(f+h*240.,240.)
            np.testing.assert_allclose((qp[0,[0,7]]-qm[0,[0,7]])/(2*h),v[0,[0,6]],atol=1e-6)
            np.testing.assert_allclose((vp-vm)/(2*h),a,atol=1e-4)
    def test_acceleration_continuity_at_transition_boundaries(self):
        for f in [80.,100.]:
            for left,right in zip(self.new.kinematics(f-1e-6,240.),self.new.kinematics(f+1e-6,240.)):
                torch.testing.assert_close(left,right,atol=2e-5,rtol=1e-5)
if __name__=='__main__':unittest.main()
