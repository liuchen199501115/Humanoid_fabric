import unittest
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from fabrics_sim.reference.pelvis_reference import WindowedPelvisRoll
from fabrics_sim.reference.whole_body import WholeBodyReference


class PelvisReferenceTests(unittest.TestCase):
    def source(self,t):
        q=torch.zeros(1,36,dtype=torch.float64);q[:,3:7]=torch.tensor(Rotation.from_euler('z',.4*t).as_quat())
        v=torch.zeros(1,35,dtype=q.dtype);v[:,5]=.4
        return WholeBodyReference(q,v,v*0.)

    def test_world_rates_include_rotating_axis_term(self):
        path=WindowedPelvisRoll(2.,-.03)
        for t in (.8,1.2,1.6):
            h=1e-5
            lo,mid,hi=[path.sample(self.source(s),s) for s in (t-h,t,t+h)]
            rotation=lambda ref:Rotation.from_quat(ref.q[0,3:7].numpy())
            omega=(rotation(hi)*rotation(lo).inv()).as_rotvec()/(2*h)
            np.testing.assert_allclose(omega,mid.v[0,3:6],atol=1e-8)
            torch.testing.assert_close((hi.v-lo.v)/(2*h),mid.a,atol=1e-8,rtol=1e-7)

    def test_preserves_boundaries_input_and_other_coordinates(self):
        path=WindowedPelvisRoll(2.,.03)
        for t in (0.,.6,1.8,2.):
            ref=self.source(t)
            self.assertIs(path.sample(ref,t),ref)
        ref=self.source(1.2);old=ref.q.clone();out=path.sample(ref,1.2)
        torch.testing.assert_close(ref.q,old)
        torch.testing.assert_close(out.q[:,:3],ref.q[:,:3])
        torch.testing.assert_close(out.q[:,7:],ref.q[:,7:])
        torch.testing.assert_close(out.v[:,6:],ref.v[:,6:])
        torch.testing.assert_close(out.q[:,3:7].norm(dim=-1),torch.ones(1,dtype=ref.q.dtype))
        for t in (.6+1e-8,1.8-1e-8):
            self.assertLess(float(path.sample(self.source(t),t).a.abs().max()),1e-5)
        for duration,amplitude in ((0.,.1),(2.,float('nan'))):
            with self.assertRaises(ValueError):WindowedPelvisRoll(duration,amplitude)
