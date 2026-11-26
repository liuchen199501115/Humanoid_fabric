import unittest
from pathlib import Path
import numpy as np,torch
from measured_stance_reference import MeasuredStanceReference,Anchor
from blended_spline import BlendedSplineReference
from fabrics_sim.reference.interpolation import MotionInterpolator
from fabrics_sim.reference.whole_body import WholeBodyReference

class MeasuredStanceTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  p=Path(__file__).parent;cls.q=torch.tensor(np.load(p/'turn_recovery_y40.npz')['control_q']);cls.names=np.load(p/'wrench_fixed_ten.npz')['final_joint_order'];cls.sampler=BlendedSplineReference(MotionInterpolator(cls.q),cls.q,stride=6,start_frame=130,ramp_frames=20)
 def projector(self):return MeasuredStanceReference('src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf',self.names,np.full(len(self.q),3),200,limit=.01)
 def test_inactive_exact_and_input_unchanged(self):
  p=self.projector();r=WholeBodyReference(*self.sampler.kinematics(820.,240.));before=[x.clone() for x in (r.q,r.v,r.a)];o=p.project(820.,240.,0.,self.sampler,r)
  for x,y,z in zip(before,(r.q,r.v,r.a),(o.q,o.v,o.a)):self.assertTrue(torch.equal(x,y));self.assertTrue(torch.equal(x,z))
 def test_source_endpoint_is_covered(self):
  p=self.projector()
  for f in (0.,1999.):
   r=WholeBodyReference(*self.sampler.kinematics(f,240.));o=p.project(f,240.,0.,self.sampler,r)
   for a,b in zip((r.q,r.v,r.a),(o.q,o.v,o.a)):self.assertTrue(torch.equal(a,b))
 def test_reference_correction_has_continuous_rates(self):
  p=self.projector();q=self.sampler.kinematics(820.,240.)[0].numpy()[0];pos,R,_=p.model.evaluate(q,1);local=np.array([0.,0.,-.035]);p.anchors[1]=Anchor(800.,908.,pos+R@local+[.005,.003,0.],local)
  def sample(f):return p.project(f,240.,0.,self.sampler,WholeBodyReference(*self.sampler.kinematics(f,240.)))
  for f in (800.,810.3,830.3,890.3,908.):
   o=sample(f);a=sample(f-.24);b=sample(f+.24)
   self.assertLess(float(abs((b.q[:,7:]-a.q[:,7:])/.002-o.v[:,6:]).max()),.02)
   self.assertLess(float(abs((b.v[:,6:]-a.v[:,6:])/.002-o.a[:,6:]).max()),2.)
   original=self.sampler.kinematics(f,240.)[0]
   self.assertTrue(torch.equal(o.q[:,:13],original[:,:13]));self.assertTrue(torch.equal(o.q[:,19:],original[:,19:]))
 def test_invalid_parameters_rejected(self):
  for limit in (0.,-1.,float('nan')):
   with self.assertRaises(ValueError):MeasuredStanceReference('',[],[],200,limit=limit)
if __name__=='__main__':unittest.main()
