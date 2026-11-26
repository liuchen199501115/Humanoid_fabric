import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from preview_rotation_transition import rotation_step,linearize_rotation
from preview_integration import advance
from preview_manifold import integrate,difference
class RotationTransitionTests(unittest.TestCase):
 def setUp(self):
  self.e=np.array([.25,-.18,.32]);self.dv=np.array([1.2,-.8,.3]);self.a=np.array([12.,-8.,25.]);self.r=Rotation.from_euler('xyz',[.1,.2,1.4]).as_quat();self.v=np.array([.7,-.4,2.1]);self.n=Rotation.from_euler('xyz',[.12,.18,1.46]).as_quat()
 def test_matches_full_semiimplicit_physical_state_update(self):
  q=np.r_[[1.,2.,.7],self.r,np.zeros(29)];e=np.r_[np.zeros(3),self.e,np.zeros(29)];q=integrate(q,e)
  v=np.r_[np.zeros(3),self.v+self.dv,np.zeros(29)];a=np.r_[np.zeros(3),self.a,np.zeros(29)]
  target=np.r_[[0.,0.,0.],self.n,np.zeros(29)]
  for dt in [.002,.01,.024,.0073]:
   moved,_=advance(q,v,a,dt)
   np.testing.assert_allclose(rotation_step(self.e,self.dv,self.a,self.r,self.v,self.n,dt),difference(moved,target)[3:6],atol=1e-12)
 def test_linearization_is_exact_at_nominal(self):
  offset,A,V,B=linearize_rotation(self.e,self.dv,self.a,self.r,self.v,self.n,.024)
  np.testing.assert_allclose(offset+A@self.e+V@self.dv+B@self.a,rotation_step(self.e,self.dv,self.a,self.r,self.v,self.n,.024),atol=1e-12)
 def test_linearization_predicts_combined_rotation_velocity_and_acceleration_changes(self):
  rng=np.random.default_rng(91);offset,A,V,B=linearize_rotation(self.e,self.dv,self.a,self.r,self.v,self.n,.024)
  errors=[]
  direction=rng.normal(size=(3,3))
  for scale in [1e-3,5e-4]:
   e,v,a=np.array([self.e,self.dv,self.a])+scale*direction
   exact=rotation_step(e,v,a,self.r,self.v,self.n,.024);linear=offset+A@e+V@v+B@a
   errors.append(np.linalg.norm(exact-linear))
  self.assertLess(errors[1],.3*errors[0]);self.assertLess(errors[0],2e-7)
if __name__=='__main__':unittest.main()
