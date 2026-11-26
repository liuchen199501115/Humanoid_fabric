import copy,unittest,numpy as np
from verified_preview_reuse import reusable_control
class ReuseTests(unittest.TestCase):
 def setUp(self):
  q=np.tile(np.r_[np.zeros(6),1.,np.zeros(29)],(3,1));q[:,0]=[0.,.00005,.025925]
  v=np.zeros((3,35));v[:,0]=[0.,1.01,1.06]
  controls=np.zeros((2,76));controls[:,0]=[1.,2.]
  self.previous=dict(success=True,nonlinear_verified=True,frames=np.array([0.,2.,7.]),physical_q=q,physical_v=v,controls=controls,contact_modes=np.array([2,2,2]))
  self.q=q[1].copy();self.q[0]+=.005*1.01+.5*.005**2*2.;self.v=v[1].copy();self.v[0]+=.005*2.
 def call(self,**changes):
  args=dict(previous=self.previous,q=self.q,v=self.v,frame=3.,rate=200.,mode=2);args.update(changes);return reusable_control(**args)
 def test_uses_postimpact_velocity_and_current_stage_control(self):
  result=self.call();self.assertEqual(result['stage'],1);self.assertEqual(result['plan'][2][0],2.);self.assertIsNone(result['plan'][3]);np.testing.assert_allclose(result['plan'][0],self.q);np.testing.assert_allclose(result['plan'][1],self.v)
 def test_roundoff_before_terminal_frame_uses_last_control(self):
  result=self.call(frame=7.-1e-10,q=self.previous['physical_q'][-1],v=self.previous['physical_v'][-1])
  self.assertIsNotNone(result);self.assertEqual(result['stage'],1)
 def test_expired_prediction_rejected(self):self.assertIsNone(self.call(frame=7.))
 def test_contact_mode_mismatch_rejected(self):self.assertIsNone(self.call(mode=1))
 def test_unverified_prediction_rejected(self):
  bad=copy.deepcopy(self.previous);bad['nonlinear_verified']=False;self.assertIsNone(self.call(previous=bad))
 def test_pose_departure_rejected(self):
  q=self.q.copy();q[0]+=.03;self.assertIsNone(self.call(q=q))
 def test_velocity_departure_rejected(self):
  v=self.v.copy();v[0]+=1.;self.assertIsNone(self.call(v=v))
if __name__=='__main__':unittest.main()
