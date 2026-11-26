import unittest
import numpy as np
import torch
from full_body_preview import integrate,difference
from preview_warm_start import shift_prediction
class Ref:
 def kinematics(self,f,r):
  q=np.zeros(36);q[6]=1.
  return torch.tensor(q),torch.zeros(35,dtype=torch.float64),torch.zeros(35,dtype=torch.float64)
class WarmStartTests(unittest.TestCase):
 def test_constant_acceleration_preserved(self):
  dt=.025;n=4;a=np.arange(35,dtype=float)/100;a[3:6]=0.;v=np.ones(35);v[3:6]=0.
  states=np.array([np.r_[v*k*dt+.5*a*(k*dt)**2,v+a*k*dt] for k in range(n+1)])
  controls=np.zeros((n,76));controls[:,:35]=a
  result=shift_prediction(dict(frames=np.arange(n+1)*5.,states=states,controls=controls),Ref(),2.,200.,dt,n,integrate,difference)
  for k in range(n+1):
   t=.01+k*dt;np.testing.assert_allclose(result['states'][k],np.r_[v*t+.5*a*t*t,v+a*t],atol=1e-12)
 def test_impact_is_not_smeared_into_previous_interval(self):
  states=np.zeros((3,70));states[1:,35]=1.;states[2,0]=.025
  previous=dict(frames=np.array([0.,5.,10.]),states=states,controls=np.zeros((2,76)))
  result=shift_prediction(previous,Ref(),2.,200.,.025,1,integrate,difference)
  self.assertEqual(result['states'][0,35],0.);self.assertEqual(result['states'][1,35],1.)
 def test_roundoff_at_impact_node_uses_postimpact_state(self):
  states=np.zeros((3,70));states[1:,35]=1.;states[2,0]=.025
  previous=dict(frames=np.array([0.,5.,10.]),states=states,controls=np.zeros((2,76)))
  result=shift_prediction(previous,Ref(),5.-1e-10,200.,.025,1,integrate,difference)
  self.assertEqual(result['states'][0,35],1.)
 def test_real_time_before_impact_stays_preimpact(self):
  states=np.zeros((3,70));states[1:,35]=1.;states[2,0]=.025
  previous=dict(frames=np.array([0.,5.,10.]),states=states,controls=np.zeros((2,76)))
  result=shift_prediction(previous,Ref(),5.-1e-6,200.,.025,1,integrate,difference)
  self.assertEqual(result['states'][0,35],0.)
 def test_stale_plan_not_used(self):
  previous=dict(frames=np.array([0.,5.,10.]))
  self.assertIsNone(shift_prediction(previous,Ref(),10.,200.,.025,1,integrate,difference))
 def test_nonuniform_grid_preserves_physical_motion(self):
  times=np.array([0.,.01,.035]);a=np.zeros(35);a[0]=2.;v=np.zeros(35);v[0]=.3
  states=np.array([np.r_[v*t+.5*a*t*t,v+a*t] for t in times]);controls=np.zeros((2,76));controls[:,:35]=a
  result=shift_prediction(dict(frames=times*200.,states=states,controls=controls),Ref(),1.,200.,np.array([.01,.025]),2,integrate,difference)
  for k,t in enumerate([.005,.015,.04]):np.testing.assert_allclose(result['states'][k],np.r_[v*t+.5*a*t*t,v+a*t],atol=1e-12)
 def test_nonuniform_grid_preserves_impact_time(self):
  states=np.zeros((3,70));states[1:,35]=1.;states[2,0]=.025
  previous=dict(frames=np.array([0.,2.,7.]),states=states,controls=np.zeros((2,76)))
  result=shift_prediction(previous,Ref(),1.,200.,np.array([.01,.025]),2,integrate,difference)
  np.testing.assert_allclose(result['states'][:,35],[0.,1.,1.])
  np.testing.assert_allclose(result['states'][:,0],[0.,.005,.03],atol=1e-12)
if __name__=='__main__':unittest.main()
