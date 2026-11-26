"""Independent admission checks for experimental nonlinear predictions."""
import copy
import unittest
from types import SimpleNamespace
import numpy as np
import torch
from full_body_preview import FullBodyPreview

class StaticReference:
 def kinematics(self, frame, rate):
  q=np.zeros(36);q[6]=1
  return torch.tensor(q),torch.zeros(35,dtype=torch.float64),torch.zeros(35,dtype=torch.float64)
class CartesianModel:
 def evaluate(self,q,v,kinematic_bias=False):
  J=np.zeros((2,6,35));J[:,:6,:6]=np.eye(6)
  feet=np.tile(np.r_[q[:3],0.,0.,0.,1.],(2,1))
  return dict(mass=np.eye(35),bias=np.zeros(35),feet=feet,jacobian=J,foot_bias=np.zeros((2,6)))

class NonlinearAuditTests(unittest.TestCase):
 def setUp(self):
  self.planner=FullBodyPreview(CartesianModel(),StaticReference(),None,None,[(np.zeros((1,3)),np.zeros(1))]*2,None,None,None,dt=.025,horizon=2)
  self.plan=dict(frames=np.arange(3),states=np.zeros((3,70)),controls=np.zeros((2,76)),impulses=np.zeros((2,12)),impact_maps={})
 def audit(self):return self.planner.nonlinear_audit(self.plan,1.)
 def test_static_feasible_prediction(self):self.assertTrue(self.audit()['nonlinear_verified'])
 def test_unbalanced_force_rejected(self):
  self.plan['controls'][1,37]=30
  audit=self.audit();self.assertEqual(audit['nonlinear_base_force_residual'],30.);self.assertFalse(audit['nonlinear_verified'])
 def test_penetration_rejected(self):
  self.plan['states'][:,2]=-.003
  audit=self.audit();self.assertEqual(audit['nonlinear_minimum_clearance'],-.003);self.assertFalse(audit['nonlinear_verified'])
 def test_state_teleport_rejected(self):
  self.plan['states'][2,10]=.05
  audit=self.audit();self.assertGreater(audit['nonlinear_integration_joint'],.049);self.assertFalse(audit['nonlinear_verified'])
 def test_impulse_momentum_mismatch_rejected(self):
  self.plan['impact_maps']={0:[0]};self.plan['impulses'][0,0]=1.
  audit=self.audit();self.assertEqual(audit['nonlinear_impact_residual'],1.);self.assertFalse(audit['nonlinear_verified'])
 def test_energy_creating_impulse_rejected(self):
  self.plan['impact_maps']={0:[0]};self.plan['impulses'][0,0]=1.
  self.plan['states'][1:,35]=1.;self.plan['states'][2,0]=self.planner.dt
  audit=self.audit();self.assertLess(audit['nonlinear_impact_residual'],1e-10);self.assertEqual(audit['nonlinear_impact_energy_increase'],.5);self.assertFalse(audit['nonlinear_verified'])
 def test_dissipative_impact_accepted(self):
  self.plan['impact_maps']={0:[0]};self.plan['impulses'][0,2]=1.
  self.plan['states'][0,37]=-1.;self.plan['states'][0,2]=self.planner.dt
  audit=self.audit();self.assertLess(audit['nonlinear_impact_residual'],1e-10);self.assertEqual(audit['nonlinear_impact_energy_increase'],0.);self.assertTrue(audit['nonlinear_verified'])
 def test_original_landing_error_is_rejected(self):
  self.planner.source=StaticReference();self.planner.modes=np.array([0,1,1])
  self.plan['states'][:,0]=.041
  result=self.audit();self.assertFalse(result['nonlinear_verified'])
  self.assertAlmostEqual(result['nonlinear_landing_error'],.041)
 def test_original_landing_within_tolerance_is_admitted(self):
  self.planner.source=StaticReference();self.planner.modes=np.array([0,1,1])
  self.plan['states'][:,0]=.039
  self.assertTrue(self.audit()['nonlinear_verified'])
 def test_nonuniform_integration_audit_accepts_correct_intervals(self):
  times=np.array([0.,.01,.035]);self.plan['step_durations']=np.diff(times)
  self.plan['states'][:,0]=.5*times**2;self.plan['states'][:,35]=times;self.plan['controls'][:,0]=1.
  self.assertTrue(self.audit()['nonlinear_verified'])
 def test_nonuniform_integration_audit_rejects_wrong_interval(self):
  times=np.array([0.,.01,.035]);self.plan['step_durations']=np.array([.025,.025])
  self.plan['states'][:,0]=.5*times**2;self.plan['states'][:,35]=times;self.plan['controls'][:,0]=1.
  self.assertFalse(self.audit()['nonlinear_verified'])
 def test_simulator_semiimplicit_path_is_admitted(self):
  self.plan['step_durations']=np.array([.01,.024])
  self.plan['states'][:,6]=[0.,.006,.0612];self.plan['states'][:,41]=[0.,1.,3.4]
  self.plan['controls'][:,6]=100.;self.plan['controls'][:,47]=100.
  self.assertTrue(self.audit()['nonlinear_verified'])
 def test_continuous_formula_does_not_pass_semiimplicit_audit(self):
  times=np.array([0.,.01,.034]);self.plan['step_durations']=np.array([.01,.024])
  self.plan['states'][:,6]=.5*100.*times**2;self.plan['states'][:,41]=100.*times
  self.plan['controls'][:,6]=100.;self.plan['controls'][:,47]=100.
  self.assertFalse(self.audit()['nonlinear_verified'])
if __name__=='__main__':unittest.main()
