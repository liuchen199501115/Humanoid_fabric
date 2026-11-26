import unittest
import numpy as np
from preview_collision import CollisionDynamics
from preview_manifold import integrate
from test_full_body_preview_audit import CartesianModel,StaticReference
from full_body_preview_collision import FullBodyPreview

class CollisionModel(CartesianModel):
 def __init__(self,bound):self.bound=bound
 def evaluate(self,*args,**kwargs):
  s=super().evaluate(*args,**kwargs);s['collision']=dict(distance=np.array([.1]),jacobian=np.zeros((1,35)),bound=np.array([self.bound]));return s

class BarrierAuditTests(unittest.TestCase):
 def audit(self,bound):
  planner=FullBodyPreview(CollisionModel(bound),StaticReference(),None,None,[(np.zeros((1,3)),np.zeros(1))]*2,None,None,None,horizon=2)
  plan=dict(frames=np.arange(3),states=np.zeros((3,70)),controls=np.zeros((2,76)),impulses=np.zeros((2,12)),impact_maps={})
  return planner.nonlinear_audit(plan,1.)
 def test_feasible_barrier_admitted(self):self.assertTrue(self.audit(1.)['nonlinear_verified'])
 def test_dynamically_valid_but_collision_infeasible_plan_rejected(self):
  r=self.audit(-1.);self.assertFalse(r['nonlinear_verified']);self.assertEqual(r['nonlinear_collision_barrier'],1.)

class SphereDerivativeTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  import xml.etree.ElementTree as ET
  urdf='src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf'
  names=[j.get('name') for j in ET.parse(urdf).getroot().findall('joint') if j.get('type')!='fixed']
  geometry=dict(spheres={'a':dict(link='left_wrist_yaw_link',center=[.02,-.01,.03],radius=.03),'b':dict(link='torso_link',center=[.04,.02,.1],radius=.08)},pairs=[('a','b')],safe_distance=.02)
  cls.model=CollisionDynamics(urdf,names,geometry)
 def test_distance_velocity_and_bias_match_finite_differences(self):
  rng=np.random.default_rng(614);q=np.r_[[.2,.1,.8],[0.,0.,0.,1.],rng.normal(0,.15,29)];v=rng.normal(0,.5,35);eps=1e-5
  c=self.model.evaluate(q,v,kinematic_bias=True)['collision'];plus=self.model.evaluate(integrate(q,eps*v),v,kinematic_bias=True)['collision'];minus=self.model.evaluate(integrate(q,-eps*v),v,kinematic_bias=True)['collision']
  np.testing.assert_allclose((plus['distance']-minus['distance'])/(2*eps),c['speed'],atol=1e-8)
  np.testing.assert_allclose((plus['jacobian']-minus['jacobian'])@v/(2*eps),c['bias'],atol=1e-7)
 def test_common_ancestors_and_unrelated_joints_do_not_change_barrier(self):
  rng=np.random.default_rng(47);q=np.r_[[.2,-.1,.8],[0.,0.,0.,1.],rng.normal(0,.2,29)];v=rng.normal(0,.7,35)
  initial=self.model.evaluate(q,v,kinematic_bias=True)['collision'];irrelevant=~initial['dependencies'][0]
  delta=rng.normal(0,.05,35)*irrelevant
  moved=self.model.evaluate(integrate(q,delta),v+rng.normal(0,.2,35)*irrelevant,kinematic_bias=True)['collision']
  for key in ('distance','jacobian','speed','bias','bound'):
   np.testing.assert_allclose(initial[key],moved[key],atol=1e-10)
 def test_root_rigid_motion_does_not_change_self_distance(self):
  q=np.r_[[0.,0.,.8],[0.,0.,0.,1.],np.zeros(29)];v=np.r_[[1.,2.,3.,.4,.5,.6],np.zeros(29)]
  c=self.model.evaluate(q,v,kinematic_bias=True)['collision'];np.testing.assert_allclose(c['jacobian'][:,:6],0.,atol=1e-12);np.testing.assert_allclose(c['bias'],0.,atol=1e-12)


class SingleSupportModel(CartesianModel):
 def __init__(self):
  from types import SimpleNamespace
  self.model=SimpleNamespace(body_mass=np.array([25./9.81]))
 def evaluate(self,*args,**kwargs):
  s=super().evaluate(*args,**kwargs);s['bias'][2]=25.
  s['collision']=dict(distance=np.array([.1,.6]),jacobian=np.zeros((2,35)),bound=np.ones(2));return s

class ContactModeRegressionTests(unittest.TestCase):
 def setUp(self):
  self.planner=FullBodyPreview(SingleSupportModel(),StaticReference(),StaticReference(),np.full(1000,2),[(np.zeros((1,3)),np.zeros(1))]*2,np.tile([-10.,10.],(29,1)),np.full(29,100.),np.full(29,30.),horizon=2)
 def test_collision_activity_does_not_enable_swing_foot_force(self):
  q=StaticReference().kinematics(0.,200.)[0].numpy();u=np.zeros(76);u[43]=25.
  first=dict(A_eq=np.eye(76),b_eq=u,G=np.zeros((0,76)),b=np.zeros(0),lb=np.full(76,-1e6),ub=np.full(76,1e6),mode=2)
  result=self.planner.refine(q,np.zeros(35),0.,200.,first,iterations=3)
  self.assertTrue(result['success'],result['status'])
  np.testing.assert_allclose(result['controls'][1,35:41],0.,atol=1e-6)
  self.assertAlmostEqual(result['controls'][1,43],25.+result['controls'][1,2],places=7)
  self.assertGreater(result['controls'][1,43],24.)
 def test_future_barrier_changes_the_optimized_acceleration(self):
  original=self.planner.model.evaluate
  def constrained(*args,**kwargs):
   state=original(*args,**kwargs);state['collision']['jacobian'][0,6]=1.;state['collision']['bound'][0]=-2.;return state
  self.planner.model.evaluate=constrained
  q=StaticReference().kinematics(0.,200.)[0].numpy();u=np.zeros(76);u[43]=25.
  first=dict(A_eq=np.eye(76),b_eq=u,G=np.zeros((0,76)),b=np.zeros(0),lb=np.full(76,-1e6),ub=np.full(76,1e6),mode=2)
  result=self.planner.refine(q,np.zeros(35),0.,200.,first,iterations=3)
  self.assertTrue(result['success'],result['status'])
  self.assertGreaterEqual(result['controls'][1,6],2.-1e-6)
  self.assertAlmostEqual(result['controls'][1,47],result['controls'][1,6],places=7)
 def test_balanced_force_on_inactive_foot_is_rejected(self):
  plan=dict(frames=np.array([0.,2.,4.]),step_durations=np.array([.01,.01]),states=np.zeros((3,70)),controls=np.zeros((2,76)),impulses=np.zeros((2,12)),impact_maps={})
  plan['controls'][:,37]=12.;plan['controls'][:,43]=13.
  r=self.planner.nonlinear_audit(plan,200.)
  self.assertLess(r['nonlinear_base_force_residual'],1e-10)
  self.assertEqual(r['nonlinear_inactive_wrench'],12.)
  self.assertFalse(r['nonlinear_verified'])

if __name__=='__main__':unittest.main()
