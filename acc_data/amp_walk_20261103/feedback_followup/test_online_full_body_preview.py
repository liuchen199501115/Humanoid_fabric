import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from online_full_body_preview import OnlineFullBodyPreview

class Planner:
 def __init__(self,fail_second=False):self.calls=0;self.fail_second=fail_second
 def refine(self,*args):
  self.calls+=1
  ok=not(self.calls==2 and self.fail_second)
  return dict(success=ok,status='Solved' if ok else 'PrimalInfeasible',linear_violation=0.,tube_violation=0.,objective=0.,first_control=np.zeros(76))

class ProjectionRecoveryTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
  self.q=torch.zeros((1,36));self.q[0,6]=1.;self.v=torch.zeros((1,35))
  problem=SimpleNamespace(A_eq=torch.zeros((1,0,76)),b_eq=torch.zeros((1,0)),G=torch.zeros((1,0,76)),b=torch.zeros((1,0)),lb=torch.full((1,76),-1000.),ub=torch.full((1,76),1000.))
  self.owner=SimpleNamespace(last_priority_problems=[problem]);self.projections=0;self.baseline=SimpleNamespace(validated=torch.tensor([True]),predicted_acc=torch.zeros((1,35)))
 def solver(self,**kwargs):
  self.last_kwargs=kwargs
  if not kwargs['task_requests']:return self.baseline
  self.projections+=1;acc=torch.zeros((1,35))
  if self.projections==2:acc[0,10]=100.
  return SimpleNamespace(validated=torch.tensor([True]),predicted_acc=acc)
 def call(self,wrapper,now):
  return wrapper.solve(self.solver,self.owner,self.q,self.v,now,now*200.,200.,task_requests=[],contact_mode=torch.tensor([2]))
 def test_unreachable_held_action_triggers_fresh_prediction(self):
  planner=Planner();wrapper=OnlineFullBodyPreview(planner,Path(self.tmp.name)/'plans.json')
  self.call(wrapper,0.);result=self.call(wrapper,.002)
  self.assertEqual(planner.calls,2);self.assertEqual(self.projections,3)
  self.assertTrue(result.validated.all());self.assertEqual(wrapper.records[-1]['reason'],'contact_projection')
  self.assertTrue(wrapper.records[-1]['applied'])
 def test_failed_replan_returns_validated_baseline(self):
  planner=Planner(True);wrapper=OnlineFullBodyPreview(planner,Path(self.tmp.name)/'plans.json')
  self.call(wrapper,0.);result=self.call(wrapper,.002)
  self.assertIs(result,self.baseline);self.assertIsNone(wrapper.plan)
  self.assertFalse(wrapper.records[-1]['success'])
 def test_wrench_reference_is_forwarded_only_when_requested(self):
  planner=Planner();wrapper=OnlineFullBodyPreview(planner,Path(self.tmp.name)/'plans.json')
  self.call(wrapper,0.)
  self.assertNotIn('contact_wrench_reference',self.last_kwargs)
  wrapper.track_wrench=True;wrapper.plan[3][2]=123.;self.projections=0
  self.call(wrapper,.002)
  self.assertEqual(float(self.last_kwargs['contact_wrench_reference'][0,2]),123.)
  self.assertTrue(torch.equal(self.last_kwargs['contact_wrench_weights'],torch.ones((1,12))))
 def reusable_planner(self):
  class ReusablePlanner(Planner):
   def refine(inner,*args):
    result=super(ReusablePlanner,inner).refine(*args)
    result.update(nonlinear_verified=result['success'],frames=np.array([0.,5.,10.]),physical_q=np.tile(self.q.numpy()[0],(3,1)),physical_v=np.zeros((3,35)),controls=np.zeros((2,76)),contact_modes=np.array([2,2,2]))
    return result
  return ReusablePlanner(True)
 def test_reused_prediction_still_passes_current_physical_projection(self):
  wrapper=OnlineFullBodyPreview(self.reusable_planner(),Path(self.tmp.name)/'plans.json',reuse_verified=True)
  self.call(wrapper,0.);result=self.call(wrapper,.002)
  self.assertIsNot(result,self.baseline);self.assertTrue(result.validated.all());self.assertEqual(self.projections,3)
  self.assertTrue(wrapper.records[-1]['reused_verified_prediction']);self.assertFalse(wrapper.records[-1]['success'])
 def test_invalid_reused_projection_returns_baseline(self):
  wrapper=OnlineFullBodyPreview(self.reusable_planner(),Path(self.tmp.name)/'plans.json',reuse_verified=True)
  self.call(wrapper,0.)
  def invalid_solver(**kwargs):
   result=self.solver(**kwargs)
   if self.projections==3:result.validated=torch.tensor([False])
   return result
  result=wrapper.solve(invalid_solver,self.owner,self.q,self.v,.002,.4,200.,task_requests=[],contact_mode=torch.tensor([2]))
  self.assertIs(result,self.baseline);self.assertIsNone(wrapper.plan)
 def test_unmatched_observation_discards_cached_action_and_replans(self):
  planner=Planner();wrapper=OnlineFullBodyPreview(planner,Path(self.tmp.name)/'plans.json')
  self.call(wrapper,0.);wrapper.observation_failure(.002,.4,'unmatched point')
  self.assertIsNone(wrapper.plan);self.assertEqual(wrapper.records[-1]['status'],'ObservedPatchUnmatched')
  self.projections=0;self.call(wrapper,.004);self.assertEqual(planner.calls,2)

class PlanFollowingTests(unittest.TestCase):
 setUp=ProjectionRecoveryTests.setUp
 solver=ProjectionRecoveryTests.solver
 call=ProjectionRecoveryTests.call
 reusable_planner=ProjectionRecoveryTests.reusable_planner
 def test_later_stage_is_sampled_without_integrating_elapsed_time_twice(self):
  class ScheduledPlanner(Planner):
   def refine(inner,*args):
    result=super(ScheduledPlanner,inner).refine(*args)
    qs=np.tile(self.q.numpy()[0],(4,1));vs=np.zeros((4,35));controls=np.zeros((3,76));controls[:,6]=[0.,2.,-1.]
    qs[2:,7]=[8e-6,12e-6];vs[2:,6]=[.004,.002]
    result.update(nonlinear_verified=True,integration='semiimplicit',frames=np.array([0.,.4,.8,1.2]),physical_q=qs,physical_v=vs,controls=controls,contact_modes=np.full(4,2))
    return result
  planner=ScheduledPlanner();wrapper=OnlineFullBodyPreview(planner,Path(self.tmp.name)/'plans.json',period=.04,track_wrench=True,follow_plan=True)
  def exact_solver(**kwargs):
   self.last_kwargs=kwargs
   return self.baseline if not kwargs['task_requests'] else SimpleNamespace(validated=torch.tensor([True]),predicted_acc=kwargs['task_requests'][-1].desired.clone())
  wrapper.solve(exact_solver,self.owner,self.q,self.v,0.,0.,200.,task_requests=[],contact_mode=torch.tensor([2]))
  self.q[0,7]=8e-6;self.v[0,6]=.004
  result=wrapper.solve(exact_solver,self.owner,self.q,self.v,.004,.8,200.,task_requests=[],contact_mode=torch.tensor([2]))
  self.assertEqual(planner.calls,1);self.assertTrue(result.validated.all());self.assertAlmostEqual(float(result.predicted_acc[0,6]),-1.,places=6)
  self.assertEqual(wrapper.records[-1]['latest_sampled_stage'],2)
  self.assertNotIn('contact_wrench_reference',self.last_kwargs)
 def test_validated_wbc_correction_is_allowed_until_state_departs(self):
  planner=self.reusable_planner();wrapper=OnlineFullBodyPreview(planner,Path(self.tmp.name)/'plans.json',period=.04,follow_plan=True,allow_projection_correction=True)
  self.call(wrapper,0.);result=self.call(wrapper,.002)
  self.assertIsNot(result,self.baseline);self.assertTrue(result.validated.all());self.assertEqual(planner.calls,1)
  self.assertTrue(wrapper.records[-1]['projection_corrected_by_wbc'])
  self.q[0,0]=.03;result=self.call(wrapper,.004)
  self.assertIs(result,self.baseline);self.assertEqual(planner.calls,2)
 def test_invalid_physical_projection_is_rejected_in_state_feedback_mode(self):
  planner=self.reusable_planner();wrapper=OnlineFullBodyPreview(planner,Path(self.tmp.name)/'plans.json',period=.04,follow_plan=True,allow_projection_correction=True)
  self.call(wrapper,0.)
  def invalid(**kwargs):
   result=self.solver(**kwargs)
   if kwargs['task_requests']:result.validated=torch.tensor([False])
   return result
  result=wrapper.solve(invalid,self.owner,self.q,self.v,.002,.4,200.,task_requests=[],contact_mode=torch.tensor([2]))
  self.assertIs(result,self.baseline);self.assertEqual(planner.calls,2)
 def test_unobserved_planned_contact_transition_forces_replan(self):
  planner=self.reusable_planner();original=planner.refine
  def with_transition(*args):
   result=original(*args);result['contact_modes']=np.array([2,1,1]);return result
  planner.refine=with_transition
  wrapper=OnlineFullBodyPreview(planner,Path(self.tmp.name)/'plans.json',period=.04,follow_plan=True)
  self.call(wrapper,0.);result=self.call(wrapper,.026)
  self.assertEqual(planner.calls,2);self.assertIs(result,self.baseline)
  self.assertTrue(wrapper.records[0]['plan_sample_mismatch'])

if __name__=='__main__':unittest.main()
