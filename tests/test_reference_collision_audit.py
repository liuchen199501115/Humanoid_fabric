import unittest
from types import SimpleNamespace
import torch
from fabrics_sim.solver.collision_barrier import wrist_collision_barrier
from fabrics_sim.reference.collision_audit import collision_reference_check, audit_collision_rows
from fabrics_sim.reference.whole_body import WholeBodyReference


class ReferenceCollisionTests(unittest.TestCase):
    def test_runtime_formula_and_fixed_rows_have_same_active_constraints(self):
        distance=torch.tensor([[.01,.1,.5,.01]])
        jac=torch.arange(140,dtype=torch.float32).reshape(1,4,35)/100
        bias=torch.tensor([[0.,.2,.3,.4]])
        speed=torch.tensor([[0.,-.1,0.,0.]])
        G,b=wrist_collision_barrier(distance,jac,bias,speed,1,3,.02)
        self.assertEqual(G.shape,(1,1,35))
        torch.testing.assert_close(G,-jac[:,1:2])
        torch.testing.assert_close(b,bias[:,1:2]+16*speed[:,1:2]+64*(distance[:,1:2]-.02))
        fixed,fb=wrist_collision_barrier(distance,jac,bias,speed,1,3,.02,True)
        torch.testing.assert_close(fixed[:,:1],G)
        torch.testing.assert_close(fb[:,:1],b)
        torch.testing.assert_close(fixed[:,1:],torch.zeros_like(fixed[:,1:]))
        torch.testing.assert_close(fb[:,1:],torch.zeros_like(fb[:,1:]))

    def fixture(self,distance):
        jac=torch.zeros(1,1,35)
        tasks=SimpleNamespace(collision_enabled=True,wrist_hip_barrier=True,
            config={'difference_step':.001,'collision':{'safe_distance':.02}},
            original_collision_pairs=0,pairs=[(0,1)],
            model=SimpleNamespace(evaluate=lambda *args:({}, {}, {})),
            _sphere_geometry=lambda state:(torch.tensor([[distance]]),jac))
        fabric=SimpleNamespace(dynamics_tasks=tasks,integrate_root_position=None)
        q=torch.zeros(1,36);q[:,6]=1.
        return fabric,WholeBodyReference(q,torch.zeros(1,35),torch.zeros(1,35))

    def test_static_overlap_rejected_even_with_zero_torque_or_motion(self):
        fabric,ref=self.fixture(-.05)
        result=collision_reference_check(fabric,ref)
        self.assertTrue(result['checked']);self.assertFalse(result['passed'])
        self.assertAlmostEqual(result['maximum_violation'],64*.07,places=5)
        row=dict(time=.2,**{k:getattr(ref,k).numpy() for k in ('q','v','a')})
        report=audit_collision_rows(fabric,[row],ref.q)
        self.assertFalse(report['passed']);self.assertEqual(report['violating_samples'],1)
        self.assertEqual(report['worst_time'],.2)
        self.assertEqual(report['violation_times'],[.2])

    def test_clear_geometry_passes_and_disabled_barrier_is_explicit(self):
        fabric,ref=self.fixture(.1)
        self.assertTrue(collision_reference_check(fabric,ref)['passed'])
        fabric.dynamics_tasks.wrist_hip_barrier=False
        self.assertFalse(collision_reference_check(fabric,ref)['checked'])
