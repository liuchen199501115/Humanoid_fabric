import unittest
from types import SimpleNamespace
from unittest.mock import patch
import torch
from fabrics_sim.reference.mocap_transition import MocapContactTransition
from fabrics_sim.reference.whole_body import WholeBodyReference


class Transfer:
    def __init__(self, fabric, reference, names, feet, support, time, **kwargs):
        self.reference=reference
        self.support=support
        self.start_time=time
        self.duration=2.
        self.phase='TRANSFER'
    def sample(self,time,mode):
        return self.reference
    def ready_for_release(self,time,com,velocity,q):
        return time >= self.start_time+self.duration


class MocapTransitionTests(unittest.TestCase):
    def setUp(self):
        q=torch.zeros(1,36,dtype=torch.float64);q[:,6]=1.
        self.source=WholeBodyReference(q,torch.zeros(1,35,dtype=q.dtype),torch.zeros(1,35,dtype=q.dtype))
        q=q.clone();q[:,0]=.02
        v=torch.zeros(1,35,dtype=q.dtype);v[:,0]=.03
        self.measured=WholeBodyReference(q,v,torch.zeros_like(v))
        self.feet=torch.zeros(1,2,7,dtype=q.dtype);self.feet[:,:,6]=1.
        self.manager=SimpleNamespace(modes=[3]*114+[1]*15+[3]*8+[2]*18+[3],
                                     pending_takeoff=114,landing_frame=129,speed=1.)
        self.clock=SimpleNamespace(phase='PREPARE_STEP',mode=3,frame=112.5)
        model=SimpleNamespace(snapshot=lambda q: {'com': q[:,:3]})
        fabric=SimpleNamespace(dynamics_tasks=SimpleNamespace(model=model))
        self.controller=MocapContactTransition(fabric,[])
        projection=patch('fabrics_sim.reference.mocap_transition.contact_pose_reference',
            side_effect=lambda fabric,q,names,feet,xy,z,**kwargs:q.clone())
        projection.start();self.addCleanup(projection.stop)
        self.factory=patch('fabrics_sim.reference.mocap_transition.ContactTransferReference',Transfer)
        self.factory.start();self.addCleanup(self.factory.stop)
    def sample(self,time,stopped=True):
        return self.controller.sample(self.source,self.measured,self.feet,time,
                                      self.clock,self.manager,30.,stopped,torch.zeros(1,3),torch.tensor([[100.,200.]]),.2)
    def test_entry_uses_stopped_source_and_exit_restores_source(self):
        entry=self.sample(0.)
        for name in ('q','v','a'):
            torch.testing.assert_close(getattr(entry,name),getattr(self.source,name))
        self.assertEqual(self.controller.unload_weight(0.),0.)
        self.assertEqual(self.controller.unload_weight(2.),1.)
        self.clock.phase='SWING';self.clock.mode=1;self.clock.frame=114.
        released=self.sample(2.)
        for name in ('q','v','a'):
            torch.testing.assert_close(getattr(released,name),getattr(self.measured,name))
        self.assertIs(self.sample(2.25),self.source)
        self.assertEqual(self.controller.unload_weight(2.25),0.)
    def test_measured_motion_prevents_transfer(self):
        result=self.controller.sample(self.source,self.measured,self.feet,0.,self.clock,
            self.manager,30.,True,torch.tensor([[.1,0.,0.]]),torch.tensor([[100.,200.]]),.2)
        self.assertIs(result,self.source)
        self.assertIsNone(self.controller.transfer)
        self.assertEqual(self.controller.phase,'WAIT_FOR_MEASURED_REST')
        self.sample(.2)
        self.assertIsNotNone(self.controller.transfer)

    def test_reference_must_stop_before_transfer(self):
        self.assertIs(self.sample(0.,False),self.source)
        self.assertIsNone(self.controller.transfer)
    def test_next_left_step_has_independent_transfer(self):
        self.sample(0.)
        self.clock.phase='SWING';self.clock.mode=1;self.clock.frame=114.
        self.sample(2.);self.sample(2.25)
        self.clock.phase='PREPARE_STEP';self.clock.mode=3;self.clock.frame=135.5
        self.manager.pending_takeoff=137;self.manager.landing_frame=155
        self.sample(3.)
        self.assertEqual(self.controller.transfer.support,1)
        self.assertEqual(self.controller.unload_weight(3.),0.)
        self.assertIsNone(self.controller.release_time)
    def test_landing_approach_is_not_touchdown(self):
        self.sample(0.)
        self.clock.phase='SWING';self.clock.mode=1;self.clock.frame=114.
        self.sample(2.)
        self.clock.phase='LANDING'
        self.sample(2.1)
        self.assertEqual(self.controller.phase,'SOURCE_JOIN')

    def test_reject_early_touchdown_during_join(self):
        self.sample(0.)
        self.clock.phase='SWING';self.clock.mode=1;self.clock.frame=114.
        self.sample(2.)
        self.clock.phase='LANDING';self.clock.mode=3
        with self.assertRaisesRegex(RuntimeError,'Touchdown'):
            self.sample(2.1)


if __name__ == '__main__':
    unittest.main()
