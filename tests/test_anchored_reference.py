import unittest
import torch
import warp as wp
from fabrics_sim.reference.whole_body import WholeBodyReference
from fabrics_sim.reference.anchored_reference import anchor_reference

class AnchoredReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from fabrics_sim.fabrics.g1_pose_fabric import G1PoseFabric
        from test_g1_two_tasks import G1_JOINT_NAMES
        wp.config.kernel_cache_dir='/tmp/g1_warp_cache';wp.init()
        cls.f=G1PoseFabric(1,'cpu',robot_name='g1_29dof_rev_1_0',task_profile='dynamics-aware');cls.names=G1_JOINT_NAMES
        cls.m=cls.f.dynamics_tasks.model
        cls.feet=[cls.m.kinematics.get_link_index(s+'_ankle_roll_link') for s in ('left','right')]
    def source(self):
        q=torch.zeros(1,36);q[:,2]=.75;q[:,6]=1
        for s in ('left','right'):
            for j,x in [('hip_pitch',-.2),('knee',.4),('ankle_pitch',-.2)]:q[:,7+self.names.index(s+'_'+j+'_joint')]=x
        v=torch.zeros(1,35);v[:,0]=.01;a=torch.zeros_like(v);a[:,1]=.002
        return WholeBodyReference(q,v,a)
    def test_each_support_preserves_swing_and_com_and_cancels_contact_rates(self):
        for side in (0,1):
            source=self.source();q=source.q.clone();old=self.m.snapshot(q)
            anchors=old['transforms'][:,self.feet].clone();saved=anchors.clone();contacts=torch.zeros(1,2,dtype=torch.bool);contacts[0,side]=True
            out=anchor_reference(self.f,source,self.names,anchors,contacts)
            state,_,_=self.m.evaluate(out.q,out.v,self.f.integrate_root_position,.001)
            J=torch.cat((state['origin_jac'][:,self.feet],state['angular_jac'][:,self.feet]),2)
            b=torch.cat((state['origin_jac_bias'][:,self.feet],state['angular_jac_bias'][:,self.feet]),2)
            torch.testing.assert_close((J@out.v[:,None,:,None]).squeeze(-1)[contacts],torch.zeros(1,6),atol=1e-5,rtol=0)
            torch.testing.assert_close(((J@out.a[:,None,:,None]).squeeze(-1)+b)[contacts],torch.zeros(1,6),atol=1e-5,rtol=0)
            torch.testing.assert_close(state['transforms'][:,self.feet,:3],anchors[:,:,:3],atol=2e-6,rtol=0)
            torch.testing.assert_close(state['com'],old['com'],atol=2e-6,rtol=0)
            torch.testing.assert_close(source.q,q);torch.testing.assert_close(anchors,saved)
    def test_empty_contacts_are_rejected(self):
        source=self.source();anchors=self.m.snapshot(source.q)['transforms'][:,self.feet]
        with self.assertRaises(ValueError):anchor_reference(self.f,source,self.names,anchors,torch.zeros(1,2,dtype=torch.bool))

    def test_constant_com_offset_preserves_feet_and_com_derivatives(self):
        source=self.source();old,_,_=self.m.evaluate(source.q,source.v,self.f.integrate_root_position,.001)
        anchors=old['transforms'][:,self.feet].clone();contacts=torch.tensor([[True,False]])
        offset=source.q.new_tensor([[.002,.003,-.004]])
        out=anchor_reference(self.f,source,self.names,anchors,contacts,com_offset=offset)
        state,_,_=self.m.evaluate(out.q,out.v,self.f.integrate_root_position,.001)
        torch.testing.assert_close(state['com'],old['com']+offset,atol=2e-6,rtol=0)
        torch.testing.assert_close(state['transforms'][:,self.feet,:3],anchors[:,:,:3],atol=2e-6,rtol=0)
        torch.testing.assert_close((state['com_jac']@out.v[...,None]).squeeze(-1),
            (old['com_jac']@source.v[...,None]).squeeze(-1),atol=1e-5,rtol=0)
        torch.testing.assert_close((state['com_jac']@out.a[...,None]).squeeze(-1)+state['com_jac_bias'],
            (old['com_jac']@source.a[...,None]).squeeze(-1)+old['com_jac_bias'],atol=1e-5,rtol=0)
        for bad in (source.q.new_tensor([[.2,0.,0.]]),source.q.new_tensor([[float('nan'),0.,0.]])):
            with self.assertRaises(ValueError):anchor_reference(self.f,source,self.names,anchors,contacts,com_offset=bad)

    def test_held_body_preserves_source_task_targets_and_is_stationary(self):
        source=self.source();source.v[:,3]=.005;source.a[:,4]=.001
        old,_,_=self.m.evaluate(source.q,source.v,self.f.integrate_root_position,.001)
        anchors=old['transforms'][:,self.feet].clone();contacts=torch.tensor([[True,False]])
        held=source.q[:,19:].clone();held[:,3]+=.01
        pelvis=source.q[:,3:7].clone()
        out=anchor_reference(self.f,source,self.names,anchors,contacts,held_upper=held,held_pelvis=pelvis)
        state,_,_=self.m.evaluate(out.q,out.v,self.f.integrate_root_position,.001)
        torch.testing.assert_close(out.q[:,19:],held);torch.testing.assert_close(out.q[:,3:7],pelvis)
        for value in (out.v[:,3:6],out.a[:,3:6],out.v[:,18:],out.a[:,18:]):
            torch.testing.assert_close(value,torch.zeros_like(value))
        torch.testing.assert_close(state['com'],old['com'],atol=2e-6,rtol=0)
        for key in ('origin_jac','angular_jac'):
            new_v=(state[key][:,self.feet[1]]@out.v[...,None]).squeeze(-1)
            old_v=(old[key][:,self.feet[1]]@source.v[...,None]).squeeze(-1)
            torch.testing.assert_close(new_v,old_v,atol=1e-5,rtol=0)
        with self.assertRaises(ValueError):
            anchor_reference(self.f,source,self.names,anchors,contacts,held_pelvis=pelvis*2)
