import unittest

import torch
import warp as wp

from fabrics_sim.reference.initial_stance import flat_foot_spawn, flat_foot_reference, contact_pose_reference


class InitialStanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.config.kernel_cache_dir='/tmp/g1_warp_cache'
        wp.init()
        from fabrics_sim.fabrics.g1_pose_fabric import G1PoseFabric
        from test_g1_two_tasks import G1_JOINT_NAMES
        cls.names=G1_JOINT_NAMES
        cls.fabric=G1PoseFabric(1,'cpu',robot_name='g1_29dof_rev_1_0',task_profile='dynamics-aware')

    def test_explicit_com_height_with_reachable_contact_geometry(self):
        q=torch.zeros(1,36);q[:,2]=.75;q[:,6]=1.
        for side in ('left','right'):
            for joint,angle in [('hip_pitch',-.3),('knee',.6),('ankle_pitch',-.3)]:
                q[:,7+self.names.index(side+'_'+joint+'_joint')]=angle
        model=self.fabric.dynamics_tasks.model
        indices=[model.kinematics.get_link_index(s+'_ankle_roll_link') for s in ('left','right')]
        goal=model.snapshot(q);poses=goal['transforms'][:,indices].clone();com=goal['com'].clone()
        source=q.clone();source[:,2]+=.01;source[:,7+self.names.index('left_knee_joint')]+=.04
        before=source.clone()
        fitted=contact_pose_reference(self.fabric,source,self.names,poses,com[:,:2],float(com[0,2]))
        state=model.snapshot(fitted)
        torch.testing.assert_close(state['com'],com,atol=2e-5,rtol=0)
        torch.testing.assert_close(state['transforms'][:,indices,:3],poses[:,:,:3],atol=2e-5,rtol=0)
        torch.testing.assert_close(state['rotations'][:,indices],goal['rotations'][:,indices],atol=2e-5,rtol=0)
        torch.testing.assert_close(source,before)
        with self.assertRaises(ValueError):
            contact_pose_reference(self.fabric,source,self.names,poses,com[:,:2],float('nan'))

    def test_explicit_contact_pose_keeps_reachable_rolled_foot_geometry(self):
        q=torch.zeros(1,36);q[:,2]=.75;q[:,6]=1.
        for side in ('left','right'):
            for joint,angle in [('hip_pitch',-.3),('knee',.6),('ankle_pitch',-.3)]:
                q[:,7+self.names.index(side+'_'+joint+'_joint')]=angle
        q[:,7+self.names.index('left_ankle_pitch_joint')]=.1
        original=q.clone();model=self.fabric.dynamics_tasks.model
        feet=[model.kinematics.get_link_index(s+'_ankle_roll_link') for s in ('left','right')]
        goal=model.snapshot(q);poses=goal['transforms'][:,feet].clone()
        result=contact_pose_reference(self.fabric,q,self.names,poses,goal['com'][:,:2])
        torch.testing.assert_close(result,q,atol=2e-6,rtol=0)
        actual=model.snapshot(result)
        torch.testing.assert_close(q,original)
        torch.testing.assert_close(result[:,3:7],q[:,3:7])
        torch.testing.assert_close(result[:,19:],q[:,19:])
        torch.testing.assert_close(actual['transforms'][:,feet,:3],poses[:,:,:3],atol=2e-5,rtol=0)
        torch.testing.assert_close(actual['rotations'][:,feet],goal['rotations'][:,feet],atol=2e-5,rtol=0)
        self.assertGreater(abs(actual['rotations'][0,feet[0],2,0]).item(),.2)

    def test_nearby_level_feet_without_modifying_clip_or_upper_body(self):
        q=torch.zeros(1,36);q[:,2]=.8;q[:,6]=1.
        for side in ('left','right'):
            for joint,angle in [('hip_pitch',-.1),('knee',.2),('ankle_pitch',-.1)]:
                q[:,7+self.names.index(side+'_'+joint+'_joint')]=angle
        q[:,7+self.names.index('left_ankle_roll_joint')]=.04
        q[:,7+self.names.index('right_ankle_roll_joint')]=-.03
        original=q.clone()
        model=self.fabric.dynamics_tasks.model
        indices=[model.kinematics.get_link_index(s+'_ankle_roll_link') for s in ('left','right')]
        before=model.snapshot(q)['transforms'][:,indices,:2].clone()
        result=flat_foot_spawn(self.fabric,q,self.names)
        state=model.snapshot(result)
        torch.testing.assert_close(q,original)
        torch.testing.assert_close(result[:,:2],q[:,:2])
        torch.testing.assert_close(result[:,3:7],q[:,3:7])
        torch.testing.assert_close(result[:,19:],q[:,19:])
        torch.testing.assert_close(state['transforms'][:,indices,:2],before,atol=2e-5,rtol=0)
        torch.testing.assert_close(state['transforms'][:,indices,2],torch.full((1,2),.035),atol=2e-5,rtol=0)
        torch.testing.assert_close(state['rotations'][:,indices,:,2],
                                  torch.tensor([[[0.,0.,1.],[0.,0.,1.]]]),atol=2e-5,rtol=0)

    def test_rolling_reference_is_not_subject_to_spawn_displacement_guard(self):
        q=torch.zeros(1,36);q[:,2]=.8;q[:,6]=1.
        q[:,7+self.names.index('left_ankle_pitch_joint')]=.5
        before=q.clone()
        result=flat_foot_reference(self.fabric,q,self.names)
        torch.testing.assert_close(q,before)
        self.assertGreater((result[:,7:]-q[:,7:]).abs().max().item(), .35)
        model=self.fabric.dynamics_tasks.model
        indices=[model.kinematics.get_link_index(s+'_ankle_roll_link') for s in ('left','right')]
        source_feet=model.snapshot(q)['transforms'][:,indices,:2].clone()
        state=model.snapshot(result)
        torch.testing.assert_close(state['transforms'][:,indices,:2],source_feet,atol=2e-5,rtol=0)
        torch.testing.assert_close(state['transforms'][:,indices,2],torch.full((1,2),.035),atol=2e-5,rtol=0)
        torch.testing.assert_close(state['rotations'][:,indices,:,2],
                                  torch.tensor([[[0.,0.,1.],[0.,0.,1.]]]),atol=2e-5,rtol=0)
        with self.assertRaisesRegex(RuntimeError,'too far'):
            flat_foot_spawn(self.fabric,q,self.names)

    def test_landing_height_uses_actual_support_and_requested_com(self):
        q=torch.zeros(1,36);q[:,2]=.8;q[:,6]=1.
        model=self.fabric.dynamics_tasks.model
        indices=[model.kinematics.get_link_index(s+'_ankle_roll_link') for s in ('left','right')]
        feet=model.snapshot(q)['transforms'][:,indices].clone()
        feet[:,0,0]+=.12
        feet[:,1,0]-=.08
        target=feet[:,:,:2].mean(1)
        result=flat_foot_reference(self.fabric,q,self.names,foot_poses=feet,com_xy=target)
        state=model.snapshot(result)
        torch.testing.assert_close(state['transforms'][:,indices,:2],feet[:,:,:2],atol=2e-5,rtol=0)
        torch.testing.assert_close(state['com'][:,:2],target,atol=2e-5,rtol=0)
        torch.testing.assert_close(result[:,3:7],q[:,3:7])
        torch.testing.assert_close(result[:,19:],q[:,19:])
        with self.assertRaises(ValueError):
            flat_foot_reference(self.fabric,q,self.names,foot_poses=feet)

    def test_long_landing_with_fixed_old_foot_and_source_com_offset(self):
        # walk1 frame 155: the planted foot is behind its sliding source pose.
        # Preserve the original new landing and upper body while fitting the
        # fore-aft COM offset and a joint-limited flat-foot configuration.
        from fabrics_sim.reference.footstep import source_com_landing_target
        q=torch.tensor([[1.55931914,3.69815540,.74539900,-.02074451,-.02876036,-.71510834,.69811362,
            -.46258470,-.07213661,.09171579,.26615846,.25947359,.05421668,
            .48247468,.07072023,-.04957330,.22927678,-.11865591,.00039359,
            .20570463,-.00578209,.25671685,.00528966,.33250538,-.64172214,
            1.05462956,.33319014,.06226958,.43987116,-.04233506,-.27537894,
            .64917690,.95990866,-.18488239,.11258462,-.49072915]])
        model=self.fabric.dynamics_tasks.model
        indices=[model.kinematics.get_link_index(s+'_ankle_roll_link') for s in ('left','right')]
        source_state=model.snapshot(q)
        source_feet=source_state['transforms'][:,indices].detach().clone()
        feet=source_feet.clone()
        feet[:,1]=q.new_tensor([1.50981975,4.08987522,.035,0.,0.,-.74498248,.66708416])
        target=q.new_tensor(source_com_landing_target(source_state['com'][0].detach().numpy(),
            source_feet[0].numpy(),feet[0,1].numpy(),feet[0,0].numpy()))[None]
        result=flat_foot_reference(self.fabric,q,self.names,foot_poses=feet,com_xy=target)
        state=model.snapshot(result)
        torch.testing.assert_close(state['transforms'][:,indices,:2],feet[:,:,:2],atol=2e-5,rtol=0)
        torch.testing.assert_close(state['transforms'][:,indices,2],torch.full((1,2),.035),atol=2e-5,rtol=0)
        torch.testing.assert_close(state['com'][:,:2],target,atol=2e-5,rtol=0)
        torch.testing.assert_close(result[:,19:],q[:,19:])
        self.assertLess((result[:,7:]-q[:,7:]).square().mean().sqrt().item(),.2)


if __name__=='__main__':unittest.main()
