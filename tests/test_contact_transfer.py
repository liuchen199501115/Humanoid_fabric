import unittest
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import torch
import warp as wp

from fabrics_sim.reference.contact_transfer import ContactTransferReference
from fabrics_sim.reference.whole_body import WholeBodyReference


class ContactTransferTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.config.kernel_cache_dir = '/tmp/g1_warp_cache'
        wp.init()
        from fabrics_sim.fabrics.g1_pose_fabric import G1PoseFabric
        from test_g1_two_tasks import G1_JOINT_NAMES
        cls.fabric = G1PoseFabric(1, 'cpu', robot_name='g1_29dof_rev_1_0', task_profile='dynamics-aware')
        cls.names = G1_JOINT_NAMES

    def planner(self, support=0):
        q = torch.zeros(1, 36); q[:, 2] = .75; q[:, 6] = 1.
        for side in ('left', 'right'):
            for joint, value in [('hip_pitch', -.2), ('knee', .4), ('ankle_pitch', -.2)]:
                q[:, 7+self.names.index(side+'_'+joint+'_joint')] = value
        z = torch.zeros(1, 35)
        ref = WholeBodyReference(q, z, z)
        model = self.fabric.dynamics_tasks.model
        feet = [model.kinematics.get_link_index(s+'_ankle_roll_link') for s in ('left', 'right')]
        poses = model.snapshot(q)['transforms'][:, feet].clone()
        return ContactTransferReference(self.fabric, ref, self.names, poses, support, 0.)

    def test_both_sides_transfer_feet_fixed_and_com_inside_support(self):
        for support in (0, 1):
            p = self.planner(support)
            for t in (0., .5, 1., 1.5, 2.):
                ref = p.sample(t, 3)
                state = ref.apply(self.fabric)
                torch.testing.assert_close(state['transforms'][:, p.feet, :3], p.stance[..., :3], atol=3e-5, rtol=0)
                torch.testing.assert_close((state['origin_jac'][:, p.feet] @ ref.v[:, None, :, None]).squeeze(-1), torch.zeros(1, 2, 3), atol=2e-5, rtol=0)
            torch.testing.assert_close(state['com'], p.end_com, atol=3e-5, rtol=0)
            torch.testing.assert_close(ref.v, torch.zeros_like(ref.v), atol=1e-6, rtol=0)
            ref = p.sample(2., support+1)
            ref = p.sample(3., support+1)
            state = ref.apply(self.fabric)
            self.assertAlmostEqual(float(state['transforms'][0, p.feet[1-support], 2]-p.stance[0, 1-support, 2]), .04, places=4)
            torch.testing.assert_close(ref.v, torch.zeros_like(ref.v), atol=1e-6, rtol=0)
            torch.testing.assert_close(ref.a, torch.zeros_like(ref.a), atol=1e-6, rtol=0)

    def test_recorded_right_support_switch_height_preserves_joint_margins(self):
        data=json.loads((Path(__file__).parent/'fixtures/g1_support_switch.json').read_text())
        q=torch.tensor(data['q']);zero=torch.zeros(1,35)
        ref=WholeBodyReference(q,zero,zero);stance=torch.tensor(data['stance'])
        limits={n.attrib['name']:n.find('limit') for n in ET.parse(self.fabric.urdf_path).getroot().findall('joint')}
        bounds=dict(lower=q.new_tensor([float(limits[n].attrib['lower']) for n in self.names]),
                    upper=q.new_tensor([float(limits[n].attrib['upper']) for n in self.names]),dt=.002,bandwidth=8.)
        plan=ContactTransferReference(self.fabric,ref,self.names,stance,1,0.,preflight_dt=.01,
                                      joint_bounds=bounds,com_height_offset=.01)
        self.assertTrue(plan.preflight['sampled_joint_bounds_passed'])
        self.assertGreaterEqual(plan.preflight['minimum_terminal_leg_margin'],.055)
        self.assertAlmostEqual(float(plan.end_com[0,2]-plan.start_com[0,2]),.01,places=6)
        before=plan.sample(0.,3)
        torch.testing.assert_close(before.q,q,atol=3e-6,rtol=0.)

    def test_recorded_second_switch_lowers_com_and_lifts_right_foot(self):
        from fabrics_sim.reference.support_switch import SupportSwitchReference
        data=json.loads((Path(__file__).parent/'fixtures/g1_second_support_switch.json').read_text())
        q=torch.tensor(data['q']);zero=torch.zeros(1,35);stance=torch.tensor(data['stance'])
        ref=WholeBodyReference(q,zero,zero)
        limits={n.attrib['name']:n.find('limit') for n in ET.parse(self.fabric.urdf_path).getroot().findall('joint')}
        bounds=dict(lower=q.new_tensor([float(limits[n].attrib['lower']) for n in self.names]),
                    upper=q.new_tensor([float(limits[n].attrib['upper']) for n in self.names]),dt=.002,bandwidth=8.)
        switch=SupportSwitchReference(self.fabric,ref,self.names,stance,0,0.,.002,bounds,
                                      com_height_offset=-.02,lift_height=.03)
        plan=switch.transfer
        self.assertTrue(plan.preflight['sampled_joint_bounds_passed'])
        self.assertGreaterEqual(plan.preflight['minimum_terminal_leg_margin'],.055)
        self.assertAlmostEqual(float(plan.end_com[0,2]-plan.start_com[0,2]),-.02,places=6)
        plan.sample(2.,3);plan.sample(2.,1);final=plan.sample(3.,1)
        state=self.fabric.dynamics_tasks.model.snapshot(final.q)
        self.assertAlmostEqual(float(state['transforms'][0,plan.feet[1],2]-stance[0,1,2]),.03,places=4)
        torch.testing.assert_close(final.v,zero,atol=1e-6,rtol=0.)

    def test_recovery_switch_roll_is_smooth_and_preserves_constraints(self):
        from scipy.spatial.transform import Rotation
        import numpy as np
        data=json.loads((Path(__file__).parent/'fixtures/g1_recovery_support_switch.json').read_text())
        q=torch.tensor(data['q']);zero=torch.zeros(1,35);stance=torch.tensor(data['stance'])
        ref=WholeBodyReference(q,zero,zero)
        limits={n.attrib['name']:n.find('limit') for n in ET.parse(self.fabric.urdf_path).getroot().findall('joint')}
        bounds=dict(lower=q.new_tensor([float(limits[n].attrib['lower']) for n in self.names]),
                    upper=q.new_tensor([float(limits[n].attrib['upper']) for n in self.names]),dt=.002,bandwidth=8.)
        with self.assertRaisesRegex(RuntimeError,'No transfer inside'):
            ContactTransferReference(self.fabric,ref,self.names,stance,0,0.,joint_bounds=bounds,
                preflight_dt=.02,com_height_offset=-.02,lift_height=.03)
        plan=ContactTransferReference(self.fabric,ref,self.names,stance,0,0.,joint_bounds=bounds,
            preflight_dt=.002,com_height_offset=-.02,lift_height=.03,pelvis_roll_offset=-.1)
        self.assertTrue(plan.preflight['sampled_joint_bounds_passed'])
        self.assertGreaterEqual(plan.preflight['minimum_terminal_leg_margin'],.055)
        start=plan.sample(0.,3)
        torch.testing.assert_close(start.q,q,atol=3e-6,rtol=0.)
        torch.testing.assert_close(start.v,zero,atol=1e-6,rtol=0.)
        torch.testing.assert_close(start.a,zero,atol=1e-6,rtol=0.)
        left=plan.sample(.799,3);center=plan.sample(.8,3);right=plan.sample(.801,3)
        angular=(Rotation.from_quat(right.q[0,3:7].numpy())*
                 Rotation.from_quat(left.q[0,3:7].numpy()).inv()).as_rotvec()/.002
        estimated=torch.cat(((right.q[:,:3]-left.q[:,:3])/.002,
            q.new_tensor(angular)[None],(right.q[:,7:]-left.q[:,7:])/.002),1)
        torch.testing.assert_close(center.v,estimated,atol=.003,rtol=.03)
        torch.testing.assert_close(center.a,(right.v-left.v)/.002,atol=.005,rtol=.05)
        end=plan.sample(2.,3)
        rotation=(Rotation.from_quat(q[0,3:7].numpy()).inv()*
                  Rotation.from_quat(end.q[0,3:7].numpy())).as_rotvec()
        np.testing.assert_allclose(rotation,[-.1,0.,0.],atol=1e-6)
        plan.sample(2.,1);lift=plan.sample(3.,1)
        state=self.fabric.dynamics_tasks.model.snapshot(lift.q)
        torch.testing.assert_close(state['transforms'][:,plan.feet[0],:3],stance[:,0,:3],atol=3e-6,rtol=0.)
        self.assertAlmostEqual(float(state['transforms'][0,plan.feet[1],2]-stance[0,1,2]),.03,places=5)
        torch.testing.assert_close(lift.v,zero,atol=1e-6,rtol=0.)
        torch.testing.assert_close(lift.a,zero,atol=1e-6,rtol=0.)
        torch.testing.assert_close(ref.q,q)
        with self.assertRaisesRegex(ValueError,'Pelvis roll'):
            ContactTransferReference(self.fabric,ref,self.names,stance,0,0.,pelvis_roll_offset=float('nan'))

    def test_reference_derivatives_follow_whole_body_path(self):
        p = self.planner()
        left = p.sample(.999, 3)
        center = p.sample(1., 3)
        right = p.sample(1.001, 3)
        estimated_v = torch.cat(((right.q[:, :3]-left.q[:, :3])/.002,
                                 torch.zeros(1, 3), (right.q[:, 7:]-left.q[:, 7:])/.002), 1)
        torch.testing.assert_close(center.v, estimated_v, atol=.003, rtol=.03)
        torch.testing.assert_close(center.a, (right.v-left.v)/.002, atol=.005, rtol=.05)

    def test_recorded_start_selects_margin_without_changing_limits(self):
        data = json.loads((Path(__file__).parent/'fixtures/g1_transfer_joint_margin.json').read_text())
        q = torch.tensor(data['q']); zero = torch.zeros(1, 35)
        reference = WholeBodyReference(q, zero, zero)
        stance = torch.tensor(data['stance'])
        limits = {n.attrib['name']: n.find('limit') for n in ET.parse(self.fabric.urdf_path).getroot().findall('joint')}
        lower = q.new_tensor([float(limits[n].attrib['lower']) for n in self.names])
        upper = q.new_tensor([float(limits[n].attrib['upper']) for n in self.names])
        bounds = dict(lower=lower, upper=upper, dt=.002, bandwidth=8.)
        with self.assertRaisesRegex(RuntimeError, 'joint braking/limit margins'):
            ContactTransferReference(self.fabric, reference, self.names, stance, 0, 0.,
                                     support_lateral_offset=0., joint_bounds=bounds)
        plan = ContactTransferReference(self.fabric, reference, self.names, stance, 0, 0.,
                                        preflight_dt=.01, joint_bounds=bounds)
        self.assertTrue(plan.preflight['sampled_joint_bounds_passed'])
        self.assertGreaterEqual(plan.preflight['minimum_terminal_leg_margin'], .055)
        self.assertLess(abs(plan.support_lateral_offset), .02)
        self.assertEqual(plan.preflight['samples'], 302)
        self.assertIsNone(plan.release_time)
        torch.testing.assert_close(plan.q, q)
        torch.testing.assert_close(reference.q, q)
        plan.sample(2., 3)
        plan.sample(2., 1)
        final = plan.sample(3., 1)
        torch.testing.assert_close(final.q[:, 19:], q[:, 19:])

    def test_release_requires_actual_tracking_of_the_transfer(self):
        p = self.planner()
        ref = p.sample(2., 3)
        zero = torch.zeros_like(p.end_com)
        self.assertTrue(p.ready_for_release(2., p.end_com, zero, ref.q))
        self.assertFalse(p.ready_for_release(1.9, p.end_com, zero, ref.q))
        self.assertFalse(p.ready_for_release(2., p.end_com+.03, zero, ref.q))
        self.assertFalse(p.ready_for_release(2., p.end_com, zero+.05, ref.q))
        displaced = ref.q.clone(); displaced[:, 7] += .1
        self.assertFalse(p.ready_for_release(2., p.end_com, zero, displaced))
        with self.assertRaises(ValueError):
            p.ready_for_release(2., p.end_com*float('nan'), zero, ref.q)

    def test_no_early_release(self):
        p = self.planner()
        with self.assertRaisesRegex(RuntimeError, 'before'):
            p.sample(1., 1)

    def test_source_and_upper_body_are_preserved(self):
        p = self.planner()
        initial = p.q.clone()
        for time in (0., .5, 1., 1.5, 2.):
            ref = p.sample(time, 3)
            torch.testing.assert_close(ref.q[:, 3:7], initial[:, 3:7])
            torch.testing.assert_close(ref.q[:, 19:], initial[:, 19:])
        with self.assertRaisesRegex(RuntimeError, 'Unexpected support'):
            p.sample(3., 2)


if __name__ == '__main__':
    unittest.main()
