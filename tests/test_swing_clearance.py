import unittest

import torch
import warp as wp

from fabrics_sim.fabrics.dynamics_aware import rotation_error
from fabrics_sim.reference.swing_clearance import swing_clearance_reference, feasible_swing_clearance_reference
from test_g1_two_tasks import G1_JOINT_NAMES


class SwingClearanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()
        from fabrics_sim.fabrics.g1_pose_fabric import G1PoseFabric
        cls.fabric = G1PoseFabric(1, 'cpu', robot_name='g1_29dof_rev_1_0',
                                  task_profile='dynamics-aware')

    def test_raise_swing_preserving_stance_pelvis_orientation_and_endpoints(self):
        q = torch.zeros(5,36)
        q[:,2],q[:,6] = .8,1.
        for side in ('left','right'):
            for name,value in (('hip_pitch',-.3),('knee',.6),('ankle_pitch',-.3)):
                q[:,7+G1_JOINT_NAMES.index(side+'_'+name+'_joint')] = value
        original = q.clone()
        events = [dict(takeoff_frame=0,end_frame=4,support=0)]
        out = swing_clearance_reference(self.fabric,q,G1_JOINT_NAMES,events,.04)
        torch.testing.assert_close(q,original,rtol=0,atol=0)
        torch.testing.assert_close(out[[0,-1]],q[[0,-1]],rtol=0,atol=0)
        torch.testing.assert_close(out[:,:13],q[:,:13],rtol=0,atol=0)
        torch.testing.assert_close(out[:,19:],q[:,19:],rtol=0,atol=0)
        model = self.fabric.dynamics_tasks.model
        foot = model.kinematics.get_link_index('right_ankle_roll_link')
        source = model.snapshot(q[2:3]);raised = model.snapshot(out[2:3])
        delta = raised['transforms'][:,foot,:3]-source['transforms'][:,foot,:3]
        torch.testing.assert_close(delta,torch.tensor([[0.,0.,.04]]),rtol=0,atol=2e-5)
        self.assertLess(float(rotation_error(raised['rotations'][:,foot],source['rotations'][:,foot]).abs().max()),2e-5)

    def test_native_midpoint_clearance_only_fills_the_deficit(self):
        q = torch.zeros(5,36);q[:,2],q[:,6] = .8,1.
        for side in ('left','right'):
            for name,value in (('hip_pitch',-.3),('knee',.6),('ankle_pitch',-.3)):
                q[:,7+G1_JOINT_NAMES.index(side+'_'+name+'_joint')] = value
        model = self.fabric.dynamics_tasks.model
        foot = model.kinematics.get_link_index('right_ankle_roll_link')
        z = float(model.snapshot(q[2:3])['transforms'][0,foot,2])
        geometry = [([[0.,0.,0.]],[z-.01])]*2
        out = swing_clearance_reference(self.fabric,q,G1_JOINT_NAMES,
            [dict(takeoff_frame=0,end_frame=4,support=0)],.04,collision_geometry=geometry)
        raised_z = float(model.snapshot(out[2:3])['transforms'][0,foot,2])
        self.assertAlmostEqual(raised_z-z,.03,delta=2e-5)
        # A source that already clears the midpoint target stays unchanged.
        out = swing_clearance_reference(self.fabric,q,G1_JOINT_NAMES,
            [dict(takeoff_frame=0,end_frame=4,support=0)],.01,collision_geometry=geometry)
        torch.testing.assert_close(out,q,rtol=0,atol=0)

    def test_infeasible_lift_is_reduced_without_changing_endpoints_or_stance(self):
        q = torch.zeros(5,36);q[:,2],q[:,6] = .8,1.
        for side in ('left','right'):
            for name,value in (('hip_pitch',-.3),('knee',.6),('ankle_pitch',-.3)):
                q[:,7+G1_JOINT_NAMES.index(side+'_'+name+'_joint')] = value
        events = [dict(takeoff_frame=0,end_frame=4,support=0)]
        geometry = [([[0.,0.,0.]],[0.])]*2
        out,report = feasible_swing_clearance_reference(self.fabric,q,G1_JOINT_NAMES,
            events,1.2,collision_geometry=geometry)
        self.assertTrue(report[0]['target_reduced'])
        self.assertGreater(report[0]['feasible_target_m'],0.)
        torch.testing.assert_close(out[[0,-1]],q[[0,-1]],rtol=0,atol=0)
        torch.testing.assert_close(out[:,:13],q[:,:13],rtol=0,atol=0)
        torch.testing.assert_close(out[:,19:],q[:,19:],rtol=0,atol=0)
        expected = swing_clearance_reference(self.fabric,q,G1_JOINT_NAMES,events,
            report[0]['feasible_target_m'],collision_geometry=geometry)
        torch.testing.assert_close(out,expected,rtol=0,atol=0)

    def test_partial_initial_swing_keeps_spawn_reference(self):
        q = torch.zeros(5,36);q[:,2],q[:,6] = .8,1.
        out = swing_clearance_reference(self.fabric,q,G1_JOINT_NAMES,
            [dict(takeoff_frame=0,end_frame=4,support=0,starts_mid_swing=True)],.04)
        torch.testing.assert_close(out,q,rtol=0,atol=0)


if __name__ == '__main__':
    unittest.main()
