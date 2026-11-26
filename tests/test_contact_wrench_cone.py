import unittest
import math
import torch
from fabrics_sim.solver.AccelerationQP import (
    build_single_support_wrench_constraints, split_foot_wrench_constraints,
)


class ContactWrenchConeTests(unittest.TestCase):
    def constraints(self, batch=1, rotation=None, coupled=True):
        if rotation is None:rotation=torch.eye(3).expand(batch,2,3,3)
        return build_single_support_wrench_constraints(
            contact_mode=torch.ones(batch,dtype=torch.long),foot_rotation_world=rotation,
            toe_length=.12,heel_length=.05,half_width=.025,sole_depth=.035,coupled_yaw=coupled)

    def test_rejects_yaw_at_saturated_corner_cop(self):
        # All normal load at one corner leaves no independent yaw capacity.
        wrench=torch.tensor([[0.,0.,100.,2.5,-12.,3.,0.,0.,0.,0.,0.,0.]])
        old,b=self.constraints(coupled=False)
        self.assertLessEqual(float((old@wrench[...,None]).squeeze(-1).sub(b).max()),1e-6)
        cone,b=self.constraints()
        self.assertGreater(float((cone@wrench[...,None]).squeeze(-1).sub(b).max()),1.)

    def test_accepts_wrenches_generated_by_physical_corner_forces(self):
        torch.manual_seed(17)
        n=128
        points=torch.tensor([[-.05,-.025,-.035],[-.05,.025,-.035],[.12,-.025,-.035],[.12,.025,-.035]])
        force=torch.zeros(n,4,3)
        force[:,:,2]=20+60*torch.rand(n,4)
        force[:,:,:2]=(2*torch.rand(n,4,2)-1)*(.6/math.sqrt(2))*force[:,:,2:]
        moment=torch.linalg.cross(points.expand_as(force),force).sum(1)
        f=force.sum(1)
        theta=torch.linspace(-3.,3.,n)
        rotation=torch.eye(3).expand(n,2,3,3).clone()
        rotation[:,:,0,0]=theta.cos()[:,None];rotation[:,:,1,1]=theta.cos()[:,None]
        rotation[:,:,0,1]=-theta.sin()[:,None];rotation[:,:,1,0]=theta.sin()[:,None]
        world_f=(rotation[:,0]@f[...,None]).squeeze(-1)
        world_m=(rotation[:,0]@moment[...,None]).squeeze(-1)
        wrench=torch.cat((world_f,world_m,torch.zeros(n,6)),1)
        cone,b=self.constraints(n,rotation)
        self.assertEqual(cone.shape,(n,40,12))
        self.assertLessEqual(float((cone@wrench[...,None]).squeeze(-1).sub(b).max()),3e-5)




class FootWrenchBlockTests(unittest.TestCase):
    def test_both_cone_sizes_preserve_each_foot_block(self):
        rotations=torch.eye(3,dtype=torch.float64).expand(4,2,3,3)
        for coupled,rows in [(False,12),(True,20)]:
            with self.subTest(coupled_yaw=coupled):
                G,b=build_single_support_wrench_constraints(
                    contact_mode=torch.tensor([0,1,2,3]),foot_rotation_world=rotations,
                    coupled_yaw=coupled)
                blocks=split_foot_wrench_constraints(G,b)
                self.assertEqual([block[0].shape[1] for block in blocks],[rows,rows])
                self.assertEqual(int(blocks[0][0][:,:,6:].count_nonzero()),0)
                self.assertEqual(int(blocks[1][0][:,:,:6].count_nonzero()),0)
                torch.testing.assert_close(torch.cat([block[0] for block in blocks],1),G)
                torch.testing.assert_close(torch.cat([block[1] for block in blocks],1),b)

    def test_right_point_patch_replaces_the_entire_tilted_sole_cone(self):
        rotations=torch.eye(3,dtype=torch.float64).repeat(1,2,1,1)
        angle=.7;c,s=math.cos(angle),math.sin(angle)
        rotations[0,1]=torch.tensor([[1.,0.,0.],[0.,c,-s],[0.,s,c]],dtype=torch.float64)
        mode=torch.tensor([2])
        G,b=build_single_support_wrench_constraints(contact_mode=mode,foot_rotation_world=rotations)
        point_G,point_b=build_single_support_wrench_constraints(
            contact_mode=mode,foot_rotation_world=torch.eye(3,dtype=torch.float64).expand(1,2,3,3),
            toe_length=0.,heel_length=0.,half_width=0.,coupled_yaw=True)
        blocks=split_foot_wrench_constraints(G,b)
        blocks[1]=split_foot_wrench_constraints(point_G,point_b)[1]
        combined_G=torch.cat([block[0] for block in blocks],1)
        combined_b=torch.cat([block[1] for block in blocks],1)
        # A vertical force at the horizontal observed point is valid. Keeping
        # any tilted ankle-frame friction rows would reject this same force.
        wrench=torch.zeros(1,12,1,dtype=torch.float64);wrench[0,8]=300.
        self.assertLessEqual(float((combined_G@wrench).squeeze(-1).sub(combined_b).max()),1e-10)
        self.assertGreater(float((G@wrench).squeeze(-1).sub(b).max()),1.)
        self.assertEqual(combined_G.shape[1],12+20)

    def test_left_patch_keeps_all_right_friction_and_load_limits(self):
        rotations=torch.eye(3,dtype=torch.float64).expand(1,2,3,3)
        for coupled in [False,True]:
            with self.subTest(coupled_yaw=coupled):
                G,b=build_single_support_wrench_constraints(
                    contact_mode=torch.tensor([3]),foot_rotation_world=rotations,coupled_yaw=coupled)
                point_G,point_b=build_single_support_wrench_constraints(
                    contact_mode=torch.tensor([3]),foot_rotation_world=rotations,
                    toe_length=0.,heel_length=0.,half_width=0.,coupled_yaw=True)
                blocks=split_foot_wrench_constraints(G,b)
                blocks[0]=split_foot_wrench_constraints(point_G,point_b)[0]
                combined_G=torch.cat([block[0] for block in blocks],1)
                combined_b=torch.cat([block[1] for block in blocks],1)
                valid=torch.zeros(1,12,1,dtype=torch.float64);valid[0,2]=200.;valid[0,8]=100.
                self.assertLessEqual(float((combined_G@valid).squeeze(-1).sub(combined_b).max()),1e-10)
                for axis,value in [(6,300.),(8,900.)]:
                    invalid=valid.clone();invalid[0,axis]=value
                    self.assertGreater(float((combined_G@invalid).squeeze(-1).sub(combined_b).max()),1.)

    def test_rejects_incomplete_or_mismatched_blocks(self):
        for G,b in [(torch.zeros(1,23,12),torch.zeros(1,23)),
                    (torch.zeros(1,24,6),torch.zeros(1,24)),
                    (torch.zeros(1,24,12),torch.zeros(1,23))]:
            with self.assertRaises(ValueError):split_foot_wrench_constraints(G,b)


if __name__ == '__main__':
    unittest.main()
