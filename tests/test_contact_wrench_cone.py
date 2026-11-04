import unittest
import math
import torch
from fabrics_sim.solver.AccelerationQP import build_single_support_wrench_constraints


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


if __name__=='__main__':unittest.main()
