import unittest
from types import SimpleNamespace
import torch
from g1_com_velocity_soft import COMVelocitySoftConstraint
from fabrics_sim.solver.admm_qp import QPData
from fabrics_sim.solver.daqp_backend import DAQPBackend


class COMSoftTest(unittest.TestCase):
    def problem(self, fixed=False):
        n=35;dtype=torch.float64
        H=torch.eye(n,dtype=dtype)[None]
        z=torch.zeros(1,n,dtype=dtype)
        low=torch.full_like(z,-100.);high=-low
        if fixed:low[:,:3]=0;high[:,:3]=0
        return QPData(H,z.clone(),torch.zeros(1,0,n,dtype=dtype),
                      torch.zeros(1,0,dtype=dtype),low,high)

    def prepare(self, error=1., correction=0., **options):
        soft=COMVelocitySoftConstraint(DAQPBackend(),onset=0.,width=.1,**options)
        A=torch.eye(35,dtype=torch.float64)[None,:3]
        zero=torch.zeros(1,3,dtype=torch.float64)
        request=SimpleNamespace(A=A,bias=zero,metric=torch.eye(3,dtype=torch.float64)[None])
        soft.prepare(request,torch.full_like(zero,error),zero,zero,
                     torch.full((1,35),correction,dtype=torch.float64))
        return soft

    def test_analytic_optimum_and_nonnegative_slack(self):
        soft=self.prepare();out=soft.solve(self.problem(),workspace='priority')
        self.assertTrue(out.status_ok.all())
        # min .5*a^2 + .5*100*s^2, s >= 1+.1*a-.82
        torch.testing.assert_close(out.z[0,:3],torch.full((3,),-.9,dtype=torch.float64),atol=1e-6,rtol=0)
        torch.testing.assert_close(soft.slack,torch.full((1,3),.09,dtype=torch.float64),atol=1e-6,rtol=0)

    def test_slack_keeps_unachievable_target_feasible(self):
        soft=self.prepare();out=soft.solve(self.problem(fixed=True),workspace='priority')
        self.assertTrue(out.status_ok.all())
        torch.testing.assert_close(out.z[:,:3],torch.zeros(1,3,dtype=torch.float64))
        torch.testing.assert_close(soft.slack,torch.full((1,3),.18,dtype=torch.float64),atol=1e-6,rtol=0)

    def test_inside_band_has_no_attraction_to_zero(self):
        soft=self.prepare(error=.05)
        out=soft.solve(self.problem(),workspace='priority')
        torch.testing.assert_close(out.z,torch.zeros_like(out.z),atol=1e-6,rtol=0)

    def test_preserves_hard_constraints_and_physical_correction(self):
        soft=self.prepare(correction=2.);qp=self.problem(fixed=True)
        aug=soft.augment(qp)
        torch.testing.assert_close(aug.lb[:,:35],qp.lb)
        torch.testing.assert_close(aug.ub[:,:35],qp.ub)
        torch.testing.assert_close(soft.command_offset,torch.full((1,3),.8,dtype=torch.float64))
        self.assertTrue(torch.isinf(aug.ub[:,35:]).all())

    def test_inactive_and_tracking_stage_are_unchanged(self):
        for error,stage in ((0.,'priority'),(1.,'tracking')):
            soft=self.prepare(error=error);qp=self.problem()
            out=soft.solve(qp,workspace=stage)
            torch.testing.assert_close(out.z,torch.zeros_like(out.z))
            self.assertIsNone(soft.last_augmented_problem)

    def test_negative_error_is_symmetric(self):
        positive=self.prepare();negative=self.prepare(error=-1.)
        a=positive.solve(self.problem(),workspace='priority')
        b=negative.solve(self.problem(),workspace='priority')
        torch.testing.assert_close(a.z,-b.z)

    def test_runtime_mixed_objective_and_constraint_precision(self):
        soft=self.prepare();qp=self.problem()
        # Runtime can accumulate constraints in float64 and retain float32 h.
        qp.h=qp.h.float()
        out=soft.solve(qp,workspace='priority')
        self.assertTrue(out.status_ok.all())
        torch.testing.assert_close(out.z[0,:3].double(),torch.full((3,),-.9,dtype=torch.float64),
                                   atol=1e-6,rtol=0)

    def test_invalid_parameters(self):
        for options in ({'horizon':0},{'weight':-1},{'contraction':1},{'allowance':-1}):
            with self.assertRaises(ValueError):COMVelocitySoftConstraint(None,**options)


if __name__=='__main__':unittest.main()
