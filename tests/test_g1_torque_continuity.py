"""Independent analytic optimum and unchanged constraints for effort continuity."""
import sys
import unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'examples'))
from g1_torque_continuity import TorqueContinuitySolver, stance_torque_weights
from fabrics_sim.solver.admm_qp import QPData


class AnalyticBackend:
    last_status = ['analytic']
    def solve(self, qp, **kwargs):
        self.kwargs=kwargs
        return torch.linalg.solve(qp.H, -qp.h[..., None]).squeeze(-1)


class TorqueContinuityTest(unittest.TestCase):
    def problem(self):
        return QPData(torch.eye(76,dtype=torch.float64)[None],
            -torch.arange(76,dtype=torch.float64)[None],
            torch.zeros(1,2,76),torch.ones(1,2),torch.full((1,76),-100.),torch.full((1,76),100.),
            torch.zeros(1,1,76),torch.zeros(1,1))

    def test_analytic_optimum_and_exact_constraint_preservation(self):
        backend=AnalyticBackend();solver=TorqueContinuitySolver(backend);qp=self.problem()
        constraints=[qp.G,qp.b,qp.lb,qp.ub,qp.A_eq,qp.b_eq]
        previous=torch.full((1,29),5.,dtype=torch.float64);solver.update(previous,3.)
        previous.fill_(100.)  # Caller mutation must not change the reference.
        result=solver.solve(qp,workspace='priority',warm_z=None)
        torch.testing.assert_close(result[:,:47],torch.arange(47,dtype=torch.float64)[None])
        torch.testing.assert_close(result[:,47:],(torch.arange(47,76,dtype=torch.float64)[None]+15.)/4.)
        for old,new in zip(constraints,[qp.G,qp.b,qp.lb,qp.ub,qp.A_eq,qp.b_eq]):self.assertIs(old,new)
        self.assertEqual(backend.kwargs['workspace'],'priority');self.assertEqual(solver.last_status,['analytic'])
        torch.testing.assert_close(qp.H[0,47:,47:],4.*torch.eye(29,dtype=torch.float64))

    def test_retry_does_not_accumulate_and_zero_restores_objective(self):
        solver=TorqueContinuitySolver(AnalyticBackend());qp=self.problem()
        solver.update(torch.zeros(1,29),2.);first=solver.solve(qp)
        torch.testing.assert_close(first,solver.solve(qp))
        solver.update(torch.zeros(1,29),0.);restored=solver.solve(qp)
        torch.testing.assert_close(restored,torch.arange(76,dtype=torch.float64)[None])

    def test_each_stage_receives_same_previous_physical_torque(self):
        solver=TorqueContinuitySolver(AnalyticBackend());solver.update(torch.ones(1,29),5.)
        a,b=self.problem(),self.problem();x=solver.solve(a,workspace='priority');y=solver.solve(b,workspace='tracking')
        torch.testing.assert_close(x,y);self.assertIsNot(a.H,b.H)

    def test_invalid_previous_and_weight_rejected(self):
        solver=TorqueContinuitySolver(AnalyticBackend())
        for value in (-1.,float('nan'),float('inf')):
            with self.assertRaises(ValueError):solver.update(torch.zeros(1,29),value)
        for previous in (torch.zeros(29),torch.zeros(1,28),torch.full((1,29),float('nan'))):
            with self.assertRaises(ValueError):solver.update(previous,1.)

    def test_passive_stop_variables_are_not_motor_commands(self):
        from torch.nn.functional import pad
        qp=self.problem()
        qp.H=pad(qp.H,(0,1,0,1));qp.H[:,76,76]=2.
        qp.h=pad(qp.h,(0,1));qp.h[:,76]=-14.
        qp.G=pad(qp.G,(0,1));qp.A_eq=pad(qp.A_eq,(0,1))
        qp.lb=pad(qp.lb,(0,1));qp.ub=pad(qp.ub,(0,1),value=100.)
        constraints=[qp.G,qp.b,qp.lb,qp.ub,qp.A_eq,qp.b_eq]
        solver=TorqueContinuitySolver(AnalyticBackend());solver.update(torch.ones(1,29),3.)
        result=solver.solve(qp)
        self.assertEqual(result[0,76].item(),7.)
        torch.testing.assert_close(result[:,47:76],(torch.arange(47,76,dtype=torch.float64)[None]+3.)/4.)
        for before,after in zip(constraints,[qp.G,qp.b,qp.lb,qp.ub,qp.A_eq,qp.b_eq]):self.assertIs(before,after)

    def test_runtime_configuration_writes_reach_wrapped_backend(self):
        backend=AnalyticBackend();backend.weight=10.
        solver=TorqueContinuitySolver(backend)
        tube=(torch.zeros(1,3,35),torch.ones(1,3),torch.ones(3))
        solver.tube=tube;solver.single_proximal=.0001;solver.equilibration_cap=10.
        self.assertIs(backend.tube,tube)
        self.assertEqual(backend.single_proximal,.0001)
        self.assertEqual(backend.equilibration_cap,10.)
        solver.update(torch.zeros(1,29),.1)
        self.assertEqual(backend.weight,10.)
        self.assertEqual(solver.weight,.1)

    def test_zero_weight_preserves_native_soft_tube_solution(self):
        from g1_soft_tracking_tube import SoftTrackingTubeBackend
        bare=SoftTrackingTubeBackend(10.,soft_rows=(True,False,False))
        wrapped_backend=SoftTrackingTubeBackend(10.,soft_rows=(True,False,False))
        wrapped=TorqueContinuitySolver(wrapped_backend)
        C=torch.zeros(1,3,35,dtype=torch.float64);C[0,0,0]=1.
        tube=(C,torch.tensor([[-1.,0.,0.]],dtype=torch.float64),torch.ones(3,dtype=torch.float64))
        for solver in (bare,wrapped):
            solver.tube=tube;solver.single_proximal=.0001;solver.equilibration_cap=10.
        direct=bare.solve(self.problem());through_wrapper=wrapped.solve(self.problem())
        self.assertTrue(bool(direct.status_ok.all()));self.assertTrue(bool(through_wrapper.status_ok.all()))
        torch.testing.assert_close(direct.z,through_wrapper.z,atol=0.,rtol=0.)
        self.assertEqual(wrapped_backend.last_augmented_problem.H.shape[-1],79)

    def test_support_mask_follows_contact_switch_without_smoothing_swing(self):
        torque=torch.zeros(4,29,dtype=torch.float64)
        mask=stance_torque_weights(torch.tensor([0,1,2,3]),torque)
        self.assertEqual(mask.sum(dim=1).tolist(),[0.,6.,6.,12.])
        self.assertTrue(bool((mask[:,12:]==0).all()))
        solver=TorqueContinuitySolver(AnalyticBackend());qp=self.problem()
        selected=mask[1:2].clone();solver.update(torch.full((1,29),5.,dtype=torch.float64),3.,selected)
        selected.fill_(1.)
        result=solver.solve(qp);expected=torch.arange(76,dtype=torch.float64)[None]
        expected[:,47:53]=(expected[:,47:53]+15.)/4.
        torch.testing.assert_close(result,expected)

    def test_invalid_support_mask_rejected(self):
        solver=TorqueContinuitySolver(AnalyticBackend());previous=torch.zeros(1,29)
        for mask in (torch.zeros(29),torch.full((1,29),-1.),torch.full((1,29),2.),torch.full((1,29),float('nan'))):
            with self.assertRaises(ValueError):solver.update(previous,1.,mask)
        for mode in (torch.tensor([4]),torch.tensor([-1]),torch.tensor([1.])):
            with self.assertRaises(ValueError):stance_torque_weights(mode,previous)

if __name__=='__main__':unittest.main()
