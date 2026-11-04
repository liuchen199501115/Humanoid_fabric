import importlib.util
import unittest
from unittest.mock import patch
import torch

from fabrics_sim.solver.admm_qp import QPData
from fabrics_sim.solver.daqp_backend import DAQPBackend


@unittest.skipUnless(importlib.util.find_spec('daqp'), 'Optional daqp dependency is not installed')
class DAQPBackendTests(unittest.TestCase):
    def test_numerical_failure_can_use_proximal_retry_with_same_certificate(self):
        for failed_flag in (-1,-2,1):
            with self.subTest(failed_flag=failed_flag):
                self.check_proximal_retry(failed_flag)

    def check_proximal_retry(self,failed_flag):
        solver=DAQPBackend();original=solver.daqp.solve;attempts=[]
        def unstable(*args,**kwargs):
            attempts.append(kwargs['eps_prox'])
            x,value,flag,info=original(*args,**kwargs)
            if kwargs['eps_prox']<0:
                x[1]+=1e4  # Failed native iterates need not be nearly feasible.
                info['lam']=info['lam'].copy()+.5
                flag=failed_flag
            return x,value,flag,info
        with patch.object(solver.daqp,'solve',side_effect=unstable), patch(
                'fabrics_sim.solver.active_set_predictor.predict_active_set',return_value=None):
            result=solver.solve(self.problem())
        self.assertTrue(result.status_ok.all())
        self.assertGreater(attempts[-1],0)
        torch.testing.assert_close(result.z,torch.tensor([[.3,1.,0.]],dtype=torch.float64))

    def test_nearly_feasible_candidate_is_repaired_before_acceptance(self):
        solver=DAQPBackend();original=solver.daqp.solve
        def inaccurate(*args,**kwargs):
            x,value,flag,info=original(*args,**kwargs)
            x[1]+=1e-4
            return x,value,flag,info
        with patch.object(solver.daqp,'solve',side_effect=inaccurate):
            result=solver.solve(self.problem())
        self.assertTrue(result.status_ok.all())
        torch.testing.assert_close(result.z,torch.tensor([[.3,1.,0.]],dtype=torch.float64),rtol=0,atol=1e-8)
        self.assertIn('KKT polished',solver.last_status[0])

    def test_iteration_limit_needs_independent_optimality_and_feasibility(self):
        for corrupt in (False,True):
            solver=DAQPBackend();original=solver.daqp.solve
            def limited(*args,**kwargs):
                x,value,_,info=original(*args,**kwargs)
                if corrupt:x[1]+=.5
                return x,value,-4,info
            with patch.object(solver.daqp,'solve',side_effect=limited):
                result=solver.solve(self.problem())
            self.assertEqual(bool(result.status_ok.all()),not corrupt)

    def problem(self):
        d=torch.float64
        return QPData(torch.eye(3,dtype=d)[None],torch.tensor([[0.,-2.,0.]],dtype=d),
            torch.tensor([[[1.,0.,0.],[0.,1.,0.]]],dtype=d),torch.tensor([[.3,1.]],dtype=d),
            torch.tensor([[-10.,-10.,0.]],dtype=d),torch.tensor([[10.,10.,0.]],dtype=d),
            torch.tensor([[[1.,0.,0.],[2.,0.,0.],[0.,0.,1.]]],dtype=d),torch.tensor([[.3,.6,0.]],dtype=d))

    def test_dependent_equalities_fixed_bounds_and_changing_active_set(self):
        solver=DAQPBackend();qp=self.problem()
        result=solver.solve(qp,workspace='priority')
        self.assertTrue(result.status_ok.all())
        torch.testing.assert_close(result.z,torch.tensor([[.3,1.,0.]],dtype=torch.float64))
        qp.h[0,1]=1.
        changed=solver.solve(qp,workspace='priority')
        self.assertTrue(changed.status_ok.all())
        torch.testing.assert_close(changed.z,torch.tensor([[.3,-1.,0.]],dtype=torch.float64))
        solver.solve(qp,workspace='tracking')
        self.assertEqual(set(solver._previous_solutions),{('priority',0),('tracking',0)})
        solver.reset(torch.tensor([0]))
        self.assertFalse(solver._previous_solutions)

    def test_infeasible_constraints_are_rejected_after_warm_start(self):
        solver=DAQPBackend();qp=self.problem();solver.solve(qp)
        qp.b[0,0]=.2
        result=solver.solve(qp)
        self.assertFalse(result.status_ok.any())
        self.assertFalse(result.converged.any())
        self.assertFalse(solver._previous_solutions)

    def test_semidefinite_objective_keeps_original_optimum(self):
        qp=self.problem();qp.H[0,2,2]=0.
        result=DAQPBackend().solve(qp)
        self.assertTrue(result.status_ok.all())
        torch.testing.assert_close(result.z,torch.tensor([[.3,1.,0.]],dtype=torch.float64))

    def test_nonconvex_objective_and_inconsistent_box_are_rejected(self):
        qp=self.problem();qp.H[0,1,1]=-1.
        qp.h[0,0]=1e9  # Linear cost cannot excuse a nonconvex Hessian.
        self.assertFalse(DAQPBackend().solve(qp).status_ok.any())
        qp=self.problem();qp.lb[0,2]=1.
        self.assertFalse(DAQPBackend().solve(qp).status_ok.any())

    def test_nonfinite_constraint_is_not_passed_to_native_solver(self):
        qp=self.problem();qp.G[0,0,0]=float('nan')
        self.assertFalse(DAQPBackend().solve(qp).status_ok.any())

    def test_batches_do_not_share_primal_warm_starts(self):
        q=self.problem()
        qp=QPData(**{k:v.repeat(2,*([1]*(v.ndim-1))) for k,v in vars(q).items()})
        qp.h[1,1]=.5
        result=DAQPBackend().solve(qp)
        self.assertTrue(result.status_ok.all())
        torch.testing.assert_close(result.z[:,1],torch.tensor([1.,-.5],dtype=torch.float64))


if __name__=='__main__':unittest.main()
