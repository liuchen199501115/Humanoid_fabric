import unittest
import numpy as np
from scipy import sparse
import clarabel
from preview_constraint_generation import solve

class ConstraintGenerationTests(unittest.TestCase):
 def settings(self):
  s=clarabel.DefaultSettings();s.verbose=False;s.tol_feas=1e-9;s.tol_gap_abs=1e-9;s.tol_gap_rel=1e-9;return s
 def run_problem(self,A,b,max_rounds=6):
  n=A.shape[1];return solve(sparse.eye(n,format='csc'),np.full(n,-2.),sparse.csc_matrix((0,n)),np.zeros(0),[],sparse.csc_matrix(A),np.array(b,dtype=float),[len(b)],np.arange(len(b),0,-1),self.settings(),initial_per_block=1,max_rounds=max_rounds)
 def test_omitted_binding_constraint_is_added_before_acceptance(self):
  s,d=self.run_problem(np.eye(2),[0.,.2]);self.assertEqual(str(s.status),'Solved')
  np.testing.assert_allclose(s.x,[0.,.2],atol=1e-7)
  self.assertEqual(d['rounds'][0]['rows'],1);self.assertGreater(d['rounds'][0]['full_collision_violation'],1.)
  self.assertEqual(d['selected_collision_rows'],2);self.assertLess(d['full_collision_linear_violation'],1e-7)
 def test_numerical_failure_retries_the_complete_problem(self):
  from unittest.mock import patch
  from types import SimpleNamespace
  original=clarabel.DefaultSolver;calls=[]
  def factory(*args):
   calls.append(1)
   if len(calls)==1:return SimpleNamespace(solve=lambda:SimpleNamespace(status='NumericalError',x=np.zeros(2)))
   return original(*args)
  with patch('preview_constraint_generation.clarabel.DefaultSolver',side_effect=factory):
   result,diagnostic=self.run_problem(np.eye(2),[0.,.2])
  self.assertEqual(str(result.status),'Solved');self.assertEqual(len(calls),2)
  self.assertEqual(diagnostic['selected_collision_rows'],2)
 def test_round_limit_restores_all_constraints(self):
  s,d=self.run_problem(np.eye(2),[0.,.2],max_rounds=0)
  self.assertEqual(str(s.status),'Solved');self.assertEqual(d['selected_collision_rows'],2)
 def test_infeasibility_cannot_be_hidden_by_initial_selection(self):
  s,d=self.run_problem(np.array([[1.],[-1.]]),[0.,-1.])
  self.assertEqual(str(s.status),'PrimalInfeasible');self.assertEqual(d['selected_collision_rows'],2)
if __name__=='__main__':unittest.main()
