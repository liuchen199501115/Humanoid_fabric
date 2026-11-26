"""Acceptance, constraint identity and executed-solution provenance for the guard."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'examples'))
from g1_guarded_shoulder import GuardedShoulderWeights

NAMES=['left_shoulder_roll_joint','right_shoulder_roll_joint']+[f'joint_{i}' for i in range(27)]


def output(shoulder=10., com=1., foot=1., valid=True):
    residuals={name:torch.ones(1,n) for name,n in (
        ('center_of_mass',3),('body_orientation',3),('source_pelvis_orientation',3),('feet_pose',12),('joint_posture',29))}
    residuals['center_of_mass'].fill_(com)
    residuals['feet_pose'][:,:3]=foot
    residuals['joint_posture'][:,:2]=shoulder
    return SimpleNamespace(validated=torch.tensor([valid]),task_residuals=residuals)


class FakeSolver:
    def __init__(self, cases, change_constraints=False):
        self.cases=cases;self.calls=[];self.resets=0;self.change_constraints=change_constraints
    def reset(self):
        self.resets+=1
    def solve(self, task_requests):
        scale=float(task_requests[0].metric[0,0,0]);self.calls.append(scale)
        p=SimpleNamespace(**{k:torch.zeros(1,1) for k in ('A_eq','b_eq','G','b','lb','ub')})
        if self.change_constraints and scale>1:p.b+=1
        self.last_problem=p;self.last_priority_problems=[p];self.last_scale=scale
        return self.cases[round(scale)]


class GuardTest(unittest.TestCase):
    def request(self):
        return SimpleNamespace(name='joint_posture',metric=torch.eye(29)[None],desired=torch.arange(29.))

    def test_backtracks_when_com_worsens(self):
        guard=GuardedShoulderWeights();request=self.request()
        owner=FakeSolver({1:output(),4:output(5.,2.),2:output(7.,1.03)})
        result=guard.solve(owner.solve,owner,joint_names=NAMES,multiplier=4,task_requests=[request])
        self.assertEqual(owner.calls,[1.,4.,1.9999998807907104])
        self.assertIs(result,owner.cases[2]);self.assertAlmostEqual(owner.last_scale,2.,places=6)
        self.assertTrue(guard.diagnostic['shoulder_guard_accepted'])
        torch.testing.assert_close(request.desired,torch.arange(29.))

    def test_rejection_restores_original_objective_and_metadata(self):
        guard=GuardedShoulderWeights();request=self.request();metric=request.metric
        owner=FakeSolver({1:output(),4:output(5.,2.),2:output(7.,2.)})
        result=guard.solve(owner.solve,owner,joint_names=NAMES,multiplier=4,task_requests=[request])
        self.assertIs(result,owner.cases[1]);self.assertIs(request.metric,metric)
        self.assertEqual(owner.last_scale,1.);self.assertEqual(owner.resets,1)
        self.assertFalse(guard.diagnostic['shoulder_guard_accepted'])

    def test_feasible_candidate_with_changed_constraints_is_rejected(self):
        guard=GuardedShoulderWeights();request=self.request()
        owner=FakeSolver({1:output(),4:output(5.),2:output(7.)},change_constraints=True)
        result=guard.solve(owner.solve,owner,joint_names=NAMES,multiplier=4,task_requests=[request])
        self.assertIs(result,owner.cases[1]);self.assertEqual(owner.last_problem.b.item(),0.)
        self.assertEqual(guard.diagnostic['shoulder_guard_reason'],'changed_constraints')

    def test_preserves_each_foot_and_requires_real_improvement(self):
        guard=GuardedShoulderWeights()
        self.assertFalse(guard.assess(output(),output(5.,foot=2.),[0,1])[0])
        self.assertFalse(guard.assess(output(),output(9.99),[0,1])[0])
        self.assertFalse(guard.assess(output(),output(5.,valid=False),[0,1])[0])

    def test_invalid_baseline_and_inactive_do_not_try_candidates(self):
        for scale,valid in ((1.,True),(4.,False)):
            guard=GuardedShoulderWeights();request=self.request()
            owner=FakeSolver({1:output(valid=valid)})
            result=guard.solve(owner.solve,owner,joint_names=NAMES,multiplier=scale,task_requests=[request])
            self.assertIs(result,owner.cases[1]);self.assertEqual(owner.calls,[1.])

    def test_invalid_margins_and_nonfinite_residuals(self):
        for relative,slack in ((-1.,1.),(.02,-1.),(float('nan'),1.)):
            with self.assertRaises(ValueError):GuardedShoulderWeights(relative,slack)
        self.assertFalse(GuardedShoulderWeights().assess(output(),output(5.,float('nan')),[0,1])[0])


if __name__=='__main__':unittest.main()
