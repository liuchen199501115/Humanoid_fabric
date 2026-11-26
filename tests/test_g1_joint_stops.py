import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'acc_data/amp_walk_20261103/feedback_followup'))
sys.path.insert(0, str(ROOT/'examples'))
from g1_joint_stop_mode import loaded_stop_directions
from unload_qp import G1DynamicsAccelerationQP as TrackingQP
from fabrics_sim.solver.AccelerationQP import G1DynamicsAccelerationQP
from fabrics_sim.solver.joint_stop import observed_stop_directions, augment_joint_stops
from fabrics_sim.solver.admm_qp import QPData
from fabrics_sim.solver.daqp_backend import DAQPBackend


class JointStopTests(unittest.TestCase):
    def test_unloaded_stops_release_and_loaded_stops_keep_their_sign(self):
        direction = torch.tensor([[1., -1., 1., -1., 0.]])
        acceleration = torch.tensor([[0., 0., 2., -2., 100.]])
        reaction = torch.tensor([[0., 3., 0., 0., -5.]])
        self.assertEqual(loaded_stop_directions(direction, acceleration, reaction).tolist(),
                         [[0., -1., 1., -1., 0.]])

    def inputs(self, sign=1.):
        z = torch.zeros(1, 35, dtype=torch.float64)
        ref = z.clone(); ref[0, 19] = sign*50
        torque = z[:, :29].clone(); torque[0, 13] = sign*5
        return dict(acc_reference=ref, mass_matrix=torch.eye(35, dtype=z.dtype)[None],
            bias_force=z, contact_jacobian=z.new_zeros(1, 12, 35),
            jdot_qdot=z.new_zeros(1, 12), contact_mode=torch.zeros(1, dtype=torch.long),
            effort_limits=20., known_joint_torque=torque,
            acc_lower=-float('inf'), acc_upper=float('inf'))

    def test_only_stationary_measured_stops_activate(self):
        q = torch.zeros(1, 29, dtype=torch.float64); v = torch.zeros_like(q)
        q[0, :4] = torch.tensor([1., -1., 1., .99]); v[0, 2] = .2
        d = observed_stop_directions(q, v, -torch.ones_like(q), torch.ones_like(q))
        self.assertEqual(d[0, :4].tolist(), [1., -1., 0., 0.])
        self.assertEqual(d.count_nonzero().item(), 2)

    def test_passive_reaction_and_dynamics_with_one_native_solve(self):
        for cls in (G1DynamicsAccelerationQP, TrackingQP):
            for sign in (-1., 1.):
                with self.subTest(backend=cls.__module__, sign=sign):
                    solver = cls(solver_backend='daqp', torque_weight=1.)
                    solver.solver.single_solve = True
                    directions = torch.zeros(1, 29, dtype=torch.float64)
                    directions[0, 13] = sign
                    with patch.object(solver.solver.daqp, 'solve', wraps=solver.solver.daqp.solve) as native:
                        r = solver.solve(**self.inputs(sign), joint_stop_directions=directions)
                    self.assertEqual(native.call_count, 1)
                    self.assertTrue(r.converged.all() and r.validated.all())
                    self.assertAlmostEqual(r.predicted_acc[0, 19].item(), 0., places=8)
                    self.assertLess(sign*r.joint_stop_reaction[0, 13].item(), -4.9)
                    self.assertLess(r.joint_dynamics_error.item(), 1e-8)
                    self.assertLessEqual(r.total_joint_torque.abs().max().item(), 20.)

    def test_inactive_mode_leaves_original_optimum_unchanged(self):
        for cls in (G1DynamicsAccelerationQP, TrackingQP):
            a = cls(solver_backend='daqp').solve(**self.inputs())
            b = cls(solver_backend='daqp').solve(**self.inputs(),
                joint_stop_directions=torch.zeros(1, 29, dtype=torch.float64))
            torch.testing.assert_close(a.acc, b.acc, atol=0, rtol=0)
            torch.testing.assert_close(a.total_joint_torque, b.total_joint_torque, atol=0, rtol=0)

    def test_stop_cannot_supply_an_outward_force(self):
        # Fixed outward torque can be balanced by an upper stop. Reversing
        # that fixed torque would require the stop to pull, and is infeasible.
        dtype = torch.float64
        d = torch.zeros(1, 29, dtype=dtype); d[0, 0] = 1
        for torque, valid in [(5., True), (-5., False)]:
            lo = torch.zeros(1, 76, dtype=dtype); hi = lo.clone()
            lo[0, 6] = -100; hi[0, 6] = 100
            lo[0, 47] = hi[0, 47] = torque
            A = torch.zeros(1, 35, 76, dtype=dtype)
            A[0, 6, 6] = 1; A[0, 6, 47] = -1
            qp = QPData(torch.eye(76, dtype=dtype)[None], torch.zeros(1, 76, dtype=dtype),
                torch.zeros(1, 0, 76, dtype=dtype), torch.zeros(1, 0, dtype=dtype), lo, hi, A,
                torch.zeros(1, 35, dtype=dtype))
            augmented, _ = augment_joint_stops(qp, d, torch.full_like(d, 20))
            result = DAQPBackend(single_solve=True).solve(augmented)
            self.assertEqual(bool(result.converged.item()), valid)
            if valid:
                self.assertAlmostEqual(result.z[0, -1].item(), 5., places=8)


if __name__ == '__main__':
    unittest.main()
