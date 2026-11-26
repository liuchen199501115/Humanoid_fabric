"""Check paper projection algebra independently of the simulator/QP."""
import unittest

import torch

from fabrics_sim.fabrics.priority import dynamic_nullspace, prioritize_requests
from fabrics_sim.fabrics.task_request import TaskRequest


def task(name, A, target=0.):
    values = torch.full(A.shape[:2], target, dtype=A.dtype)
    return TaskRequest(name, A, torch.zeros_like(values), values,
                       torch.diag_embed(torch.ones_like(values)),
                       torch.ones_like(values, dtype=torch.bool))


class FabricPriorityTest(unittest.TestCase):
    def test_float32_inertia_roundoff_but_not_asymmetric_model_is_accepted(self):
        D = torch.eye(35)[None]
        D[0, 2, 4] = 1e-7
        J = D[:, :3].clone()
        N = dynamic_nullspace(J, D)
        torch.testing.assert_close(J.double()@N, torch.zeros_like(J).double(), atol=1e-12, rtol=0.)
        D[0, 2, 4] = .01
        with self.assertRaisesRegex(ValueError, 'not symmetric'):
            dynamic_nullspace(J, D)

    def test_mass_consistent_projector_with_redundant_and_zero_rows(self):
        torch.manual_seed(14)
        L = torch.randn(2, 35, 35, dtype=torch.float64)
        D = L @ L.transpose(-1, -2) + torch.eye(35)
        J = torch.randn(2, 3, 35, dtype=D.dtype)
        J = torch.cat((J, 2*J[:, :1], torch.zeros_like(J[:, :1])), 1)
        N = dynamic_nullspace(J, D)
        torch.testing.assert_close(J@N, torch.zeros_like(J), atol=1e-11, rtol=0)
        torch.testing.assert_close(N@N, N, atol=1e-11, rtol=0)
        torch.testing.assert_close(N.transpose(-1,-2)@D, D@N, atol=1e-11, rtol=0)
        # With no active stability directions the projector is identity.
        torch.testing.assert_close(dynamic_nullspace(J*0, D), torch.eye(35, dtype=D.dtype)[None].expand_as(D))

    def test_conflicting_low_target_cannot_change_high_solution(self):
        A = torch.zeros(1, 1, 35, dtype=torch.float64)
        A[..., 6] = 1.
        low_A = A.clone()
        low_A[..., 7] = 1.
        D = torch.eye(35, dtype=A.dtype)[None]
        samples = {'center_of_mass': A, 'joint_posture': low_A}
        for target in (-100., 0., 100.):
            high, low = task('center_of_mass', A, 2.), task('joint_posture', low_A, target)
            projected, diagnostics = prioritize_requests([high, low], (D, D, D),
                (samples, samples, samples), torch.zeros(1, 35), .001)
            H, f = zip(*(r.natural_form() for r in projected))
            solution = -torch.linalg.pinv(sum(H)) @ sum(f)[..., None]
            self.assertAlmostEqual((A@solution).item(), 2., places=10)
            self.assertEqual(projected[1].A[..., :6].count_nonzero(), 0)
            self.assertEqual(diagnostics['fabric_priority_nullspace_error'].item(), 0.)
            torch.testing.assert_close(low.A, low_A)  # original physical leaf is untouched

    def test_curvature_includes_derivative_of_nullspace(self):
        # J_high(t) = [1,t], J_low(t) = [0,1]. At t=0, derivative
        # of J_low*N is [-1,0], even though derivative of J_low is zero.
        eps = 1e-4
        A = torch.zeros(1, 1, 35, dtype=torch.float64)
        A[..., 6] = 1.
        B = A.roll(1, dims=-1)
        samples = tuple({'center_of_mass': A+t*B, 'hand_pose': B}
                        for t in (0., eps, -eps))
        D = torch.eye(35, dtype=A.dtype)[None]
        velocity = A[:, 0].clone()
        projected, _ = prioritize_requests([task('center_of_mass', A), task('hand_pose', B)],
            (D, D, D), samples, velocity, eps)
        self.assertAlmostEqual(projected[1].bias.item(), -1., places=7)
        self.assertEqual(projected[0].bias.item(), 0.)

    def test_generalized_root_retains_floating_base_task_columns(self):
        A = torch.zeros(1, 1, 35, dtype=torch.float64)
        A[..., 1], A[..., 6] = 1., .1
        D = torch.eye(35, dtype=A.dtype)[None]
        samples = {'center_of_mass': A, 'joint_posture': D}
        projected, _ = prioritize_requests([task('center_of_mass', A), task('joint_posture', D)],
            (D, D, D), (samples, samples, samples), torch.zeros(1, 35), .001)
        torch.testing.assert_close(projected[0].A, A)
        self.assertGreater(projected[1].A[..., :6].abs().max().item(), 0.)

    def test_inactive_and_zero_weight_stability_directions_are_released(self):
        A = torch.zeros(1, 2, 35, dtype=torch.float64)
        A[0, 0, 6], A[0, 1, 7] = 1., 1.
        high = task('center_of_mass', A)
        high.active[0, 0] = False
        high.metric[0, 1, 1] = 0.
        D = torch.eye(35, dtype=A.dtype)[None]
        samples = {'center_of_mass': A, 'joint_posture': A}
        projected, diagnostics = prioritize_requests([high, task('joint_posture', A)],
            (D, D, D), (samples, samples, samples), torch.zeros(1, 35), .001)
        torch.testing.assert_close(projected[1].A, A)
        torch.testing.assert_close(diagnostics['fabric_priority_nullspace'], D)

    def test_invalid_sampling_rejected(self):
        for step in (0., -1., float('nan')):
            with self.assertRaises(ValueError):
                prioritize_requests([], (), (), torch.zeros(1, 35), step)
        with self.assertRaises(ValueError):
            prioritize_requests([], (), (), torch.zeros(1, 35), .001)


if __name__ == '__main__':
    unittest.main()
