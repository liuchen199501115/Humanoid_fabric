import sys
import unittest
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'examples'))
from g1_com_vertical_limit import limit_upward_target, limit_upward_speed_target


class VerticalTargetLimitTest(unittest.TestCase):
    def test_cap_preserves_horizontal_and_downward_targets(self):
        desired = torch.tensor([[4., -8., 17.], [-2., 9., -3.]], dtype=torch.float64)
        original = desired.clone()
        capped = limit_upward_target(desired, 3., 1.)
        torch.testing.assert_close(capped, torch.tensor([[4., -8., 3.], [-2., 9., -3.]], dtype=torch.float64))
        torch.testing.assert_close(desired, original, rtol=0, atol=0)
        torch.testing.assert_close(limit_upward_target(desired, 3., 0.), desired, rtol=0, atol=0)
        torch.testing.assert_close(limit_upward_target(desired, 3., .5), (desired+capped)/2)

    def test_invalid_policy_is_rejected(self):
        for maximum, blend in [(-1., 1.), (float('inf'), 1.), (3., -1.), (3., float('nan'))]:
            with self.subTest(maximum=maximum, blend=blend), self.assertRaises(ValueError):
                limit_upward_target(torch.zeros(1, 3), maximum, blend)

    def test_speed_policy_reduces_ascent_but_allows_descent_braking(self):
        desired = torch.tensor([[1., 2., 6.], [3., 4., 6.], [5., 6., -2.]])
        velocity = torch.tensor([.8, -.5, 2.])
        result = limit_upward_speed_target(desired, velocity, .3, 10., -6., 1.)
        torch.testing.assert_close(result, torch.tensor([[1., 2., -5.], [3., 4., 6.], [5., 6., -6.]]))
        torch.testing.assert_close(limit_upward_speed_target(desired, velocity, .3, 10., -6., 0.), desired, rtol=0, atol=0)
        for wrong in [torch.zeros(2), torch.tensor([float('nan'), 0., 0.])]:
            with self.assertRaises(ValueError):
                limit_upward_speed_target(desired, wrong, .3, 10., -6., 1.)
