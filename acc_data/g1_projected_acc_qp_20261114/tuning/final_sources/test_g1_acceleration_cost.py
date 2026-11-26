import sys
import unittest
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'examples'))
from g1_acceleration_cost import preserve_com_acceleration


class AccelerationCostTest(unittest.TestCase):
    def test_cost_matches_com_acceleration_error_and_preserves_input(self):
        g = torch.Generator().manual_seed(7)
        J = torch.randn(2, 3, 35, generator=g, dtype=torch.float64)
        delta = torch.randn(2, 35, generator=g, dtype=torch.float64)
        weights = torch.linspace(1., 5., 35, dtype=torch.float64)
        W = preserve_com_acceleration(weights, J, 100.)
        measured = (delta[:, None] @ W @ delta[:, :, None]).flatten()
        expected = (weights*delta.square()).sum(-1) + 100.*(J@delta[..., None]).square().sum((1,2))
        torch.testing.assert_close(measured, expected)
        self.assertTrue(bool((torch.linalg.eigvalsh(W) > 0).all()))
        torch.testing.assert_close(weights, torch.linspace(1., 5., 35, dtype=torch.float64))
        self.assertIs(preserve_com_acceleration(weights, J, 0.), weights)
        for value in (-1., float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                preserve_com_acceleration(weights, J, value)
