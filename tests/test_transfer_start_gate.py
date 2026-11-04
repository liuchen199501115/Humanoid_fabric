import unittest
import torch
from fabrics_sim.reference.contact_transfer import TransferStartGate


class TransferStartGateTests(unittest.TestCase):
    def test_sustained_rest_and_reset(self):
        gate = TransferStartGate()
        args = (True, torch.zeros(3), torch.zeros(29), torch.ones(2)*100, .002)
        for _ in range(99):
            self.assertFalse(gate.update(*args))
        self.assertTrue(gate.update(*args))
        self.assertFalse(gate.update(False, *args[1:]))
        self.assertFalse(gate.update(*args))

    def test_rejects_motion_contact_loss_and_nonfinite_data(self):
        for cv, jv, loads in [(torch.ones(3)*.04, torch.zeros(29), torch.ones(2)*100),
                              (torch.zeros(3), torch.ones(29)*.27, torch.ones(2)*100),
                              (torch.zeros(3), torch.zeros(29), torch.tensor([100., 0.])),
                              (torch.ones(3)*float('nan'), torch.zeros(29), torch.ones(2)*100)]:
            gate = TransferStartGate()
            for _ in range(110):
                self.assertFalse(gate.update(True, cv, jv, loads, .002))

    def test_invalid_timestep_rejected(self):
        for dt in (0, -1, float('nan')):
            with self.assertRaises(ValueError):
                TransferStartGate().update(True, torch.zeros(3), torch.zeros(29), torch.ones(2), dt)
