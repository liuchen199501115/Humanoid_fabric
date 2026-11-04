import unittest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
import torch
from test_g1 import source_start_contacts, update_contact_mode_state_machine
from fabrics_sim.solver.dynamic_consistent_gpu import BatchedFootContactDetector, ContactDetectorConfig


class DynamicStartTests(unittest.TestCase):
    def test_airborne_swing_foot_is_not_seeded_as_support(self):
        feet = torch.tensor([[[0., .1, .115, 0., 0., 0., 1.],
                              [0., -.1, .035, 0., 0., 0., 1.]]])
        self.assertEqual(source_start_contacts(feet).tolist(), [[False, True]])

    def test_tilted_foot_uses_lowest_corner_not_ankle_height(self):
        angle = torch.tensor(.2)
        height = .12 * angle.sin() + .035 * angle.cos()
        feet = torch.tensor([[[0., .1, .2, 0., 0., 0., 1.],
                              [0., -.1, height, 0., (angle/2).sin(), 0., (angle/2).cos()]]])
        self.assertEqual(source_start_contacts(feet).tolist(), [[False, True]])

    def test_geometry_seed_is_removed_when_measured_support_is_absent(self):
        detector = BatchedFootContactDetector(num_envs=1, device='cpu',
            config=ContactDetectorConfig(force_on=20., force_off=10., confirm_steps=3))
        detector.contact[:] = torch.tensor([[False, True]])
        for _ in range(3):
            left, right, _ = detector.update(torch.zeros(1, 2))
        actual = torch.stack((left, right), dim=1)
        mode, _ = update_contact_mode_state_machine(torch.tensor([2]), torch.tensor([2]),
            actual, torch.zeros(1, dtype=torch.long), 3, torch.zeros(1, 2))
        self.assertEqual(mode.tolist(), [0])

    def test_short_ground_impact_is_acquired_without_weakening_release(self):
        detector=BatchedFootContactDetector(1,'cpu',ContactDetectorConfig(
            force_on=20.,force_off=10.,confirm_steps=3,impact_force_on=100.))
        detector.contact[:]=torch.tensor([[False,True]])
        _,_,mode=detector.update(torch.tensor([[1042.,150.]]))
        self.assertEqual(mode.item(),3)
        for _ in range(2):
            _,_,mode=detector.update(torch.tensor([[0.,150.]]))
            self.assertEqual(mode.item(),3)
        _,_,mode=detector.update(torch.tensor([[0.,150.]]))
        self.assertEqual(mode.item(),2)

    def test_small_single_sample_still_needs_confirmation(self):
        detector=BatchedFootContactDetector(1,'cpu',ContactDetectorConfig(confirm_steps=3,impact_force_on=100.))
        _,_,mode=detector.update(torch.tensor([[30.,0.]]))
        self.assertEqual(mode.item(),0)


if __name__ == '__main__':
    unittest.main()
