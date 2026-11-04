import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from motion_generation.data import (
    joint_spec, load_clip, localize, worldize, rotation6d, matrix6d, quat_matrix, matrix_quat, MotionWindows,
)
from motion_generation.model import MotionDenoiser, sample, schedule


class MotionGenerationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_rotations_and_world_roundtrip(self):
        rng = np.random.default_rng(4)
        q = rng.normal(size=(27, 4))
        q[0] = [1, 0, 0, 0]  # pi rotation
        r = quat_matrix(q)
        np.testing.assert_allclose(quat_matrix(matrix_quat(r)), r, atol=1e-6)
        np.testing.assert_allclose(matrix6d(rotation6d(r)), r, atol=1e-6)
        x = np.concatenate((rng.normal(size=(27, 3)), rotation6d(r), rng.normal(size=(27, 29))), 1).astype(np.float32)
        local, origin, frame = localize(x)
        np.testing.assert_allclose(worldize(local, origin, frame), x, atol=1e-6)
        np.testing.assert_allclose(local[1, :2], 0, atol=1e-7)
        degenerate = matrix6d(np.zeros((2, 6)))
        np.testing.assert_allclose(np.linalg.det(degenerate), 1)

    def test_resampling_slerp_joint_order_duration(self):
        names, _ = joint_spec()
        angle = np.linspace(0, np.pi, 11)
        q = np.stack((angle*0, angle*0, np.sin(angle/2), np.cos(angle/2)), -1)
        q[1::2] *= -1
        data = dict(fps=10., root_pos=np.stack((angle/np.pi, angle*0, angle*0+1), -1),
                    root_rot=q, dof_names=names[::-1], dof_pos=np.tile(np.arange(29)[::-1], (11, 1)))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'motion.pkl'
            with path.open('wb') as f:
                pickle.dump(data, f)
            x = load_clip(path, 20., names)
        self.assertEqual(x.shape, (21, 38))
        np.testing.assert_allclose(x[-1, :3], [1, 0, 1])
        np.testing.assert_allclose(x[:, 9:], np.tile(np.arange(29), (21, 1)))
        r = matrix6d(x[10, 3:9])
        np.testing.assert_allclose(r[:, 0], [0, 1, 0], atol=1e-6)

    def test_windows_cannot_cross_clips_or_split(self):
        x = np.zeros((90, 38), dtype=np.float32)
        x[:, 3:9] = [1, 0, 0, 0, 1, 0]
        x[:30, 9:] = 1
        x[30:60, 9:] = 2
        x[60:, 9:] = 3
        archive = dict(states=x, offsets=np.array([0, 30, 60, 90]), split=np.array([0, 1, 0]))
        train = MotionWindows(archive, 0, 25, 1)
        valid = MotionWindows(archive, 1, 25, 1)
        self.assertEqual(len(train), 8)
        self.assertEqual(len(valid), 4)
        for h, f, direction in train:
            self.assertIn(h[0, 9], (1, 3))
            self.assertTrue(np.all(f[:, 9] == h[0, 9]))
            self.assertTrue(np.isfinite(direction).all())

    def test_window_is_invariant_to_world_translation_and_yaw(self):
        x = np.zeros((27, 38), dtype=np.float32)
        x[:, 0] = np.linspace(0, 1, 27)
        x[:, 2] = .8
        x[:, 3:9] = [1, 0, 0, 0, 1, 0]
        frame = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        transformed = worldize(x, np.array([4., -3., 0.]), frame)
        original_local, _, _ = localize(x)
        rotated_local, _, _ = localize(transformed)
        np.testing.assert_allclose(original_local, rotated_local, atol=1e-6)

    def test_backward_and_sampling_depend_on_conditions(self):
        torch.manual_seed(4)
        model = MotionDenoiser(width=32, layers=1, horizon=4)
        h = torch.randn(2, 2, 38)
        direction = torch.tensor([[1., 0.], [0., 1.]])
        x = torch.randn(2, 4, 38)
        out = model(x, torch.tensor([1, 2]), h, direction)
        out.square().mean().backward()
        self.assertGreater(model.heading.weight.grad.abs().sum().item(), 0)
        self.assertTrue(torch.isfinite(out).all())
        alpha = schedule(20, 'cpu')
        self.assertTrue(torch.all(alpha[1:] < alpha[:-1]))
        torch.manual_seed(5)
        a = sample(model, h, direction, alpha, steps=5)
        torch.manual_seed(5)
        b = sample(model, h, -direction, alpha, steps=5)
        self.assertEqual(a.shape, (2, 4, 38))
        self.assertFalse(torch.allclose(a, b))
        self.assertTrue(torch.isfinite(a).all())


if __name__ == '__main__':
    unittest.main()
