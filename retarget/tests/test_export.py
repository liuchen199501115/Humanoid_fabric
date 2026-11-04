import pickle
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from retargeting.export import (
    _forward_kinematics,
    _remove_short_true_runs,
    core_contact_sequence,
    export_robot_pkl,
    project_rotations_to_urdf_dofs,
)
from retargeting.urdf import DEFAULT_FEET, load_urdf_spec


REPO_ROOT = Path(__file__).resolve().parents[1]
URDF_PATH = REPO_ROOT / "assets/g1_description/g1_29dof.urdf"


def rotation_matrix_from_rotvec(rotvec):
    rotvec = np.asarray(rotvec, dtype=np.float64)
    angle = np.linalg.norm(rotvec)
    if angle < 1e-12:
        return np.eye(3, dtype=np.float64)
    x, y, z = rotvec / angle
    skew = np.array(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]],
        dtype=np.float64,
    )
    return np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)


def rotations_at_urdf_rest(spec, frame_count=1):
    rotations = np.broadcast_to(
        spec.base_rotations[None],
        (frame_count, *spec.base_rotations.shape),
    ).copy()
    rotations[:, 0] = np.eye(3)
    return rotations


class RobotExportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = load_urdf_spec(URDF_PATH)

    def test_off_axis_rotation_uses_rotation_vector_projection(self):
        rotations = rotations_at_urdf_rest(self.spec)
        joint_name = "left_shoulder_roll_joint"
        joint_idx = self.spec.joint_names.index(joint_name)
        expected_rotvec = np.array([-0.26, 2.9, 0.1])
        rotations[0, joint_idx] = (
            self.spec.base_rotations[joint_idx]
            @ rotation_matrix_from_rotvec(expected_rotvec)
        )

        _, _, dof_pos, _ = project_rotations_to_urdf_dofs(
            rotations, self.spec.joint_names, URDF_PATH
        )
        dof_idx = self.spec.dof_names.index(joint_name)
        self.assertAlmostEqual(dof_pos[0, dof_idx], expected_rotvec[0], places=6)

        # Projection is name based, so custom target H5 ordering remains safe.
        permutation = np.arange(len(self.spec.joint_names))[::-1]
        _, _, reordered_dof, _ = project_rotations_to_urdf_dofs(
            rotations[:, permutation],
            [self.spec.joint_names[idx] for idx in permutation],
            URDF_PATH,
        )
        np.testing.assert_allclose(reordered_dof, dof_pos, atol=1e-9)

    def test_export_clamps_limits_and_aligns_foot_collision_geometry(self):
        rotations = rotations_at_urdf_rest(self.spec, frame_count=2)
        joint_name = "right_knee_joint"
        joint_idx = self.spec.joint_names.index(joint_name)
        dof_idx = self.spec.dof_names.index(joint_name)
        out_of_range_angle = self.spec.lower[dof_idx] - 0.5
        axis = self.spec.axes[joint_idx]
        rotations[:, joint_idx] = (
            self.spec.base_rotations[joint_idx]
            @ rotation_matrix_from_rotvec(axis * out_of_range_angle)
        )

        with TemporaryDirectory() as output_dir:
            output_path = Path(output_dir) / "motion.pkl"
            export_robot_pkl(
                rotations,
                np.zeros((2, 3), dtype=np.float64),
                self.spec.joint_names,
                URDF_PATH,
                fps=30.0,
                out_path=output_path,
            )
            with output_path.open("rb") as handle:
                motion = pickle.load(handle)

        dof_pos = np.asarray(motion["dof_pos"], dtype=np.float64)
        self.assertEqual(tuple(motion["dof_names"]), self.spec.dof_names)
        self.assertTrue(np.all(dof_pos >= self.spec.lower[None] - 1e-6))
        self.assertTrue(np.all(dof_pos <= self.spec.upper[None] + 1e-6))
        self.assertAlmostEqual(dof_pos[0, dof_idx], self.spec.lower[dof_idx], places=6)

        positions, orientations = _forward_kinematics(
            np.asarray(motion["root_pos"], dtype=np.float64),
            np.broadcast_to(np.eye(3), (2, 3, 3)),
            dof_pos,
            self.spec,
        )
        name_to_index = {
            name: idx for idx, name in enumerate(self.spec.joint_names)
        }
        minimum_heights = []
        for foot_name in DEFAULT_FEET:
            foot_idx = name_to_index[foot_name]
            collision = self.spec.foot_collisions[foot_name]
            centers = positions[:, foot_idx, None] + np.einsum(
                "tij,sj->tsi", orientations[:, foot_idx], collision.centers
            )
            minimum_heights.append(
                np.min(centers[0, :, 2] - collision.radii)
            )
        self.assertAlmostEqual(min(minimum_heights), 0.0, places=6)

    def test_raw_projection_can_be_exported_without_limit_clamping(self):
        rotations = rotations_at_urdf_rest(self.spec)
        joint_name = "right_knee_joint"
        joint_idx = self.spec.joint_names.index(joint_name)
        dof_idx = self.spec.dof_names.index(joint_name)
        out_of_range_angle = self.spec.lower[dof_idx] - 0.5
        rotations[:, joint_idx] = (
            self.spec.base_rotations[joint_idx]
            @ rotation_matrix_from_rotvec(self.spec.axes[joint_idx] * out_of_range_angle)
        )

        with TemporaryDirectory() as output_dir:
            motion = export_robot_pkl(
                rotations,
                np.zeros((1, 3), dtype=np.float64),
                self.spec.joint_names,
                URDF_PATH,
                fps=30.0,
                out_path=Path(output_dir) / "raw.pkl",
                align_ground=False,
                clamp_limits=False,
            )
        self.assertLess(motion["dof_pos"][0, dof_idx], self.spec.lower[dof_idx])

    def test_core_contact_sequence_rejects_short_stationary_runs(self):
        mask = np.array([0, 1, 0, 1, 1, 1, 0], dtype=bool)
        np.testing.assert_array_equal(
            _remove_short_true_runs(mask, 2),
            np.array([0, 0, 0, 1, 1, 1, 0], dtype=bool),
        )

    def _grounded_root_positions(self, frame_count):
        root_pos = np.zeros((frame_count, 3), dtype=np.float64)
        root_matrix = np.broadcast_to(
            np.eye(3), (frame_count, 3, 3)
        ).copy()
        dof_pos = np.zeros(
            (frame_count, len(self.spec.dof_names)), dtype=np.float64
        )
        positions, orientations = _forward_kinematics(
            root_pos[:1], root_matrix[:1], dof_pos[:1], self.spec
        )
        name_to_index = {
            name: idx for idx, name in enumerate(self.spec.joint_names)
        }
        minimum_height = np.inf
        for foot_name in DEFAULT_FEET:
            foot_idx = name_to_index[foot_name]
            collision = self.spec.foot_collisions[foot_name]
            centers = positions[:, foot_idx, None] + np.einsum(
                "tij,sj->tsi",
                orientations[:, foot_idx],
                collision.centers,
            )
            minimum_height = min(
                minimum_height,
                np.min(centers[..., 2] - collision.radii[None]),
            )
        root_pos[:, 2] -= minimum_height
        return root_pos, root_matrix, dof_pos

    def test_core_contact_sequence_rejects_stationary_airborne_feet(self):
        root_pos, root_matrix, dof_pos = self._grounded_root_positions(20)
        root_pos[:, 2] += 0.20
        contacts = core_contact_sequence(
            root_pos,
            root_matrix,
            dof_pos,
            self.spec,
            fps=30.0,
            minimum_duration=0.0,
            smoothing_duration=0.0,
        )
        self.assertFalse(contacts.any())

    def test_core_contact_sequence_retains_near_ground_support(self):
        root_pos, root_matrix, dof_pos = self._grounded_root_positions(16)
        root_pos[:, 2] += np.linspace(0.0, 0.07, len(root_pos))
        contacts = core_contact_sequence(
            root_pos,
            root_matrix,
            dof_pos,
            self.spec,
            fps=30.0,
            velocity_threshold=0.08,
            minimum_duration=0.0,
            smoothing_duration=0.0,
            support_fallback_height=0.08,
        )
        self.assertTrue(contacts.any(axis=1).all())

    def test_export_contains_core_contact_sequence(self):
        frame_count = 8
        rotations = rotations_at_urdf_rest(self.spec, frame_count=frame_count)
        with TemporaryDirectory() as output_dir:
            motion = export_robot_pkl(
                rotations,
                np.zeros((frame_count, 3), dtype=np.float64),
                self.spec.joint_names,
                URDF_PATH,
                fps=30.0,
                out_path=Path(output_dir) / "motion.pkl",
            )
        self.assertEqual(motion["contact_sequence"].shape, (frame_count, 2))
        self.assertEqual(motion["contact_sequence"].dtype, np.bool_)
        self.assertTrue(motion["contact_sequence"].all())
        self.assertEqual(motion["contact_names"], ["left_foot", "right_foot"])


if __name__ == "__main__":
    unittest.main()
