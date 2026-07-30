#!/usr/bin/env python3
"""Smoke and integration tests for the production robot helpers.

Run the tests from any directory with::

    python prof/test.py

Open the imported G1 in Isaac Sim with::

    python prof/test.py --visualize

The integration tests use the G1 29-DoF URDF and run on CPU, so a CUDA GPU is
not required.  Missing runtime dependencies are reported as skipped tests.
"""

from __future__ import annotations

import sys
import unittest
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest import mock

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
G1_URDF = (
    SRC_ROOT
    / "fabrics_sim"
    / "models"
    / "robots"
    / "urdf"
    / "g1"
    / "g1_29dof.urdf"
)
G1_USD = (
    SRC_ROOT
    / "fabrics_sim"
    / "models"
    / "robots"
    / "USD"
    / "g1_sim"
    / "29dof"
    / "usd"
    / "g1_29dof_rev_1_0"
    / "g1_29dof_rev_1_0.usd"
)
G1_VISUALIZATION_URDF = G1_URDF.with_name("g1_29dof_rev_1_0.urdf")


def is_git_lfs_pointer(path: Path) -> bool:
    """Return True when path is metadata rather than the requested LFS asset."""
    try:
        return path.read_bytes().startswith(b"version https://git-lfs.github.com/spec/v1")
    except OSError:
        return False

sys.path.insert(0, str(SRC_ROOT))

try:
    import torch
    import warp as wp

    from fabrics_sim.prod import cuda_stream_utils
    from fabrics_sim.prod.import_urdf import parse_urdf_annotated
    from fabrics_sim.prod.kinematics import (
        Kinematics,
        extract_link_index_path,
        extract_link_paths,
        isaac_pose_to_warp_configuration,
    )
except (ImportError, AttributeError) as exc:
    RUNTIME_IMPORT_ERROR: Exception | None = exc
else:
    RUNTIME_IMPORT_ERROR = None


class RuntimeTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if RUNTIME_IMPORT_ERROR is not None:
            raise unittest.SkipTest(
                f"fabrics runtime dependencies are unavailable: {RUNTIME_IMPORT_ERROR}"
            )
        wp.init()


class TestG1Assets(unittest.TestCase):
    def test_g1_model_files_exist(self) -> None:
        self.assertTrue(G1_URDF.is_file(), G1_URDF)
        self.assertTrue(G1_USD.is_file(), G1_USD)
        self.assertGreater(G1_URDF.stat().st_size, 0)
        self.assertGreater(G1_USD.stat().st_size, 0)
        self.assertFalse(
            is_git_lfs_pointer(G1_USD),
            f"{G1_USD} is only a Git LFS pointer; download the real LFS asset first",
        )
        print(f"[G1] URDF found: {G1_URDF}")
        print(f"[G1] Isaac Sim USD found: {G1_USD}")


class TestKinematicsHelpers(RuntimeTestCase):
    def test_link_paths(self) -> None:
        parents = np.array([-1, 0, 1, 1, 3], dtype=np.int32)
        self.assertEqual(extract_link_index_path(4, parents), [0, 1, 3, 4])
        self.assertEqual(
            extract_link_paths(parents),
            [[0], [0, 1], [0, 1, 2], [0, 1, 3], [0, 1, 3, 4]],
        )

    def test_g1_forward_kinematics_and_jacobians_on_cpu(self) -> None:
        self.assertTrue(G1_URDF.is_file(), G1_URDF)
        batch_size = 2
        kinematics = Kinematics(
            str(G1_URDF), batch_size, thread_across_links=True, device="cpu"
        )
        self.assertEqual(kinematics.cspace_dim, 29)

        q = wp.zeros((batch_size, kinematics.cspace_dim), dtype=float, device="cpu")
        kinematics.eval(q, jacobians=True)

        transforms = kinematics.batch_link_transforms.numpy()
        jacobians = kinematics.batch_link_jacobians.numpy()
        self.assertEqual(transforms.shape, (batch_size, kinematics.num_links, 7))
        self.assertEqual(
            jacobians.shape,
            (batch_size, kinematics.num_links, kinematics.cspace_dim, 3),
        )
        self.assertTrue(np.isfinite(transforms).all())
        self.assertTrue(np.isfinite(jacobians).all())

        # Both kernel implementations should produce the same zero-pose FK.
        multithreaded = transforms.copy()
        kinematics.eval(q, thread_across_links=False)
        np.testing.assert_allclose(
            kinematics.batch_link_transforms.numpy(), multithreaded, atol=1e-6
        )
        print(
            f"[G1] Kinematics OK: {kinematics.cspace_dim} DoF, "
            f"{kinematics.num_links} links"
        )

    def test_g1_floating_base_pelvis_world_pose(self) -> None:
        kinematics = Kinematics(
            str(G1_URDF),
            batch_size=1,
            thread_across_links=True,
            device="cpu",
            floating_base=True,
        )
        self.assertEqual(kinematics.configuration_dim, 36)
        self.assertEqual(kinematics.velocity_dim, 35)

        # Isaac: xyz + quaternion wxyz. This is a 45-degree yaw.
        isaac_pose = torch.tensor(
            [[1.0, 2.0, 0.8, 0.9238795, 0.0, 0.0, 0.3826834]],
            dtype=torch.float32,
        )
        warp_base_q = isaac_pose_to_warp_configuration(isaac_pose)
        self.assertTrue(
            torch.allclose(
                warp_base_q,
                torch.tensor(
                    [[1.0, 2.0, 0.8, 0.0, 0.0, 0.3826834, 0.9238795]]
                ),
            )
        )

        q = torch.zeros((1, 36), dtype=torch.float32)
        q[:, :7] = warp_base_q
        kinematics.eval(wp.torch.from_torch(q), jacobians=True)

        pelvis_index = kinematics.get_link_index("pelvis")
        pelvis_world_pose = kinematics.batch_link_transforms.numpy()[0, pelvis_index]
        np.testing.assert_allclose(pelvis_world_pose[:3], [1.0, 2.0, 0.8], atol=1e-6)
        np.testing.assert_allclose(
            pelvis_world_pose[3:], [0.0, 0.0, 0.3826834, 0.9238795], atol=1e-6
        )
        pelvis_jacobian = kinematics.batch_link_jacobians.numpy()[0, pelvis_index]
        np.testing.assert_allclose(pelvis_jacobian[:3], np.eye(3), atol=1e-6)
        np.testing.assert_allclose(pelvis_jacobian[3:6], 0.0, atol=1e-6)
        print(
            "[G1] Floating pelvis world pose OK: "
            f"position={pelvis_world_pose[:3]}, quaternion_xyzw={pelvis_world_pose[3:]}"
        )

        joint_names, child_link_names, joint_world_poses = (
            kinematics.get_joint_world_poses()
        )
        self.assertEqual(len(joint_names), 30)  # floating base + 29 actuated joints
        self.assertEqual(joint_world_poses.shape, (1, 30, 7))
        print("[G1] All joint world poses (quaternion order: xyzw):")
        for joint_name, link_name, pose in zip(
            joint_names, child_link_names, joint_world_poses[0].cpu().numpy()
        ):
            print(
                f"  {joint_name} -> {link_name}: "
                f"position={pose[:3].tolist()}, quaternion_xyzw={pose[3:].tolist()}"
            )


class TestImportUrdf(RuntimeTestCase):
    def test_g1_annotated_import(self) -> None:
        self.assertTrue(G1_URDF.is_file(), G1_URDF)
        builder = wp.sim.ModelBuilder()
        info = parse_urdf_annotated(
            str(G1_URDF),
            builder,
            wp.transform_identity(),
            floating=True,
            include_collisions=False,
        )

        self.assertEqual(len(info.cspace_names), 29)
        self.assertEqual(len(info.cspace2link), 29)
        self.assertEqual(len(info.cspace_joint_limits), 29)
        self.assertEqual(len(info.cspace_name2index_map), 29)
        self.assertEqual(len(set(info.link_index_map.values())), len(info.link_index_map))
        for index, joint_name in enumerate(info.cspace_names):
            self.assertEqual(info.cspace_name2index_map[joint_name], index)

        model = builder.finalize("cpu")
        # A free joint contributes 7 configuration values (xyz + quaternion)
        # and 6 velocity values; the articulated G1 joints contribute 29.
        self.assertEqual(len(model.joint_q), 36)
        self.assertEqual(len(model.joint_qd), 35)
        print(
            f"[G1] Floating-base URDF imported: {G1_URDF} "
            f"(q=36, qd=35, actuated joints={len(info.cspace_names)})"
        )

class TestCudaStreamUtils(RuntimeTestCase):
    def tearDown(self) -> None:
        cuda_stream_utils.is_cuda_using_warp_streams = False

    def test_setup_is_safe_without_cuda_and_idempotent(self) -> None:
        cuda_stream_utils.is_cuda_using_warp_streams = False
        with mock.patch.object(torch.cuda, "device_count", return_value=0):
            cuda_stream_utils.setup_torch_to_use_warp_streams(verbose=False)
            self.assertTrue(cuda_stream_utils.is_cuda_using_warp_streams)
            cuda_stream_utils.setup_torch_to_use_warp_streams(verbose=False)
            self.assertTrue(cuda_stream_utils.is_cuda_using_warp_streams)

    def test_set_default_torch_device_requests_cuda_float(self) -> None:
        with mock.patch.object(torch, "set_default_tensor_type") as set_default:
            cuda_stream_utils.set_default_torch_device_to_cuda(verbose=False)
        set_default.assert_called_once_with("torch.cuda.FloatTensor")

    @unittest.skipUnless(
        RUNTIME_IMPORT_ERROR is None and torch.cuda.is_available(),
        "CUDA is not available",
    )
    def test_torch_and_warp_share_cuda_stream(self) -> None:
        cuda_stream_utils.is_cuda_using_warp_streams = False
        cuda_stream_utils.setup_torch_to_use_warp_streams(verbose=False)
        torch_stream = torch.cuda.current_stream().cuda_stream
        warp_stream = wp.get_cuda_device().stream.cuda_stream
        self.assertEqual(torch_stream, warp_stream)


def visualize_g1() -> None:
    """Open one G1 articulation in Isaac Sim and render until its window closes."""
    if not G1_USD.is_file():
        raise FileNotFoundError(f"G1 USD was not found: {G1_USD}")
    if is_git_lfs_pointer(G1_USD):
        raise RuntimeError(
            f"G1 USD is a Git LFS pointer, not a loadable USD: {G1_USD}. "
            "Run 'git lfs pull' in the repository from which the G1 assets were "
            "obtained, then copy/checkout the real USD files before visualizing."
        )
    if not G1_VISUALIZATION_URDF.is_file():
        raise FileNotFoundError(
            f"Matching G1 visualization URDF was not found: {G1_VISUALIZATION_URDF}"
        )

    # Isaac Sim only needs the USD. Read the matching joint names with the
    # standard library so a broken optional urdfpy install cannot block USD
    # visualization.
    urdf_root = ET.parse(G1_VISUALIZATION_URDF).getroot()
    joint_names = [
        joint.attrib["name"]
        for joint in urdf_root.findall("joint")
        if joint.attrib.get("type") in {"revolute", "continuous", "prismatic"}
    ]
    if len(joint_names) != 29:
        raise RuntimeError(
            f"Expected 29 movable G1 joints, found {len(joint_names)} in "
            f"{G1_VISUALIZATION_URDF}"
        )

    from fabrics_sim.visualization.robot_visualizer import RobotVisualizer

    print(f"[G1] Importing into Isaac Sim: {G1_USD}")
    visualizer = RobotVisualizer(
        robot_dir_name="g1_sim/29dof/usd/g1_29dof_rev_1_0",
        robot_name="g1_29dof_rev_1_0",
        batch_size=1,
        device="cuda:0",
        robot_body_sphere_radii=None,
        robot_body_sphere_position=None,
        world_model=None,
        vertical_offset=0.0,
        fabric_joint_names=joint_names,
    )
    joint_positions = np.zeros((1, len(joint_names)), dtype=np.float32)
    joint_velocities = np.zeros_like(joint_positions)

    print("[G1] Visualization ready; close the Isaac Sim window to exit.")
    try:
        while visualizer.simulation_app.is_running():
            visualizer.render(joint_positions, joint_velocities, None, None)
    except KeyboardInterrupt:
        print("[G1] Visualization interrupted.")
    finally:
        visualizer.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate and visualize the G1 model")
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="run tests, then open the G1 USD in Isaac Sim",
    )
    args = parser.parse_args()

    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        return 1
    if args.visualize:
        visualize_g1()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
