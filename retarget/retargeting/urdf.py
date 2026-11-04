from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


DEFAULT_FEET = ("left_ankle_roll_joint", "right_ankle_roll_joint")


@dataclass(frozen=True)
class FootCollisionSpheres:
    joint_name: str
    link_name: str
    centers: np.ndarray
    radii: np.ndarray


@dataclass(frozen=True)
class URDFKinematicSpec:
    joint_names: tuple[str, ...]
    parents: np.ndarray
    offsets: np.ndarray
    base_rotations: np.ndarray
    axes: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    foot_collisions: dict[str, FootCollisionSpheres]

    @property
    def dof_names(self):
        return self.joint_names[1:]


def rpy_to_matrix(rpy):
    roll, pitch, yaw = rpy
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def parse_urdf(urdf_path: str | Path):
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    rev_joints = [j for j in root.findall("joint") if j.get("type") == "revolute"]

    joint_names = ["pelvis"]
    joint_parents = [-1]
    joint_offsets = [np.zeros(3, dtype=np.float32)]
    base_rots = [np.eye(3, dtype=np.float64)]
    joint_axes = [np.array([0.0, 0.0, 1.0], dtype=np.float64)]
    link_to_idx = {"pelvis": 0}

    for joint in rev_joints:
        name = joint.get("name")
        parent_link = joint.find("parent").get("link")
        child_link = joint.find("child").get("link")
        origin = joint.find("origin")
        xyz = (
            np.fromstring(origin.get("xyz"), sep=" ", dtype=np.float64)
            if origin is not None and origin.get("xyz") is not None
            else np.zeros(3, dtype=np.float64)
        )
        rpy = (
            np.fromstring(origin.get("rpy"), sep=" ", dtype=np.float64)
            if origin is not None and origin.get("rpy") is not None
            else np.zeros(3, dtype=np.float64)
        )
        axis = joint.find("axis")
        axis_vec = (
            np.fromstring(axis.get("xyz"), sep=" ", dtype=np.float64)
            if axis is not None and axis.get("xyz") is not None
            else np.array([0.0, 0.0, 1.0], dtype=np.float64)
        )

        parent_idx = link_to_idx.get(parent_link, 0)
        joint_names.append(name)
        joint_parents.append(parent_idx)
        joint_offsets.append(xyz.astype(np.float32))
        base_rots.append(rpy_to_matrix(rpy))
        joint_axes.append(axis_vec)
        link_to_idx[child_link] = len(joint_names) - 1

    return (
        joint_names,
        np.asarray(joint_parents, dtype=np.int64),
        np.stack(joint_offsets),
        base_rots,
        joint_axes,
    )


def load_urdf_spec(urdf_path: str | Path) -> URDFKinematicSpec:
    """Load the G1 revolute chain, limits, and foot collision spheres."""
    joint_names, parents, offsets, base_rotations, axes = parse_urdf(urdf_path)
    root = ET.parse(urdf_path).getroot()

    limits = {}
    joint_children = {}
    for joint in root.findall("joint"):
        if joint.get("type") not in ("revolute", "continuous"):
            continue
        name = str(joint.get("name"))
        limit = joint.find("limit")
        if limit is None or limit.get("lower") is None or limit.get("upper") is None:
            raise ValueError(f"Limited range missing for URDF joint {name!r}")
        limits[name] = (float(limit.get("lower")), float(limit.get("upper")))
        child = joint.find("child")
        if child is None or child.get("link") is None:
            raise ValueError(f"URDF joint {name!r} is missing its child link")
        joint_children[name] = str(child.get("link"))

    missing_limits = [name for name in joint_names[1:] if name not in limits]
    if missing_limits:
        raise ValueError(f"URDF joints missing signed limits: {missing_limits}")

    links = {str(link.get("name")): link for link in root.findall("link")}
    foot_collisions = {}
    for joint_name in DEFAULT_FEET:
        link_name = joint_children.get(joint_name)
        if link_name is None or link_name not in links:
            raise ValueError(f"Could not resolve child link for foot joint {joint_name!r}")
        centers = []
        radii = []
        for collision in links[link_name].findall("collision"):
            geometry = collision.find("geometry")
            sphere = None if geometry is None else geometry.find("sphere")
            if sphere is None or sphere.get("radius") is None:
                continue
            origin = collision.find("origin")
            center = (
                np.fromstring(origin.get("xyz"), sep=" ", dtype=np.float64)
                if origin is not None and origin.get("xyz") is not None
                else np.zeros(3, dtype=np.float64)
            )
            if center.shape != (3,):
                raise ValueError(
                    f"Malformed collision origin for {joint_name!r}: {center}"
                )
            centers.append(center)
            radii.append(float(sphere.get("radius")))
        if not centers:
            raise ValueError(f"No collision spheres found on foot link {link_name!r}")
        foot_collisions[joint_name] = FootCollisionSpheres(
            joint_name=joint_name,
            link_name=link_name,
            centers=np.stack(centers),
            radii=np.asarray(radii, dtype=np.float64),
        )

    return URDFKinematicSpec(
        joint_names=tuple(joint_names),
        parents=np.asarray(parents, dtype=np.int64),
        offsets=np.asarray(offsets, dtype=np.float64),
        base_rotations=np.asarray(base_rotations, dtype=np.float64),
        axes=np.asarray(axes, dtype=np.float64),
        lower=np.asarray([limits[name][0] for name in joint_names[1:]], dtype=np.float64),
        upper=np.asarray([limits[name][1] for name in joint_names[1:]], dtype=np.float64),
        foot_collisions=foot_collisions,
    )
