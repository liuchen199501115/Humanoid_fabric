#!/usr/bin/env python3
"""Add CoRe-style left/right contact sequences to robot-motion PKLs.

Only dictionaries containing ``root_pos``, ``root_rot`` and ``dof_pos`` are
treated as robot motions. Other pickle files (for example SMPL-X models) are
left untouched. Writes use a temporary file followed by an atomic replace.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import pickle
import sys
import tempfile

import numpy as np


RETARGET_ROOT = Path(__file__).resolve().parents[1]
if str(RETARGET_ROOT) not in sys.path:
    sys.path.insert(0, str(RETARGET_ROOT))

from retargeting.export import core_contact_sequence  # noqa: E402
from retargeting.urdf import load_urdf_spec  # noqa: E402


MOTION_KEYS = {"root_pos", "root_rot", "dof_pos"}
CONTACT_NAMES = ["left_foot", "right_foot"]


def quaternion_xyzw_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    if quaternion.ndim != 2 or quaternion.shape[1] != 4:
        raise ValueError(f"root_rot must have shape (T,4), got {quaternion.shape}")
    norm = np.linalg.norm(quaternion, axis=1, keepdims=True)
    if np.any(norm < 1.0e-8):
        raise ValueError("root_rot contains a zero quaternion")
    x, y, z, w = (quaternion / norm).T
    return np.stack(
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - w * z),
            2.0 * (x * z + w * y),
            2.0 * (x * y + w * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - w * x),
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
            1.0 - 2.0 * (x * x + y * y),
        ),
        axis=1,
    ).reshape(-1, 3, 3)


def motion_dofs_in_urdf_order(motion: dict, expected_names: tuple[str, ...]):
    dof_pos = np.asarray(motion["dof_pos"], dtype=np.float64)
    if dof_pos.ndim != 2:
        raise ValueError(f"dof_pos must have shape (T,D), got {dof_pos.shape}")
    names = motion.get("dof_names")
    if names is None:
        if dof_pos.shape[1] != len(expected_names):
            raise ValueError(
                f"unnamed dof_pos has {dof_pos.shape[1]} columns; "
                f"expected {len(expected_names)}"
            )
        return dof_pos
    names = [str(name) for name in names]
    missing = sorted(set(expected_names) - set(names))
    if missing:
        raise ValueError(f"dof_names is missing G1 joints: {missing}")
    return dof_pos[:, [names.index(name) for name in expected_names]]


def add_contact_sequence(
    path: Path,
    spec,
    *,
    velocity_threshold: float,
    minimum_duration: float,
    smoothing_duration: float,
    height_threshold: float,
    support_fallback_height: float,
    overwrite: bool,
    dry_run: bool,
) -> str:
    with path.open("rb") as handle:
        motion = pickle.load(handle)
    if not isinstance(motion, dict) or not MOTION_KEYS.issubset(motion):
        return "non_motion"

    root_pos = np.asarray(motion["root_pos"], dtype=np.float64)
    root_rot = np.asarray(motion["root_rot"], dtype=np.float64)
    dof_pos = motion_dofs_in_urdf_order(motion, spec.dof_names)
    if root_pos.ndim != 2 or root_pos.shape[1] != 3:
        raise ValueError(f"root_pos must have shape (T,3), got {root_pos.shape}")
    frame_count = root_pos.shape[0]
    if root_rot.shape != (frame_count, 4) or dof_pos.shape[0] != frame_count:
        raise ValueError(
            "root_pos, root_rot and dof_pos frame counts do not agree: "
            f"{root_pos.shape}, {root_rot.shape}, {dof_pos.shape}"
        )

    existing = motion.get("contact_sequence")
    existing_names = motion.get("contact_names")
    if (
        not overwrite
        and existing is not None
        and np.asarray(existing).shape == (frame_count, 2)
        and list(existing_names or CONTACT_NAMES) == CONTACT_NAMES
    ):
        return "already_present"

    fps = float(motion.get("fps", 30.0))
    sequence = core_contact_sequence(
        root_pos,
        quaternion_xyzw_to_matrix(root_rot),
        dof_pos,
        spec,
        fps,
        velocity_threshold=velocity_threshold,
        minimum_duration=minimum_duration,
        smoothing_duration=smoothing_duration,
        height_threshold=height_threshold,
        support_fallback_height=support_fallback_height,
    )
    if sequence.shape != (frame_count, 2) or sequence.dtype != np.bool_:
        raise RuntimeError(
            f"invalid generated contact sequence {sequence.shape}/{sequence.dtype}"
        )
    if dry_run:
        return "would_update"

    motion["contact_sequence"] = sequence
    motion["contact_names"] = CONTACT_NAMES.copy()
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{path.name}.", suffix=".tmp",
            dir=path.parent, delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            pickle.dump(motion, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return "updated"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths", nargs="*", type=Path,
        default=[RETARGET_ROOT / "output", RETARGET_ROOT / "outputs"],
        help="PKL files or directories to scan recursively.",
    )
    parser.add_argument(
        "--urdf", type=Path,
        default=RETARGET_ROOT / "assets/g1_description/g1_29dof.urdf",
    )
    parser.add_argument("--velocity-threshold", type=float, default=0.08)
    parser.add_argument("--minimum-duration", type=float, default=0.10)
    parser.add_argument("--smoothing-duration", type=float, default=0.10)
    parser.add_argument("--height-threshold", type=float, default=0.04)
    parser.add_argument(
        "--support-fallback-height", type=float, default=0.08
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.velocity_threshold <= 0.0:
        raise ValueError("--velocity-threshold must be positive")
    if args.minimum_duration < 0.0 or args.smoothing_duration < 0.0:
        raise ValueError("duration arguments must be nonnegative")
    if args.height_threshold <= 0.0:
        raise ValueError("--height-threshold must be positive")
    if args.support_fallback_height < args.height_threshold:
        raise ValueError(
            "--support-fallback-height must be at least --height-threshold"
        )

    files = set()
    for path in args.paths:
        if path.is_file():
            files.add(path.resolve())
        elif path.is_dir():
            files.update(item.resolve() for item in path.rglob("*.pkl"))
        else:
            raise FileNotFoundError(path)

    spec = load_urdf_spec(args.urdf)
    counts = {
        "updated": 0,
        "would_update": 0,
        "already_present": 0,
        "non_motion": 0,
        "failed": 0,
    }
    failures = []
    for index, path in enumerate(sorted(files), start=1):
        try:
            status = add_contact_sequence(
                path,
                spec,
                velocity_threshold=args.velocity_threshold,
                minimum_duration=args.minimum_duration,
                smoothing_duration=args.smoothing_duration,
                height_threshold=args.height_threshold,
                support_fallback_height=args.support_fallback_height,
                overwrite=args.overwrite,
                dry_run=args.dry_run,
            )
            counts[status] += 1
        except Exception as error:  # Continue so every bad path is reported.
            counts["failed"] += 1
            failures.append((path, error))
        if index % 100 == 0 or index == len(files):
            print(f"processed {index}/{len(files)}", flush=True)

    print("summary: " + ", ".join(f"{key}={value}" for key, value in counts.items()))
    for path, error in failures:
        print(f"[failed] {path}: {error}", file=sys.stderr)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
