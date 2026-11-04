#!/usr/bin/env python3
from pathlib import Path
import argparse

from retargeting.inference import run_retarget


DEFAULT_MOTION_ID = "cmu_35_17_run_jog"


def parse_args(argv=None):
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Standalone G1 robot motion retargeting inference.")
    parser.add_argument(
        "--input-h5",
        default=root / f"examples/motions/{DEFAULT_MOTION_ID}.h5",
        help="Human/source motion h5.",
    )
    parser.add_argument("--target-h5", default=root / "examples/g1_reference.h5", help="Robot/target skeleton reference h5.")
    parser.add_argument("--checkpoint-dir", default=root / "checkpoints/g1_retarget", help="Checkpoint directory.")
    parser.add_argument("--stats-path", default=None, help="Optional normalization stats npz.")
    parser.add_argument("--urdf", default=root / "assets/g1_description/g1_29dof.urdf", help="Robot URDF for exporting dof_pos.")
    parser.add_argument("--output-dir", default=root / "outputs", help="Output directory.")
    parser.add_argument("--tag", default=DEFAULT_MOTION_ID, help="Output file tag.")
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Override output motion FPS (default: input h5 fps attribute, then 30 for legacy files).",
    )
    parser.add_argument("--device", default="auto", help="'auto', 'cuda', 'cuda:0' or 'cpu'.")
    parser.add_argument("--no-bvh", action="store_true", help="Only export robot pkl.")
    parser.add_argument(
        "--no-clamp-limits",
        action="store_true",
        help="Keep raw projected joint angles instead of clamping to URDF limits.",
    )
    parser.add_argument(
        "--contact-velocity-threshold", type=float, default=0.08,
        help="CoRe toe vertical-speed threshold in m/s (default: 0.08).",
    )
    parser.add_argument(
        "--contact-minimum-duration", type=float, default=0.10,
        help="Minimum CoRe contact segment duration in seconds.",
    )
    parser.add_argument(
        "--contact-smoothing-duration", type=float, default=0.10,
        help="Toe-height moving-average window in seconds.",
    )
    parser.add_argument(
        "--contact-height-threshold", type=float, default=0.04,
        help="Maximum sole height in metres for direct contact classification.",
    )
    parser.add_argument(
        "--contact-support-fallback-height", type=float, default=0.08,
        help="Keep the lower foot as support below this sole height.",
    )
    return parser.parse_args(argv)


def main():
    args = parse_args()
    if args.contact_velocity_threshold <= 0.0:
        raise ValueError("--contact-velocity-threshold must be positive.")
    if args.contact_minimum_duration < 0.0:
        raise ValueError("--contact-minimum-duration must be nonnegative.")
    if args.contact_smoothing_duration < 0.0:
        raise ValueError("--contact-smoothing-duration must be nonnegative.")
    if args.contact_height_threshold <= 0.0:
        raise ValueError("--contact-height-threshold must be positive.")
    if (
        args.contact_support_fallback_height
        < args.contact_height_threshold
    ):
        raise ValueError(
            "--contact-support-fallback-height must be at least "
            "--contact-height-threshold."
        )
    outputs = run_retarget(
        input_h5=args.input_h5,
        target_h5=args.target_h5,
        checkpoint_dir=args.checkpoint_dir,
        stats_path=args.stats_path,
        output_dir=args.output_dir,
        urdf_path=args.urdf,
        tag=args.tag,
        fps=args.fps,
        device=args.device,
        export_bvh=not args.no_bvh,
        clamp_limits=not args.no_clamp_limits,
        contact_velocity_threshold=args.contact_velocity_threshold,
        contact_minimum_duration=args.contact_minimum_duration,
        contact_smoothing_duration=args.contact_smoothing_duration,
        contact_height_threshold=args.contact_height_threshold,
        contact_support_fallback_height=(
            args.contact_support_fallback_height
        ),
    )
    print(f"[done] fps: {outputs['fps']:g}")
    print(f"[done] pkl: {outputs['pkl']}")
    if outputs["bvh"] is not None:
        print(f"[done] bvh: {outputs['bvh']}")


if __name__ == "__main__":
    main()
