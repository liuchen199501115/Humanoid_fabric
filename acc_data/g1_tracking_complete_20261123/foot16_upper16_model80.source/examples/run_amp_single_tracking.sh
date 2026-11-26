#!/usr/bin/env bash
# Repeatedly verified nine source landings and tenth-step flight with one QP.
# No MPC; attempts the whole original clip with unchanged acceptance limits.
# Uses an adapted full-body reference, soft auxiliary joint tube and QP scaling.
set -euo pipefail
exec bash "$(dirname "$0")/run_amp_full_motion.sh" \
  --fabric-priority-mode none \
  --fabric-tracking-feedback \
  --fabric-secondary-scale 0 \
  --fabric-root-regularization 1e-10 \
  --qp-task-only-objective \
  --posture-metric-scale 1 \
  --source-pelvis-weight 1 \
  --joint-feedback-limit 30 \
  --clock-second-start 2.3 \
  --clock-second-ramp .25 \
  --clock-second-speed .8 \
  --qp-single-proximal .0001 \
  --qp-single-proximal-start 2.5 \
  --qp-cold-start-time 2.5 \
  --anticipate-unload-start 2.7 \
  --anticipate-unload-horizon .12 \
  --late-unload-start 2.7 \
  --late-unload-ramp .04 \
  --late-unload-weight .2 \
  --tracking-tube-soft-weight 10 \
  --tracking-tube-soft-start 2.54 \
  --tracking-tube-soft-joint-only \
  --collision-barrier-bandwidth-start 2.0 \
  --collision-barrier-bandwidth 20 \
  --reference-pose-candidate acc_data/g1_tracking_complete_20261123/prepared_combined_root_y50.npz \
  --qp-equilibration-start 2.5 \
  --qp-equilibration-cap 100 \
  --clock-third-start 3.0 \
  --clock-third-ramp .2 \
  --clock-third-speed 1.2 \
  --support-load-floor-start 3.7 \
  --support-load-floor 100 \
  --support-load-floor-transfer \
  --output-path acc_data/g1_tracking_complete_20261123/continued_tracking.npz "$@"
