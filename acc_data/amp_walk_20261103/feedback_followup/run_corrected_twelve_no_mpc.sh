#!/usr/bin/env bash
# Twelve source landings verified. Full tracking is still under development.
set -euo pipefail
cd /home/sky/Humanoid_fabric
exec bash acc_data/amp_walk_20261103/feedback_followup/run_corrected_eleven_recovery.sh \
  --convex-wrist-collision --convex-wrist-collision-start 3.5 \
  --model-effort-start 4.15 --tracking-margin-start 4.4 \
  --late-foot-metric-start 4.5 --late-foot-metric-scale 4 \
  --reference-pose-candidate acc_data/amp_walk_20261103/feedback_followup/wbc_convex_catch15_25_clear35.npz \
  --output-path acc_data/amp_walk_20261103/feedback_followup/corrected_twelve_no_mpc.npz "$@" \
  --direct-wbc --full-body-preview-start inf --centroidal-preview-start inf
