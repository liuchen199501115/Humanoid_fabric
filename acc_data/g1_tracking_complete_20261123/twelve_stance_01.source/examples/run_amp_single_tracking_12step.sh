#!/usr/bin/env bash
# Independently reproduced twelve source landings; attempts all 2000 frames.
# One native QP per update. Full tracking is not yet achieved.
# The late 50000-iteration budget is for offline experiments, not a 500 Hz claim.
set -euo pipefail
exec bash "$(dirname "$0")/run_amp_single_tracking_11step.sh" \
  --reference-pose-candidate acc_data/g1_tracking_complete_20261123/prepared_return_source_transfer60.npz \
  --leg-posture-metric-start 4.5 \
  --leg-posture-metric-scale 64 \
  --late-upper-metric-start 4.5 \
  --late-upper-metric-scale 256 \
  --hip-yaw-barrier-start 4.5 \
  --hip-yaw-barrier-bandwidth 20 \
  --clock-fifth-start 4.42 \
  --clock-fifth-ramp .12 \
  --clock-fifth-speed 1.0 \
  --late-tube-soft-weight-start 4.5 \
  --late-tube-soft-weight 1000 \
  --torque-continuity-start 4.348 \
  --torque-continuity-ramp .06 \
  --torque-continuity-weight 1000 \
  --extra-contact-pause 4.6 4.8 12 \
  --support-orientation-priority-start 4.5 \
  --support-orientation-roll-pitch-only \
  --support-orientation-metric-scale 64 \
  --support-flatten-start 4.5 \
  --late-foot-metric-end 4.65 \
  --late-foot-metric-end-ramp .04 \
  --late-foot-metric-end-scale 4096 \
  --foot-contact-search-start 4.7 \
  --foot-contact-search-acceleration 2 \
  --output-path acc_data/g1_tracking_complete_20261123/continued_tracking_12step.npz "$@"
