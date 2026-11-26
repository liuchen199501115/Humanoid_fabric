#!/usr/bin/env bash
# Nine verified original-source landings; continue to the full source endpoint or a failed validation.
set -euo pipefail
cd /home/sky/Humanoid_fabric
exec bash acc_data/amp_walk_20261103/feedback_followup/run_corrected_six.sh \
 --reference-pose-candidate acc_data/amp_walk_20261103/feedback_followup/sixth_plant_roll_y30.npz \
 --output-path acc_data/amp_walk_20261103/feedback_followup/corrected_nine.npz "$@"
