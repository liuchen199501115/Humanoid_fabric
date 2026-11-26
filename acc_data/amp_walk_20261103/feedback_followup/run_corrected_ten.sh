#!/usr/bin/env bash
# Ten verified original-source landings; continue to the full source endpoint or a failed validation.
set -euo pipefail
cd /home/sky/Humanoid_fabric
exec bash acc_data/amp_walk_20261103/feedback_followup/run_corrected_six.sh \
 --anticipate-unload-hysteresis --anticipate-unload-start 3.7 --anticipate-unload-horizon .1 \
 --late-unload-start 3.7 --late-unload-ramp .02 --late-unload-weight .2 \
 --reference-pose-candidate acc_data/amp_walk_20261103/feedback_followup/tenth_landing_forward15_right20.npz \
 --output-path acc_data/amp_walk_20261103/feedback_followup/corrected_ten.npz "$@"
