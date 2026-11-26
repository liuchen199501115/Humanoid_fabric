#!/usr/bin/env bash
# Eleven verified original-source landings; full-clip verification remains required.
set -euo pipefail
cd /home/sky/Humanoid_fabric
exec bash acc_data/amp_walk_20261103/feedback_followup/run_corrected_ten.sh \
 --late-body-angular-metric-start 4.2 --late-body-angular-metric-scale .1 \
 --output-path acc_data/amp_walk_20261103/feedback_followup/corrected_eleven.npz "$@"
