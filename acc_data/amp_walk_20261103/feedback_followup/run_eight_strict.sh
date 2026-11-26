#!/usr/bin/env bash
# Eight verified source landings; the complete PKL has not passed yet.
set -euo pipefail
cd /home/sky/Humanoid_fabric
exec bash acc_data/amp_walk_20261103/feedback_followup/run_seven_strict.sh \
 --reference-correction-rmse-limit .08 "$@"
