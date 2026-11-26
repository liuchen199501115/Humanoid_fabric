#!/usr/bin/env bash
# Nine verified source landings; full tracking remains incomplete.
set -euo pipefail
cd /home/sky/Humanoid_fabric
exec bash acc_data/amp_walk_20261103/feedback_followup/run_eight_strict.sh \
 --smooth-stride 6 --tube-smooth-stride 2 "$@"
