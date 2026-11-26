#!/usr/bin/env bash
# Seven verified source landings; full tracking remains incomplete.
set -euo pipefail
cd /home/sky/Humanoid_fabric
exec bash acc_data/amp_walk_20261103/feedback_followup/run_tracking_tube.sh \
 --foot-feedback-starts 1.3 .88 --foot-feedback-limits 150 60 \
 --foot-feedback-joints leg ankles --flight-feedback-ramp .04 \
 --vertical-feedback-start 1.55 --waist-feedback-start 100 \
 --late-leg-feedback-start 100 --tracking-tube-bandwidth 10 "$@"
