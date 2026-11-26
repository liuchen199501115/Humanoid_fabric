#!/usr/bin/env bash
# Experimental: five verified landings; full source tracking is not yet achieved.
set -euo pipefail
cd /home/sky/Humanoid_fabric
exec bash acc_data/amp_walk_20261103/feedback_followup/run_asymmetric_feedback.sh \
  --foot-feedback-starts 1.3 .88 --foot-feedback-limits 150 60 \
  --foot-feedback-joints leg ankles --flight-feedback-ramp .04 \
  --vertical-feedback-start 1.55 "$@"
