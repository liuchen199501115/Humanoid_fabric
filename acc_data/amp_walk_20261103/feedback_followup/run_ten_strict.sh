#!/usr/bin/env bash
# Ten verified source landings; the complete 2000-frame clip remains incomplete.
set -euo pipefail
cd /home/sky/Humanoid_fabric
exec bash acc_data/amp_walk_20261103/feedback_followup/run_nine_strict.sh \
 --late-unload-start 3.84 --late-unload-ramp .01 --late-unload-weight .2 \
 --landing-gated-unload-start 3.84 --anticipate-unload-start 3.84 \
 --gate-contact-transitions --horizontal-feedback-start 3.84 \
 --horizontal-feedback-limit 80 --horizontal-feedback-metric-scale 4 \
 --full-swing-pose-start 3.84 --relax-joint-tube \
 --reference-pose-candidate acc_data/amp_walk_20261103/feedback_followup/balance_y004.npz \
 --joint-correction-overlay acc_data/amp_walk_20261103/feedback_followup/tenth_tail5_y004_bounded.npz \
 --secondary-joint-correction-overlay acc_data/amp_walk_20261103/feedback_followup/ninth_plant20.npz \
 --duration 10.1 --stop-at-fidelity-loss \
 --output-path acc_data/amp_walk_20261103/feedback_followup/ten_strict.npz "$@"
