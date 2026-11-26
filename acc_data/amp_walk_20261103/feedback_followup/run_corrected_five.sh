#!/usr/bin/env bash
# Five landings with the corrected per-foot wrench blocks; full motion is still unverified.
set -euo pipefail
cd /home/sky/Humanoid_fabric
exec bash acc_data/amp_walk_20261103/feedback_followup/run_nine_strict.sh \
 --smooth-start .65 --smooth-ramp .1 \
 --reference-pose-candidate acc_data/amp_walk_20261103/feedback_followup/early_flat_lower15.npz \
 --foot-feedback-joints leg leg --foot-feedback-limits 150 150 \
 --global-posture-feedback-start .70 --global-posture-feedback-limit 150 \
 --horizontal-feedback-start 1.4 --horizontal-feedback-limit 80 --horizontal-feedback-metric-scale 4 \
 --vertical-feedback-start 1.4 --full-swing-pose-start 1.4 \
 --landing-gated-unload-start 1.48 --gate-contact-transitions --relax-joint-tube \
 --duration 15 --stop-at-fidelity-loss \
 --output-path acc_data/amp_walk_20261103/feedback_followup/corrected_five.npz "$@"
