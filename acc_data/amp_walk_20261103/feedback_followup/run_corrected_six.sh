#!/usr/bin/env bash
# Six verified landings after the physical wrench fix; full tracking is still unverified.
set -euo pipefail
cd /home/sky/Humanoid_fabric
exec bash acc_data/amp_walk_20261103/feedback_followup/run_tracking_tube_clock.sh \
 --foot-feedback-starts 1.3 .88 --foot-feedback-limits 150 150 --foot-feedback-joints leg leg \
 --flight-feedback-ramp .04 --waist-feedback-start 100 --late-leg-feedback-start 100 \
 --global-posture-feedback-start .70 --global-posture-feedback-limit 150 \
 --smooth-start .65 --smooth-ramp .1 --smooth-stride 6 --tube-smooth-stride 2 \
 --reference-correction-rmse-limit .08 \
 --reference-pose-candidate acc_data/amp_walk_20261103/feedback_followup/early_flat_lower15.npz \
 --landing-gated-unload-start 1.48 --gate-contact-transitions \
 --horizontal-feedback-start 1.4 --horizontal-feedback-limit 80 --horizontal-feedback-metric-scale 4 \
 --vertical-feedback-start 1.4 --full-swing-pose-start 1.4 \
 --relax-joint-tube --recover-orientation-tube \
 --clock-transition-start .90 --clock-transition-ramp .35 --clock-final-speed 1.2 \
 --stop-at-fidelity-loss --duration 18 \
 --output-path acc_data/amp_walk_20261103/feedback_followup/corrected_six.npz "$@"
