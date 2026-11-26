#!/usr/bin/env bash
# Verified eleven source landings; attempts all 2000 frames with unchanged audits.
# One native QP per update, no MPC or added PD. Not full tracking yet.
# The 50000-iteration late budget is for offline experiments, not a 500 Hz claim.
set -euo pipefail
exec bash "$(dirname "$0")/run_amp_single_tracking.sh" \
  --reference-pose-candidate acc_data/g1_tracking_complete_20261123/prepared_rooty_clearance10_lift10.npz \
  --root-position-task-start 2.5 \
  --root-position-task-ramp .5 \
  --root-position-task-reference source \
  --root-position-task-axes y \
  --root-position-task-weight 1000 \
  --com-feedback-start 3.2 \
  --com-feedback-limit 20 \
  --com-feedback-gain-scale 1.25 \
  --minimum-support-acceleration-start 3.9 \
  --minimum-support-acceleration-metric-scale 4 \
  --com-upward-speed-limit .2 \
  --com-upward-speed-start 4.08 \
  --waist-feedback-start 3.7 \
  --waist-feedback-limit 150 \
  --late-foot-angular-metric-start 3.8 \
  --late-foot-angular-metric-end 4.1 \
  --late-foot-angular-metric-side right \
  --late-foot-angular-metric-scale 4 \
  --late-foot-angular-feedback-limit 100 \
  --late-foot-linear-feedback-start 4.08 \
  --late-foot-linear-feedback-side left \
  --late-foot-linear-feedback-limit 60 \
  --synchronous-effort-budget-start 4.05 \
  --observed-joint-stop-start 4.08 \
  --clock-fourth-start 3.81 \
  --clock-fourth-ramp .12 \
  --clock-fourth-speed 1.4 \
  --qp-late-equilibration-start 3.806 \
  --qp-late-equilibration-cap 1 \
  --qp-late-iteration-start 3.806 \
  --qp-late-iterations 50000 \
  --contact-pause-start 4.3 \
  --contact-pause-end 4.5 \
  --contact-pause-landings 11 \
  --contact-pause-resume .1 \
  --contact-pause-early-resume \
  --output-path acc_data/g1_tracking_complete_20261123/continued_tracking_11step.npz "$@"
