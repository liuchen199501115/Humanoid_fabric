#!/usr/bin/env bash
# Repeatedly verified seven source landings with one inverse-dynamics QP.
# No MPC. Original PKL and all physical/fidelity limits are retained.
# Attempts the whole clip and stops on the unchanged original-PKL audit.
set -euo pipefail
exec bash "$(dirname "$0")/run_amp_full_motion.sh" \
  --fabric-priority-mode none --fabric-tracking-feedback \
  --fabric-secondary-scale 0 --fabric-root-regularization 1e-10 \
  --qp-task-only-objective --posture-metric-scale 1 \
  --source-pelvis-weight 1 --joint-feedback-limit 30 \
  --reference-pose-candidate acc_data/g1_tracking_fix_20261123/landing15.npz \
  --contact-pause-start 2.55 --contact-pause-end 2.8 \
  --contact-pause-landings 7 --contact-pause-resume .3 \
  --qp-single-proximal .0001 --qp-single-proximal-start 2.5 \
  --output-path acc_data/g1_tracking_fix_20261123/continued_tracking.npz "$@"
