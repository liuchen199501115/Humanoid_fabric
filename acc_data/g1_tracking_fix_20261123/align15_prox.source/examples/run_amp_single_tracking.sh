#!/usr/bin/env bash
# Verified five source landings with one inverse-dynamics QP and no MPC.
# Attempts the whole clip and stops on the unchanged original-PKL audit.
set -euo pipefail
exec bash "$(dirname "$0")/run_amp_full_motion.sh" \
  --fabric-priority-mode none --fabric-tracking-feedback \
  --fabric-secondary-scale 0 --fabric-root-regularization 1e-10 \
  --qp-task-only-objective --posture-metric-scale 1 \
  --source-pelvis-weight 1 --joint-feedback-limit 30 \
  --output-path acc_data/g1_priority_search_20261123/continued_tracking.npz "$@"
