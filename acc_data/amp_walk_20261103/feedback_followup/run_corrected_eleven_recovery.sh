#!/usr/bin/env bash
# Fabric + direct WBC only; eleven verified landings, full-source audit pending.
# Keep preview disabled after caller arguments to enforce the no-MPC profile.
set -euo pipefail
cd /home/sky/Humanoid_fabric
exec bash acc_data/amp_walk_20261103/feedback_followup/run_corrected_eleven.sh \
 --anticipate-unload-overdue \
 --reference-pose-candidate acc_data/amp_walk_20261103/feedback_followup/turn_recovery_y40.npz \
 --output-path acc_data/amp_walk_20261103/feedback_followup/corrected_eleven_recovery.npz "$@" \
 --full-body-preview-start inf --centroidal-preview-start inf
