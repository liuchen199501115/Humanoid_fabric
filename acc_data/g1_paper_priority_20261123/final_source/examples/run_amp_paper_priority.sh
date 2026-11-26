#!/usr/bin/env bash
# Paper-inspired stability-first Fabric; the original baseline remains available.
# This configuration is an experiment, not a full-PKL tracking success claim.
set -euo pipefail
exec bash "$(dirname "$0")/run_amp_full_motion.sh" \
  --fabric-priority-mode paper \
  --output-path acc_data/g1_paper_priority_20261123/continued_tracking.npz "$@"
