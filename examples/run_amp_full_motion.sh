#!/usr/bin/env bash
# Entire 2000-frame AMP clip; terminal partial swing is not a stationary stop.
set -euo pipefail
cd "$(dirname "$0")/.."
exec bash examples/run_amp_four_steps.sh --tracking-steps 0 --track-full-motion \
  --duration 10.1 --output-path acc_data/amp_walk_20261102/full_clip/full_motion.npz "$@"
