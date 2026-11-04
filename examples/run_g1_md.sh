#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Shared PKL preprocessing/contact setup; g1.md selects a different command path.
exec bash examples/run_walk1_four_steps.sh --g1-md \
  --output-path acc_data/walk1_g1_md.npz "$@"
