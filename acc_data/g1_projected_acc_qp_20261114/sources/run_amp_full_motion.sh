#!/usr/bin/env bash
# Attempt the entire original AMP clip through examples/test_g1.py.
# Project Fabric acceleration, then track it in one QP per tick.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="src:examples:.runtime/qp_solvers${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec "${G1_PYTHON:-python}" -u examples/test_g1.py --tracking-profile amp-full \
  --output-path acc_data/g1_projected_acc_qp_20261114/full_motion.npz "$@" \
  --direct-wbc --tracking-steps 0 --track-full-motion --stop-at-fidelity-loss \
  --full-body-preview-start inf --centroidal-preview-start inf
