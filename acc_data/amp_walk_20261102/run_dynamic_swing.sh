#!/usr/bin/env bash
set -euo pipefail
cd /home/sky/Humanoid_fabric
export PYTHONPATH="src:examples:.runtime/qp_solvers${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec /home/sky/anaconda3/envs/humanoid_fabric/bin/python -u examples/test_g1.py \
  --direct-wbc --tracking-servo --contact-aware-servo --nominal-implicit-servo --control-frequency 500 --balance-priority --swing-pose-priority --body-orientation-priority --separate-balance-priority --motion-start-in-motion \
  --motion-pkl acc_data/amp_walk_20261102/amp_walk_dynamic.pkl \
  --motion-interpolation pchip-analytic --reference-mode unified \
  --observed-contact-patches --source-pelvis-weight 1 \
  --device cpu --duration 9.996 --joint-velocity-limits \
  --joint-limit-avoidance-weight 20 --joint-limit-barrier-bandwidth 8 \
  --qp-solver daqp --wrist-hip-avoidance --wrist-hip-barrier --wrist-body-avoidance \
  --report-self-contacts --diagnose-com --diagnose-contacts --motor-armature \
  --output-path acc_data/amp_walk_20261102/dynamic_swing.npz "$@"
