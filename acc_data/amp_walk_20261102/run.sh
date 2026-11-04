#!/usr/bin/env bash
set -euo pipefail
cd /home/sky/Humanoid_fabric
export PYTHONPATH="src:examples:.runtime/qp_solvers${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec /home/sky/anaconda3/envs/humanoid_fabric/bin/python -u examples/test_g1.py \
  --g1-md --g1-com-priority --mocap-pipeline --until-motion-end \
  --motion-pkl acc_data/amp_walk_20261102/amp_walk_tracking.pkl \
  --no-acceleration-and-braking-limits --joint-velocity-limits \
  --device cpu --duration 120 --motion-warmup-duration 5 --motion-ramp-duration .25 \
  --source-pelvis-weight 1 --observed-contact-patches --early-touchdown --align-takeoff-frame \
  --joint-limit-avoidance-weight 20 --joint-limit-barrier-bandwidth 8 \
  --qp-solver daqp --wrist-hip-avoidance --wrist-hip-barrier --wrist-body-avoidance \
  --report-self-contacts --diagnose-com --diagnose-contacts --motor-armature \
  --output-path acc_data/amp_walk_20261102/tracking.npz "$@"
