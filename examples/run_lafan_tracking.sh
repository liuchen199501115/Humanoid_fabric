#!/usr/bin/env bash
# One q/v/a reference for all tracking tasks; full physical tracking is not certified.
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="src:.runtime/qp_solvers${PYTHONPATH:+:$PYTHONPATH}"
exec /home/sky/anaconda3/envs/humanoid_fabric/bin/python examples/test_g1.py \
  --device cpu --motion-pkl retarget/lafan/walk1_g1.pkl \
  --reference-mode unified --reference-manager --reference-footsteps source \
  --motion-interpolation pchip-analytic --motion-ramp-duration .25 \
  --direct-wbc --control-frequency 500 --source-pelvis-weight 1 \
  --observed-contact-patches --early-touchdown --align-takeoff-frame \
  --joint-limit-avoidance-weight 20 --joint-limit-barrier-bandwidth 8 \
  --tracking-servo --contact-aware-servo --nominal-implicit-servo \
  --qp-solver daqp --wrist-hip-avoidance --wrist-hip-barrier --wrist-body-avoidance \
  --report-self-contacts --diagnose-com --motor-armature \
  --duration 600 --until-motion-end "$@"
