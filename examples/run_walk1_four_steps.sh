#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="src:examples:.runtime/qp_solvers${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec "${G1_PYTHON:-python}" examples/test_g1.py \
  --mocap-pipeline --tracking-steps 4 --no-acceleration-and-braking-limits --joint-velocity-limits \
  --motion-pkl retarget/lafan/walk1_g1.pkl \
  --contacts-pkl retarget/lafan/walk1_g1_with_contacts.pkl \
  --device cpu --duration 120 --motion-warmup-duration 5 --motion-ramp-duration .25 \
  --control-frequency 500 --source-pelvis-weight 1 \
  --observed-contact-patches --early-touchdown --align-takeoff-frame \
  --joint-limit-avoidance-weight 20 --joint-limit-barrier-bandwidth 8 \
  --tracking-servo --contact-aware-servo --nominal-implicit-servo \
  --qp-solver daqp --wrist-hip-avoidance --wrist-hip-barrier --wrist-body-avoidance \
  --report-self-contacts --diagnose-com --motor-armature --flat-foot-spawn \
  --output-path acc_data/walk1_strict_four.npz "$@"
