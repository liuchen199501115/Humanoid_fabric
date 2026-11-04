#!/usr/bin/env bash
# AMP source: retarget/amp_reference/amp_walk_1ms.pkl (metadata: 200 Hz).
# Prepared input preserves source poses, with verified joint/quaternion ordering
# and inferred native-sole contact labels. Real ground forces certify support.
# This experimental profile reports landing success and strict fidelity separately.
set -euo pipefail
cd /home/sky/Humanoid_fabric
export PYTHONPATH="src:examples:.runtime/qp_solvers${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec "${G1_PYTHON:-python}" -u acc_data/amp_walk_20261102/continuation/trial_height_clearance.py \
  --direct-wbc --tracking-servo --contact-aware-servo --nominal-implicit-servo \
  --control-frequency 500 --motion-start-in-motion --track-full-motion --physics-iterations 32 8 \
  --motion-pkl acc_data/amp_walk_20261102/amp_walk_native_contacts.pkl \
  --motion-interpolation pchip-analytic --reference-mode unified \
  --motion-feedback-scale 9 --preserve-reference-acceleration \
  --balance-priority --swing-position-priority --whole-body-posture-priority \
  --dynamic-unload-weight 1 --swing-midpoint-clearance .056 \
  --observed-contact-patches --source-pelvis-weight 1 \
  --device cpu --duration 10.1 --joint-velocity-limits \
  --joint-limit-avoidance-weight 20 --joint-limit-barrier-bandwidth 8 \
  --qp-solver daqp --wrist-hip-avoidance --wrist-hip-barrier --wrist-body-avoidance \
  --report-self-contacts --diagnose-com --diagnose-contacts --motor-armature \
  --output-path acc_data/amp_walk_20261102/continuation/height_clearance.npz "$@"
