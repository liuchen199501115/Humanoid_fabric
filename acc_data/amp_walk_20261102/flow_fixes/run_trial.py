"""Reproducible AMP controller comparison; original PKL and physical limits retained."""
import os,shlex,subprocess,sys
from pathlib import Path
root=Path('/home/sky/Humanoid_fabric');os.chdir(root)
text=(root/'acc_data/amp_walk_20261102/run_dynamic_wbc.sh').read_text().split('exec ',1)[1]
args=shlex.split(text.replace('\\\n',' ').replace('"$@"',''))
remove={'--balance-priority','--whole-body-posture-priority','--separate-balance-priority','--posture-last-priority'}
args=[a for a in args if a not in remove]
env=dict(os.environ,PYTHONPATH='src:examples:.runtime/qp_solvers',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
raise SystemExit(subprocess.call(args+sys.argv[1:],env=env))
