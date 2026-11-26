"""Sequential matched-execution Isaac experiments; preserve failures and commands."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
PYTHON = '/home/sky/anaconda3/envs/humanoid_fabric/bin/python'
TRIALS = [
    ('control_wbc', ['--tracking-profile', 'twelve-wbc']),
    ('formula_weights', ['--tracking-profile', 'formula-wbc', '--formula-scope', 'weights']),
    ('formula_weights12', ['--tracking-profile', 'formula-wbc', '--formula-scope', 'weights', '--formula-maximum-weight', '1.2']),
    ('formula_phase_v2', ['--tracking-profile', 'formula-wbc']),
    ('formula_source_v2', ['--tracking-profile', 'formula-wbc', '--formula-source-reference']),
]
files = ['examples/test_g1_2.py', 'examples/test_g1.py',
         'acc_data/amp_walk_20261103/feedback_followup/unload_qp.py',
         'src/fabrics_sim/solver/AccelerationQP.py']
env = os.environ.copy()
env.update(PYTHONPATH='src:examples:.runtime/qp_solvers', OMP_NUM_THREADS='1',
           OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
if sys.argv[1:]:
    requested = set(sys.argv[1:])
    unknown = requested - {name for name, _ in TRIALS}
    if unknown:
        raise SystemExit(f'Unknown trials: {unknown}')
    TRIALS = [(name, options) for name, options in TRIALS if name in requested]
for name, options in TRIALS:
    prefix = OUT / name
    manifest_path = prefix.with_suffix('.command.json')
    if manifest_path.exists():
        raise SystemExit(f'Refuse to overwrite {manifest_path}')
    command = [PYTHON, '-u', 'examples/test_g1_2.py', *options,
               '--output-path', str(prefix.with_suffix('.npz'))]
    manifest = dict(command=command, status='running', sha256={
        file: hashlib.sha256((ROOT / file).read_bytes()).hexdigest() for file in files})
    manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
    print(f'Starting {name}', flush=True)
    start = time.monotonic()
    with prefix.with_suffix('.log').open('w') as log:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    manifest.update(status='finished', returncode=result.returncode,
                    wall_seconds=time.monotonic()-start)
    manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(dict(run=name, **{k: manifest[k] for k in ('returncode', 'wall_seconds')})), flush=True)
    print('\n'.join(prefix.with_suffix('.log').read_text().splitlines()[-10:]), flush=True)
