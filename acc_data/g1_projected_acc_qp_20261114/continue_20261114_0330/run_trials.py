"""Run isolated continuations with bounded runtime and initialization traces."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
TRIALS = {'baseline_traced': [],
          'com100': ['--com-acceleration-preservation-weight', '100'],
          'com1000': ['--com-acceleration-preservation-weight', '1000']}
BOOTSTRAP = "import faulthandler,runpy; faulthandler.dump_traceback_later(60,repeat=True); runpy.run_path('examples/test_g1.py',run_name='__main__')"

for name in sys.argv[1:]:
    if (OUT / (name + '.command.json')).exists():
        raise FileExistsError('Refusing to overwrite trial ' + name)
    options = TRIALS[name]
    command = [sys.executable, '-u', '-c', BOOTSTRAP,
        '--tracking-profile', 'amp-full', '--output-path', str(OUT / (name + '.npz')),
        *options, '--direct-wbc', '--tracking-steps', '0', '--track-full-motion',
        '--stop-at-fidelity-loss', '--full-body-preview-start', 'inf', '--centroidal-preview-start', 'inf']
    (OUT / (name + '.command.json')).write_text(json.dumps(command, indent=2) + '\n')
    sources = ['examples/test_g1.py', 'examples/g1_acceleration_cost.py',
               'src/fabrics_sim/solver/dynamic_consistent_gpu.py',
               'src/fabrics_sim/solver/daqp_backend.py']
    (OUT / (name + '.sources.json')).write_text(json.dumps({p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources}, indent=2) + '\n')
    environment = dict(os.environ, PYTHONPATH='src:examples:.runtime/qp_solvers',
                       OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    timed_out = False
    with (OUT / (name + '.log')).open('w') as log:
        proc = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log,
                                stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = proc.wait(timeout=240)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                code = proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                code = proc.wait()
    row = dict(name=name, options=options, returncode=code, timed_out=timed_out,
               tracking_evaluated=False)
    for suffix in ('steps', 'tracking'):
        path = OUT / (name + '.' + suffix + '.json')
        if path.exists():
            row[suffix] = json.loads(path.read_text())
            row['tracking_evaluated'] = True
    (OUT / (name + '.result.json')).write_text(json.dumps(row, indent=2) + '\n')
    print(json.dumps({k:v for k,v in row.items() if k not in ('steps','tracking')}), flush=True)
    if timed_out and not row['tracking_evaluated']:
        break  # Diagnose initialization before attempting more candidates.
