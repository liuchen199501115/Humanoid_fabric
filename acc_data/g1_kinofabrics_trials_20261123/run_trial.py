"""Run one isolated full-source trial; never overwrite prior evidence."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
name, *options = sys.argv[1:]
path = OUT / (name + '.npz')
command = [sys.executable, '-u', 'examples/test_g1.py', '--tracking-profile',
           'amp-full', '--output-path', str(path), *options]
with path.with_suffix('.command.json').open('x') as stream:
    json.dump(command, stream, indent=2)
sources = ['examples/test_g1.py', 'src/fabrics_sim/solver/daqp_backend.py',
           'src/fabrics_sim/fabrics/dynamics_aware.py',
           'src/fabrics_sim/fabrics/g1_pose_fabric.py',
           'src/fabrics_sim/solver/active_set_predictor.py',
           'examples/g1_contact_reference.py']
path.with_suffix('.sources.json').write_text(json.dumps({s: hashlib.sha256(
    (ROOT / s).read_bytes()).hexdigest() for s in sources}, indent=2))
env = dict(os.environ, PYTHONPATH='src:examples:.runtime/qp_solvers',
           OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
timed_out = False
with path.with_suffix('.log').open('w') as log:
    proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                            stderr=subprocess.STDOUT, start_new_session=True)
    try:
        code = proc.wait(timeout=900)
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            code = proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            code = proc.wait()
report = dict(name=name, options=options, returncode=code, timed_out=timed_out)
for suffix in ('steps', 'tracking'):
    p = path.with_suffix('.' + suffix + '.json')
    if p.exists():
        report[suffix] = json.loads(p.read_text())
path.with_suffix('.result.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2), flush=True)
