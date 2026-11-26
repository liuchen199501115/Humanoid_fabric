"""Run one full-source shoulder-weight trial with immutable provenance and logs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / 'closed_loop'
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('name')
parser.add_argument('--scale', type=float, default=1.)
parser.add_argument('--start', type=float, default=3.3)
parser.add_argument('--ramp', type=float, default=.1)
parser.add_argument('--side', choices=('left', 'right', 'both'), default='both')
parser.add_argument('--directory', choices=('closed_loop','guarded','com_velocity'), default='closed_loop')
parser.add_argument('--guard', action='store_true')
parser.add_argument('--guard-relative', type=float, default=.02)
parser.add_argument('--guard-slack', type=float, default=1.)
parser.add_argument('--com-weight', type=float, default=0.)
parser.add_argument('--com-horizon', type=float, default=.1)
parser.add_argument('--com-onset', type=float, default=.5)
parser.add_argument('--com-width', type=float, default=.2)
args = parser.parse_args()
if args.com_weight and '--com-velocity-soft-weight' not in (ROOT/'examples/test_g1.py').read_text():
    parser.error('COM velocity experiment is archived; test_g1.py was restored to the baseline. See com_velocity/REPORT.md and sources/.')
OUT = Path(__file__).resolve().parent / args.directory
if not args.name.replace('_', '').isalnum():
    parser.error('Use an alphanumeric run name with optional underscores')
OUT.mkdir(parents=True, exist_ok=True)
prefix = OUT / args.name
manifest_path = prefix.with_suffix('.command.json')
if manifest_path.exists():
    raise SystemExit(f'Refuse to overwrite {manifest_path}')
command = [sys.executable, '-u', 'examples/test_g1.py', '--tracking-profile', 'amp-full',
           '--output-path', str(prefix.with_suffix('.npz')),
           '--shoulder-roll-weight-scale', str(args.scale),
           '--shoulder-roll-weight-start', str(args.start),
           '--shoulder-roll-weight-ramp', str(args.ramp),
           '--shoulder-roll-weight-side', args.side]
if args.guard:
    command += ['--shoulder-roll-guard', '--shoulder-roll-guard-relative', str(args.guard_relative),
                '--shoulder-roll-guard-slack', str(args.guard_slack)]
if args.com_weight:
    command += ['--com-velocity-soft-weight',str(args.com_weight),
                '--com-velocity-soft-horizon',str(args.com_horizon),
                '--com-velocity-soft-onset',str(args.com_onset),
                '--com-velocity-soft-width',str(args.com_width)]
for t in (3.48, 3.58, 3.65, 3.75):
    command += ['--diagnose-qp-time', str(t)]
files = ['examples/test_g1.py', 'examples/g1_shoulder_roll.py', 'examples/g1_guarded_shoulder.py',
         'src/fabrics_sim/solver/AccelerationQP.py',
         'examples/g1_com_velocity_soft.py',
         'acc_data/amp_walk_20261103/feedback_followup/unload_qp.py',
         'acc_data/amp_walk_20261103/feedback_followup/tube_recovery.py',
         'retarget/amp_reference/amp_walk_1ms.pkl',
         'acc_data/amp_walk_20261102/amp_walk_native_contacts.pkl',
         'acc_data/amp_walk_20261103/feedback_followup/wbc_convex_catch15_25_clear35.npz']
hashes = {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in files}
snapshots = OUT / 'sources'
snapshots.mkdir(exist_ok=True)
for f in [f for f in files if f.endswith('.py')]:
    snapshot = snapshots / (hashes[f] + '_' + Path(f).name)
    if not snapshot.exists():
        snapshot.write_bytes((ROOT / f).read_bytes())
manifest = dict(command=command, status='running', sha256=hashes,
                execution_route='GPU-visible execution outside the sandbox')
manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
env = os.environ.copy()
env.update(PYTHONPATH='src:examples:.runtime/qp_solvers', OMP_NUM_THREADS='1',
           OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
start = time.monotonic()
print(f'Starting {args.name}: shoulder scale={args.scale:g}, side={args.side}, start={args.start:g}', flush=True)
with prefix.with_suffix('.log').open('w') as log:
    result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
manifest.update(status='finished', returncode=result.returncode, wall_seconds=time.monotonic()-start)
manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
report = prefix.with_suffix('.tracking.json')
print(report.read_text() if report.exists() else prefix.with_suffix('.log').read_text()[-4000:], flush=True)
# A tracking failure is an experimental result, distinct from runner failure.
print(f'Finished {args.name}: simulation returncode={result.returncode}, wall={manifest["wall_seconds"]:.1f}s', flush=True)
