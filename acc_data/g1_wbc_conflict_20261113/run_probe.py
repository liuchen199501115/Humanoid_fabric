"""Collect unchanged-baseline WBC snapshots through the thirteenth-step attempt."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
manifest_path=OUT/'baseline_probe.command.json'
if manifest_path.exists():raise SystemExit('Refuse to overwrite baseline probe')
command=[sys.executable,'-u','examples/test_g1.py','--tracking-profile','amp-full',
         '--output-path',str(OUT/'baseline_probe.npz')]
for t in (3.48,3.58,3.65,3.75,3.96,4.216,4.28):
    command+=['--diagnose-qp-time',str(t)]
files=['examples/test_g1.py','examples/analyze_g1_shoulder_roll.py',
       'src/fabrics_sim/solver/AccelerationQP.py',
       'acc_data/amp_walk_20261103/feedback_followup/unload_qp.py',
       'acc_data/amp_walk_20261103/feedback_followup/tube_recovery.py',
       'retarget/amp_reference/amp_walk_1ms.pkl',
       'acc_data/amp_walk_20261102/amp_walk_native_contacts.pkl',
       'acc_data/amp_walk_20261103/feedback_followup/wbc_convex_catch15_25_clear35.npz']
hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
assert hashes['examples/test_g1.py']=='e738f5460f70f46040887b1f30168220badff1239c6646a8a142342661d9a7a9'
(OUT/'sources').mkdir(exist_ok=True)
for f in files:
    if f.endswith('.py'):(OUT/'sources'/(hashes[f]+'_'+Path(f).name)).write_bytes((ROOT/f).read_bytes())
manifest=dict(command=command,sha256=hashes,status='running')
manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
env=os.environ.copy()
env.update(PYTHONPATH='src:examples:.runtime/qp_solvers',OMP_NUM_THREADS='1',
           OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
start=time.monotonic()
print('Collecting seven unchanged-baseline snapshots',flush=True)
with (OUT/'baseline_probe.log').open('w') as log:
    result=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
manifest.update(status='finished',returncode=result.returncode,wall_seconds=time.monotonic()-start)
manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
report=OUT/'baseline_probe.tracking.json'
print(report.read_text() if report.exists() else (OUT/'baseline_probe.log').read_text()[-4000:])
