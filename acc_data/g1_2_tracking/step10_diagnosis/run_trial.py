"""Run one full-source experiment without overwriting earlier evidence."""
from pathlib import Path
import sys,os,json,subprocess,hashlib,time
root=Path(__file__).resolve().parents[3];out=Path(__file__).resolve().parent
name=sys.argv[1];prefix=out/name
manifest_path=prefix.with_suffix('.command.json')
if manifest_path.exists():raise SystemExit('Refuse overwrite '+name)
command=['/home/sky/anaconda3/envs/humanoid_fabric/bin/python','-u','examples/test_g1_2.py','--tracking-profile','response-wbc','--response-gain','0','--output-path',str(prefix.with_suffix('.npz'))]+sys.argv[2:]
files=['examples/test_g1_2.py','examples/g1_response_feedback.py','examples/test_g1.py','src/fabrics_sim/solver/AccelerationQP.py']
manifest=dict(command=command,status='running',sha256={f:hashlib.sha256((root/f).read_bytes()).hexdigest() for f in files})
snapshots=out/'sources';snapshots.mkdir(exist_ok=True)
for f,digest in manifest['sha256'].items():
 snapshot=snapshots/(digest+'_'+Path(f).name)
 if not snapshot.exists():snapshot.write_bytes((root/f).read_bytes())
manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
env=os.environ.copy();env.update(PYTHONPATH='src:examples:.runtime/qp_solvers',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
start=time.monotonic()
with prefix.with_suffix('.log').open('w') as log:
 result=subprocess.run(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
manifest.update(status='finished',returncode=result.returncode,wall_seconds=time.monotonic()-start)
manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
report=prefix.with_suffix('.tracking.json')
print(name,report.read_text() if report.exists() else prefix.with_suffix('.log').read_text()[-3000:],flush=True)
