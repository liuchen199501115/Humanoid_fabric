from pathlib import Path
import os,sys,json,subprocess,time,hashlib
root=Path('/home/sky/Humanoid_fabric')
out=root/'acc_data/g1_2_tracking/formula_verification'
name=sys.argv[1];profile=sys.argv[2]
prefix=out/name
if prefix.with_suffix('.log').exists():raise SystemExit('Refuse to overwrite existing trial')
cmd=['/home/sky/anaconda3/envs/humanoid_fabric/bin/python','-u','examples/test_g1_2.py','--tracking-profile',profile,'--output-path',str(prefix.with_suffix('.npz'))]+sys.argv[3:]
files=['examples/test_g1.py','examples/test_g1_2.py','src/fabrics_sim/solver/AccelerationQP.py','acc_data/amp_walk_20261103/feedback_followup/unload_qp.py']
manifest=dict(command=cmd,sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in files},status='running')
prefix.with_suffix('.command.json').write_text(json.dumps(manifest,indent=2)+'\n')
env=os.environ.copy();env.update(PYTHONPATH='src:examples:.runtime/qp_solvers',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
start=time.time()
with prefix.with_suffix('.log').open('w') as log:
    process=subprocess.run(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
manifest.update(status='finished',returncode=process.returncode,wall_seconds=time.time()-start)
prefix.with_suffix('.command.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(dict(name=name,returncode=process.returncode,wall_seconds=manifest['wall_seconds'])))
print('\n'.join(prefix.with_suffix('.log').read_text().splitlines()[-18:]))
