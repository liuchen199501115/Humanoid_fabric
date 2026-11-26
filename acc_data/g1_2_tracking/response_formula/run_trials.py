"""Execute selected complete-source trials, preserving commands and failures."""
from pathlib import Path
import os,sys,json,subprocess,time,hashlib
root=Path(__file__).resolve().parents[3];out=Path(__file__).resolve().parent
trials={
 'brake_waist20':['--response-gain','0','--state-limit-priority','waist','--state-limit-weight','20'],
 'brake_waist80':['--response-gain','0','--state-limit-priority','waist','--state-limit-weight','80'],
 'brake_upper80':['--response-gain','0','--state-limit-priority','upper','--state-limit-weight','80'],
 'response_gate025':['--response-gain','.25','--response-error-onset','.10'],
 'response_gate075':['--response-gain','.75','--response-error-onset','.10'],
 'stop25_80':['--response-gain','0','--state-limit-priority','waist','--state-limit-weight','80','--state-braking-budget','25'],
 'stop25_320':['--response-gain','0','--state-limit-priority','waist','--state-limit-weight','320','--state-braking-budget','25'],
 'stop25_root500':['--response-gain','0','--state-limit-priority','waist','--state-limit-weight','320','--state-braking-budget','25','--state-root-metric','500'],
 'stop25_hard':['--response-gain','0','--state-limit-priority','waist','--state-limit-weight','80','--state-braking-budget','25','--state-braking-hard'],
 'stop25_hard_root500':['--response-gain','0','--state-limit-priority','waist','--state-limit-weight','80','--state-braking-budget','25','--state-braking-hard','--state-root-metric','500'],
 'stop25_pose500':['--response-gain','0','--state-limit-priority','waist','--state-limit-weight','320','--state-braking-budget','25','--state-root-metric','500','--state-root-angular-scale','5'],
 'stop25_pose_early':['--response-gain','0','--state-limit-priority','waist','--state-limit-weight','320','--state-braking-budget','25','--state-root-metric','500','--state-root-angular-scale','5','--state-root-onset','.07'],
 'stop25_hard_recovery':['--response-gain','0','--state-limit-priority','waist','--state-limit-weight','80','--state-braking-budget','25','--state-braking-hard','--state-soft-orientation'],
 'stop25_hard_pose_recovery':['--response-gain','0','--state-limit-priority','waist','--state-limit-weight','80','--state-braking-budget','25','--state-braking-hard','--state-soft-orientation','--state-root-metric','500','--state-root-angular-scale','5','--state-root-onset','.07'],
 'response_zero':['--response-gain','0'],
 'response025':['--response-gain','.25'],
 'response075':['--response-gain','.75'],
}
for name in sys.argv[1:]:
 options=trials[name];prefix=out/name
 if prefix.with_suffix('.command.json').exists():raise SystemExit('Refuse overwrite '+name)
 command=['/home/sky/anaconda3/envs/humanoid_fabric/bin/python','-u','examples/test_g1_2.py','--tracking-profile','response-wbc','--output-path',str(prefix.with_suffix('.npz'))]+options
 files=['examples/test_g1_2.py','examples/g1_response_feedback.py','examples/test_g1.py','src/fabrics_sim/solver/AccelerationQP.py','acc_data/amp_walk_20261103/feedback_followup/unload_qp.py']
 manifest=dict(command=command,status='running',sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in files})
 snapshot_dir=out/'sources';snapshot_dir.mkdir(exist_ok=True)
 manifest['source_snapshots']={}
 for file in files:
  snapshot=snapshot_dir/(manifest['sha256'][file]+'_'+Path(file).name)
  if not snapshot.exists():snapshot.write_bytes((root/file).read_bytes())
  manifest['source_snapshots'][file]=str(snapshot.relative_to(root))
 prefix.with_suffix('.command.json').write_text(json.dumps(manifest,indent=2)+'\n')
 env=os.environ.copy();env.update(PYTHONPATH='src:examples:.runtime/qp_solvers',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
 start=time.monotonic();print('Starting '+name,flush=True)
 with prefix.with_suffix('.log').open('w') as log:
  process=subprocess.run(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
 manifest.update(status='finished',returncode=process.returncode,wall_seconds=time.monotonic()-start)
 prefix.with_suffix('.command.json').write_text(json.dumps(manifest,indent=2)+'\n')
 report=prefix.with_suffix('.tracking.json')
 print(name,json.loads(report.read_text()) if report.exists() else '\n'.join(prefix.with_suffix('.log').read_text().splitlines()[-12:]),flush=True)
