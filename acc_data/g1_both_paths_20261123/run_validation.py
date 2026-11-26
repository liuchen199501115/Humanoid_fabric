"""Independent closed-loop recheck of both objective paths; no controller edits."""
import hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
TRIALS={
 'baseline':[],
 'com_original':['--qp-com-feedback-scale','1'],
 'com_delayed':['--qp-com-feedback-scale','4','--qp-com-feedback-start','.7','--qp-com-feedback-ramp','.1'],
 'fabric_metric':['--qp-fabric-metric'],
}
source_paths=['examples/test_g1.py','examples/g1_acceleration_cost.py','src/fabrics_sim/fabrics/dynamics_aware.py','src/fabrics_sim/fabrics/g1_pose_fabric.py','src/fabrics_sim/solver/AccelerationQP.py','src/fabrics_sim/solver/daqp_backend.py','src/fabrics_sim/solver/com_diagnostics.py']
for name in sys.argv[1:]:
 path=OUT/(name+'.npz')
 if path.with_suffix('.command.json').exists():raise FileExistsError(name)
 cmd=[sys.executable,'-u','-c',"import faulthandler,runpy; faulthandler.dump_traceback_later(60,repeat=True); runpy.run_path('examples/test_g1.py',run_name='__main__')",'--tracking-profile','amp-full','--output-path',str(path),*TRIALS[name]]
 path.with_suffix('.command.json').write_text(json.dumps(cmd,indent=2)+'\n')
 path.with_suffix('.sources.json').write_text(json.dumps({s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in source_paths},indent=2)+'\n')
 env=dict(os.environ,PYTHONPATH='src:examples:.runtime/qp_solvers',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
 started=time.monotonic();timed_out=False
 with path.with_suffix('.log').open('w') as log:
  proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  try:code=proc.wait(timeout=240)
  except subprocess.TimeoutExpired:
   timed_out=True;os.killpg(proc.pid,signal.SIGTERM)
   try:code=proc.wait(timeout=10)
   except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);code=proc.wait()
 report=dict(name=name,options=TRIALS[name],returncode=code,timed_out=timed_out,wall_seconds=time.monotonic()-started)
 for suffix in ('steps','tracking'):
  p=path.with_suffix('.'+suffix+'.json')
  if p.exists():report[suffix]=json.loads(p.read_text())
 report['tracking_evaluated']='steps' in report and 'tracking' in report
 path.with_suffix('.result.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps({k:v for k,v in report.items() if k not in ('steps','tracking')}),flush=True)
