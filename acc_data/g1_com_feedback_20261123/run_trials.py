"""Isolated closed-loop comparisons for the original COM feedback objective."""
import hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
TRIALS={'baseline':0.,'feedback1':1.,'feedback4':4.,'feedback025':.25,'baseline_v2':0.,'feedback1_v2':1.,'feedback4_v2':4.,'fabric_metric':0.,'fabric_metric_com1':1.,'feedback1_late':1.,'feedback4_late':4.}
# Serialize GPU experiments, including a previous bounded initialization attempt.
wait_deadline=time.monotonic()+300
while any(not p.with_name(p.name.replace('.command.json','.result.json')).exists() for p in OUT.glob('*.command.json')):
 if time.monotonic()>wait_deadline:raise TimeoutError('Previous experiment has not finished')
 time.sleep(1)
for name in sys.argv[1:]:
 scale=TRIALS[name];path=OUT/(name+'.npz')
 if path.with_suffix('.command.json').exists():raise FileExistsError(name)
 cmd=[sys.executable,'-u','-c',"import faulthandler,runpy; faulthandler.dump_traceback_later(60,repeat=True); runpy.run_path('examples/test_g1.py',run_name='__main__')",'--tracking-profile','amp-full','--output-path',str(path),'--qp-com-feedback-scale',str(scale)]
 if name.startswith('fabric_metric'):cmd+=['--qp-fabric-metric']
 if name.endswith('_late'):cmd+=['--qp-com-feedback-start','.7','--qp-com-feedback-ramp','.1']
 path.with_suffix('.command.json').write_text(json.dumps(cmd,indent=2)+'\n')
 sources=['examples/test_g1.py','examples/g1_acceleration_cost.py','src/fabrics_sim/fabrics/dynamics_aware.py','src/fabrics_sim/solver/com_diagnostics.py','src/fabrics_sim/fabrics/g1_pose_fabric.py']
 path.with_suffix('.sources.json').write_text(json.dumps({s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in sources},indent=2)+'\n')
 env=dict(os.environ,PYTHONPATH='src:examples:.runtime/qp_solvers',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
 with path.with_suffix('.log').open('w') as log:
  proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  try:code=proc.wait(timeout=240)
  except subprocess.TimeoutExpired:
   os.killpg(proc.pid,signal.SIGTERM)
   try:code=proc.wait(timeout=10)
   except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);code=proc.wait()
 report=dict(name=name,scale=scale,returncode=code)
 for suffix in ('steps','tracking'):
  p=path.with_suffix('.'+suffix+'.json')
  if p.exists():report[suffix]=json.loads(p.read_text())
 path.with_suffix('.result.json').write_text(json.dumps(report,indent=2)+'\n')
 print(name,code,flush=True)
