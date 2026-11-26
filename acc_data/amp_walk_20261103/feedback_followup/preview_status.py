"""Read completed-trial results before interpreting a possibly stale solver log."""
from pathlib import Path
import sys,json,time
import numpy as np
p=Path(__file__).parent;run=sys.argv[1];result=dict(run=run)
report=p/(run+'.tracking.json')
if report.exists():
 result['state']='finished';data=json.loads(report.read_text())
 for key in ['full_motion_tracking_passed','measured_landings','expected_landings','last_source_frame','stop_reason','fidelity_stop']:result[key]=data.get(key)
else:
 active=[];output=(p/(run+'.npz')).resolve()
 for proc in Path('/proc').iterdir():
  if not proc.name.isdigit():continue
  try:args=[x.decode() for x in proc.joinpath('cmdline').read_bytes().split(b'\0') if x]
  except (OSError,UnicodeDecodeError):continue
  if str(output) in args and any(Path(x).name=='trial_tracking_tube_clock.py' for x in args):active.append(int(proc.name))
 result['state']='running' if active else 'no_active_physics_process';result['physics_pids']=active
 problem=p/(run+'.full_body_preview.latest_problem.npz')
 if problem.exists():
  with np.load(problem) as data:result['latest_replan_time']=float(data['time']);result['latest_source_frame']=float(data['frame'])
  result['latest_problem_age_seconds']=round(time.time()-problem.stat().st_mtime,1)
 fidelity=p/(run+'.replan_states/fidelity.jsonl')
 if fidelity.exists():result['latest_actual_fidelity']=json.loads(fidelity.read_text().splitlines()[-1])
records=p/(run+'.full_body_preview.json')
if records.exists():
 rows=json.loads(records.read_text());result['last_plans']=[{k:x.get(k) for k in ['time','success','status','reason','solve_seconds']} for x in rows[-2:]]
print(json.dumps(result,indent=2))
