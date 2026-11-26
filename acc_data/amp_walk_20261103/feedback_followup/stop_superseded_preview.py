"""Stop only this workspace's identified experiment and its predictor worker."""
from pathlib import Path
import os,json,signal,sys,re
import numpy as np
p=Path(__file__).resolve().parent;root=p.parents[2];run=sys.argv[1];reason=sys.argv[2]
if not re.fullmatch(r'fixed_clock120_body10_full_body_preview[0-9a-z_]+',run):raise ValueError('Expected a local preview experiment name')
if (p/(run+'.tracking.json')).exists():raise SystemExit('Trial already has a completed report; nothing stopped')
stopped=[]
for proc in Path('/proc').iterdir():
 if not proc.name.isdigit():continue
 try:
  args=[x.decode() for x in (proc/'cmdline').read_bytes().split(b'\0') if x]
  physics=str(p/(run+'.npz')) in args and any(Path(x).name=='trial_tracking_tube_clock.py' for x in args)
  worker=str(p/'preview_worker.py') in args and (proc/'fd/2').resolve()==p/(run+'.planner_worker.log')
  if (physics or worker) and proc.stat().st_uid==os.getuid() and (proc/'cwd').resolve()==root:
   os.kill(int(proc.name),signal.SIGTERM);stopped.append(int(proc.name))
 except (OSError,UnicodeDecodeError):pass
if not stopped:raise SystemExit('No matching owned experiment process; no report written')
record=dict(reason=reason,stopped_pids=stopped,full_motion_tracking_passed=False,incomplete=True)
problem=p/(run+'.full_body_preview.latest_problem.npz')
if problem.exists():
 with np.load(problem) as d:record.update(last_replan_time=float(d['time']),last_replan_source_frame=float(d['frame']))
(p/(run+'.superseded.json')).write_text(json.dumps(record,indent=2)+'\n');print(record)
