"""Summarize this continuation without equating a solver pass with tracking."""
from pathlib import Path
import json,hashlib
p=Path(__file__).parent
names=['root_task415','root_task415_weight300','root_task415_support','flatten_support415',
       'flatten_support415_run','physics2000_servo','weighted_tasks415',
       'weighted_tasks415_history_fixed','retimed090_before_turn','weighted_after12',
       'weighted_root_after12','weighted_root_prepare12']
rows=[]
for name in names:
 command=json.loads((p/(name+'.command.json')).read_text())
 row=dict(run=name,status=command['status'],returncode=command.get('returncode'))
 steps=p/(name+'.steps.json');raw=p/(name+'.raw_source.json');audit=p/(name+'.audit.json')
 row['completed_physical_record']=steps.exists() and raw.exists()
 if row['completed_physical_record']:
  s=json.loads(steps.read_text());r=json.loads(raw.read_text());a=json.loads(audit.read_text())
  row.update(verified_prefix_landings=a.get('prefix_fidelity_passing_landings',0),
      raw_source_prefix_landings=r['raw_source_prefix_landings'],last_frame=r['last_source_frame'],
      full_motion_tracking_passed=a.get('full_motion_tracking_passed',False))
 else:
  row.update(full_motion_tracking_passed=False,reason=('Argument parsing rejected duplicate positional diagnostic time; corrected in flatten_support415_run' if name=='flatten_support415' else 'History serialization KeyError; fixed and rerun in weighted_tasks415_history_fixed' if name=='weighted_tasks415' else 'No completed physical record'))
 log=(p/(name+'.log')).read_text()
 for line in log.splitlines():
  if line.startswith('Original-source fidelity limit exceeded: '):row['first_fidelity_failure']=json.loads(line.split(': ',1)[1])
 rows.append(row)
report=dict(scope='Continuation through examples/test_g1.py; all attempts request the complete original 2000-frame clip and all 29 landings.',
 source_sha256=hashlib.sha256(Path('retarget/amp_reference/amp_walk_1ms.pkl').read_bytes()).hexdigest(),
 completed_physical_trials=sum(r['completed_physical_record'] for r in rows),
 full_motion_tracking_passed=any(r['full_motion_tracking_passed'] for r in rows),
 selected_profile_unchanged=True,tests=json.loads((p/'task_followup_tests.json').read_text()),
 default_disabled_options=['root-position-task-start','support-flatten-start','weighted-tasks-start'],
 reference_force_audit='reference_force_audit.json',retimed_force_audit='retimed090_force_audit.json',
 response_audit='applied_response_audit.json',trials=rows)
(p/'task_followup_report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ('trials','tests')},indent=2))
for r in rows:print(r['run'],r.get('verified_prefix_landings'),r.get('last_frame'),r['status'])
