"""Compact completed follow-up trials without treating landings as full tracking."""
import json
from pathlib import Path
OUT=Path(__file__).resolve().parent
rows=[]
for file in sorted(OUT.glob('twelve_*.result.json')):
 r=json.loads(file.read_text());s=r.get('steps',{});t=r.get('tracking',{});f=t.get('first_fidelity_exceeded') or {}
 rows.append(dict(name=file.name.removesuffix('.result.json'),landings=s.get('measured_steps'),last_source_frame=t.get('last_source_frame'),time=f.get('time_s'),joint_rmse=f.get('joint_rmse_rad'),root_position_error=f.get('root_position_m'),root_orientation_error=f.get('root_orientation_rad'),stop_reason=t.get('stop_reason'),full_tracking=bool(t.get('full_motion_tracking_passed')),next_step=s.get('next_step_diagnostics')))
(OUT/'twelve_followup_summary.json').write_text(json.dumps(rows,indent=2)+'\n')
print('Completed follow-up trials:',len(rows),'full passes:',sum(r['full_tracking'] for r in rows))
