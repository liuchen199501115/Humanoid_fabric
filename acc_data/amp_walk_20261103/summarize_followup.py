"""Keep complete tracking gates separate from experimental landing counts."""
from pathlib import Path
import importlib.util,json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('previous_summary',ROOT.parent/'amp_walk_20261102/contact_clock_followup/summarize_trials.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
manifest_path=ROOT/'feedback_followup/contact_wrench_fix_manifest.json'
corrected_runs=set(json.loads(manifest_path.read_text())['runs']) if manifest_path.exists() else set()
rows=[]
for path in sorted(ROOT.rglob('*.tracking.json')):
 stem=path.name.removesuffix('.tracking.json')
 if not (path.parent/(stem+'.steps.json')).exists():continue
 module.ROOT=path.parent
 row=module.summarize(stem);row['run']=str(path.parent.relative_to(ROOT)/stem)
 report=json.loads(path.read_text());row['stop_reason']=report.get('stop_reason');row['failed_time_s']=report.get('failed_time_s',report.get('fidelity_stop',{}).get('time_s'))
 with np.load(row['recording']) as recording:
  frame=recording['history_frame'].ravel();time=recording['history_time'].ravel()
  stop=row['first_fidelity_exceeded_time_s'];valid=np.ones(len(time),dtype=bool) if stop is None else time<stop-1e-9
  row['last_continuously_fidelity_passing_frame']=float(frame[valid][-1]) if valid.any() else None
 row['contact_wrench_split_corrected']=row['run'] in corrected_runs
 rows.append(row)
preflights=[]
for path in sorted(ROOT.rglob('*.centroidal_path.json')):
 data=json.loads(path.read_text());data['report']=str(path.relative_to(ROOT));preflights.append(data)
optimization_preflights=[]
for path in sorted((ROOT/'trajectory_sqp').glob('*_candidate.json')):
 data=json.loads(path.read_text());history=data.get('iterations',[])
 optimization_preflights.append(dict(report=str(path.relative_to(ROOT)),optimized_source_frames=data.get('optimized_source_frames'),last_iteration=history[-1] if history else None,full_tracking_passed=False))
current_rows=[row for row in rows if row['contact_wrench_split_corrected']]
historical_rows=[row for row in rows if not row['contact_wrench_split_corrected']]
best_strict=max(current_rows,key=lambda row:(row['full_motion_tracking_passed'],row['prefix_fidelity_passing_landings'],row['last_continuously_fidelity_passing_frame'] or 0.)) if current_rows else None
best_historical=max(historical_rows,key=lambda row:row['prefix_fidelity_passing_landings']) if historical_rows else None
source=ROOT.parent.parent/'retarget/amp_reference/amp_walk_1ms.pkl'
report=dict(source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),source_frames=2000,source_fps=200.,
 full_motion_tracking_passed=any(row['full_motion_tracking_passed'] for row in current_rows),
 unchanged_validation_thresholds=dict(joint_rmse_rad=.2,root_position_m=.2,root_orientation_rad=.35,landing_error_m=.04,landing_hold_s=.05,flight_clearance_m=.02),
 best_preexisting_experimental_record='acc_data/amp_walk_20261102/contact_clock_followup/vertical_feedback_verified.failure.npz',
 best_preexisting_landing_events=7,best_preexisting_strict_fidelity_prefix_events=3,
 best_current_strict_prefix=best_strict,
 best_historical_or_unverified_strict_prefix=best_historical,
 current_model_scope="Only runs listed in feedback_followup/contact_wrench_fix_manifest.json have the corrected per-foot wrench split.",
 trials=rows,reference_preflights=preflights,trajectory_optimization_preflights=optimization_preflights)
(ROOT/'progress.json').write_text(json.dumps(report,indent=2)+'\n')
for row in rows:print(row['run'],'landings',row['measured_landing_events'],'strict-prefix',row['prefix_fidelity_passing_landings'],'full',row['full_motion_tracking_passed'])
