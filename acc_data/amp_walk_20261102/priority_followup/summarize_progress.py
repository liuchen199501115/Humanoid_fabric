from pathlib import Path
import json,hashlib
p=Path(__file__).resolve().parent;rows=[]
for path in sorted(p.glob('*.tracking.json')):
 t=json.loads(path.read_text());stem=path.name.removesuffix('.tracking.json');s=json.loads((p/(stem+'.steps.json')).read_text())
 rows.append(dict(run=stem,goal_policy=s.get('landing_goal_policy','fixed_contact_onset'),measured_landings=s['measured_steps'],
                  full_tracking_passed=t.get('full_motion_tracking_passed',False),last_time_s=s.get('last_time_s'),failed_time_s=t.get('failed_time_s'),
                  peak_joint_rmse_rad=t['peak_joint_rmse'],peak_root_orientation_error_rad=t['peak_root_orientation_error']))
source=Path('retarget/amp_reference/amp_walk_1ms.pkl');sha=hashlib.sha256(source.read_bytes()).hexdigest()
report=dict(source=str(source),source_sha256=sha,source_unchanged=sha=='a870f6642380a5d7e23ae34fbc8a9614bf4164d6a9ce0a26432ae35f3401b8c4',
            full_motion_tracking_passed=False,experimental_candidate='landing_tail_verified.failure.npz',
            verified_landing_events=6,strict_six_step_tracking_passed=False,expected_landing_events=29,
            fidelity=json.loads((p/'six_landing_fidelity.json').read_text()),reproduction=json.loads((p/'verified_reproduction.json').read_text()),
            retained_changes=['Continuous tracking audits original same-frame source foot-link positions; retains fixed-contact-onset comparison',
                              'Reports next unconfirmed step and missing flight/load/position/hold evidence'],
            unchanged_gates=dict(min_flight_clearance_m=.02,min_support_force_n=100,min_landing_force_n=20,landing_error_m=.04,landing_hold_s=.05,joint_rmse_rad=.2,root_position_m=.2,root_angle_rad=.35),
            evaluation_defect=dict(event=5,source_contact_onset_frame=352,source_link_translation_after_touchdown_m=.056,
                                   ideal_source_max_fixed_goal_hold_s=.022,required_hold_s=.05),
            validation=dict(regression_tests=490,regression_passed=True,log=str(p/'regression.log')),
            trials=rows,unexecuted_configuration=dict(run='braking',reason='Finite predicted-acceleration limits required; capped variant was run separately'))
(p/'progress.json').write_text(json.dumps(report,indent=2)+'\n')
status=Path('acc_data/amp_walk_20261102/tracking_progress.json');d=json.loads(status.read_text());d['priority_followup']=dict(report=str(p/'progress.json'),experimental_record=str(p/'landing_tail_verified.failure.npz'),verified_landing_events=6,strict_six_step_tracking_passed=False,full_motion_tracking_passed=False,landing_goal_policy='source_pose',historical_fixed_goal_comparison_retained=True,control_states_unchanged_by_audit_fix=True)
d['validation'].update(full_regression_tests=490,full_regression_passed=True,full_regression_log=str(p/'regression.log'),strict_tracking_passed=False)
d['full_clip_attempt'].update(experimental_max_verified_landings=6,experimental_strict_tracking_passed=False)
status.write_text(json.dumps(d,indent=2)+'\n');print('Recorded',len(rows),'completed tracking trials; full tracking remains false.')
