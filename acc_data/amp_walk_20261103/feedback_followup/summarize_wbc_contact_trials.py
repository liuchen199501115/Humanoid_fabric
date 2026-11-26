"""Recompute source-prefix fidelity and retain every completed new WBC trial."""
from pathlib import Path
import importlib.util,json,hashlib,numpy as np
p=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('prefix_summary',p.parent.parent/'amp_walk_20261102/contact_clock_followup/summarize_trials.py')
summary=importlib.util.module_from_spec(spec);spec.loader.exec_module(summary);summary.ROOT=p
names=['wbc_configuration_from_start','wbc_tracking_margin38','wbc_convex_before_turn','wbc_convex_margin41','wbc_convex_landing_pause900','wbc_convex_margin44','wbc_convex_early_resume','wbc_convex_margin44_clear35','wbc_convex_catch15_25_trial','wbc_convex_clear35_foot4','wbc_convex_catch_foot4','wbc_twelve_com4','wbc_twelve_com2','wbc_twelve_out25','wbc_twelve_reproduced','wbc_twelve_com2_transfer_floor','wbc_twelve_earlier_descent','wbc_twelve_tenth_out33','wbc_twelve_native_effort100','wbc_twelve_supported_recovery','wbc_twelve_roll_regulation','wbc_twelve_position15','wbc_twelve_early_foot4','wbc_twelve_yaw_lead10','wbc_twelve_pelvis_yaw','wbc_twelve_turn_clock_valley','wbc_twelve_roll08_native','wbc_twelve_transfer250_native','wbc_twelve_root_tail1120_only','wbc_twelve_pelvis05_transfer','wbc_twelve_support_tilt05','wbc_convex_barrier32','wbc_twelve_late_pelvis05','wbc_twelve_leg_secondary_native','wbc_twelve_tail1120_native','wbc_root_early700_trial','wbc_convex_from_start','wbc_convex_start30','wbc_twelve_recovery_diagnostic','wbc_twelve_soft_recovery_diagnostic','wbc_twelve_pelvis_replaces_torso','full_motion_no_mpc']
names.extend([
 'wbc_twelve_forward_out20_trial', 'wbc_thirteen_contact_clock',
 'wbc_twelve_com_recovery39', 'wbc_twelve_slow_before_turn',
 'wbc_twelve_clearance10_trial', 'wbc_tenth_angular4_trial',
 'wbc_low12_roll08_native', 'wbc_tenth_angular_window_back20',
 'wbc_tenth_out33_low12_trial', 'wbc_out33_pelvis20_native', 'wbc_out33_support150_native', 'wbc_out33_soft_position',
 'wbc_transfer_bridge808_trial', 'wbc_transfer_left_lead8_trial',
 'wbc_transfer_left_forward30_trial', 'wbc_out33_foot42',
 'wbc_out33_early_pelvis10', 'wbc_transfer_collision_iter1_physics',
 'wbc_transfer_collision_iter1_nativeplan',
])
rows=[]
for name in names:
 if not (p/(name+'.tracking.json')).exists():continue
 r=summary.summarize(name)
 with np.load(r['recording']) as d:
  options=json.loads(str(d['controller_options'].item()))
  assert options['direct_wbc'] and np.isinf(options['full_body_preview_start']) and np.isinf(options['centroidal_preview_start'])
  assert options['track_full_motion'] and options['tracking_steps']==0
  r['no_mpc_verified']=True
  r['self_contact_events']=d['self_contact_events'].tolist()
 (p/(name+'.strict_prefix.json')).write_text(json.dumps(r,indent=2)+'\n')
 r.pop('landings');rows.append(r)
source=Path('retarget/amp_reference/amp_walk_1ms.pkl')
report=dict(source=str(source.resolve()),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),expected_landings=29,source_frames=2000,source_fps=200.,source_duration_seconds=9.995,selected_profile='examples/run_amp_full_motion.sh',selected_prefix_recording=str(p/'wbc_twelve_reproduced.npz'),full_motion_tracking_passed=any(r['full_motion_tracking_passed'] for r in rows),best_verified_landings=max((r['prefix_fidelity_passing_landings'] for r in rows),default=0),trials=rows)
(p/'wbc_contact_tracking_status.json').write_text(json.dumps(report,indent=2)+'\n')
for row in rows:print(row['run'],row['prefix_fidelity_passing_landings'],row['last_source_frame'],row['full_motion_tracking_passed'])
