"""Independent source fidelity and landing evidence for completed trials."""
import importlib.util
import sys
import json
import hashlib
from pathlib import Path
import numpy as np

root = Path(__file__).resolve().parents[2]
out = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('prefix_audit', root / 'acc_data/amp_walk_20261102/contact_clock_followup/summarize_trials.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
audit.ROOT = out
rows = []
selected = set(sys.argv[1:])
for path in sorted(out.glob('*.command.json')):
    command = json.loads(path.read_text())
    if command.get('status') != 'finished':
        continue
    name = path.name.removesuffix('.command.json')
    if selected and name not in selected:
        continue
    try:
        result = audit.summarize(name)
    except (FileNotFoundError, KeyError, ValueError) as error:
        result = dict(run=name, audit_unavailable=str(error), full_motion_tracking_passed=False)
    result['returncode'] = command['returncode']
    recording = out / (name + '.npz')
    if not recording.exists():
        recording = out / (name + '.failure.npz')
    if recording.exists():
        with np.load(recording) as data:
            options = json.loads(str(data['controller_options']))
            result['execution'] = {key: options.get(key) for key in (
                'tracking_profile', 'direct_wbc', 'control_frequency', 'physics_frequency',
                'physics_iterations', 'rolling_contact_kinematics_start', 'landing_speed_limit',
                'landing_speed_limit_start', 'late_contact_stabilization_start',
                'late_contact_stabilization_kd', 'late_balance_hierarchy_start',
                'stance_posture_metric_start', 'stance_posture_metric_scale',
                'stance_posture_secondary_start', 'late_pelvis_priority_start',
                'support_orientation_priority_start', 'support_load_floor_start',
                'support_load_floor', 'reference_pose_candidate', 'root_position_task_start',
                'root_position_task_weight', 'support_flatten_start', 'weighted_tasks_start',
                'weighted_tasks_ramp', 'position_tube_start', 'position_tube_radius',
                'orientation_tube_start', 'orientation_tube_radius',
                'late_physics_substeps_start', 'late_physics_substeps',
                'late_momentum_priority_start', 'late_momentum_zero_target',
                'stance_heading_start', 'stance_heading_limit', 'stance_heading_ramp',
                'torque_continuity_start', 'torque_continuity_weight', 'torque_continuity_ramp',
                'contact_rays_start', 'physical_braking_start', 'physical_braking_upper_only',
                'physical_braking_waist_only')}
            for key in ('servo_valid', 'servo_saturation_fraction', 'qp_validated'):
                if key in data:
                    result[key] = data[key].tolist()
            if 'history_rolling_bias_correction' in data:
                result['rolling_correction_peak_m_s2'] = float(abs(data['history_rolling_bias_correction']).max())
                result['rolling_matched_samples'] = int(data['history_rolling_bias_matched'].sum())
    (out / (name + '.audit.json')).write_text(json.dumps(result, indent=2) + '\n')
    rows.append(result)
    print(name, result.get('prefix_fidelity_passing_landings'), result.get('last_source_frame'), result.get('full_motion_tracking_passed'))
if selected:
    rows = [json.loads(path.read_text()) for path in sorted(out.glob('*.audit.json'))]
(out / 'results.json').write_text(json.dumps(dict(
    source='retarget/amp_reference/amp_walk_1ms.pkl', expected_landings=29, source_frames=2000,
    full_motion_tracking_passed=any(row['full_motion_tracking_passed'] for row in rows),
    selected_profile='amp-full', selected_entry='examples/test_g1.py',
    selected_launcher='examples/run_amp_full_motion.sh', selected_experiment='baseline',
    best_verified_landings=max((row.get('prefix_fidelity_passing_landings', 0) for row in rows), default=0),
    source_sha256=hashlib.sha256((root/'retarget/amp_reference/amp_walk_1ms.pkl').read_bytes()).hexdigest(),
    validation=dict(joint_rmse_rad=.2, root_position_m=.2, root_orientation_rad=.35,
                    foot_position_m=.04, unloaded_clearance_m=.02, loaded_hold_seconds=.05),
    trials=rows), indent=2) + '\n')
