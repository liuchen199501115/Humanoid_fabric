"""Verify the recorded single-QP contract and unchanged tracking acceptance."""
import json
from pathlib import Path
import numpy as np

root = Path(__file__).resolve().parent
rows = []
for path in sorted(root.glob('*.result.json')):
    result = json.loads(path.read_text())
    name = result['name']
    if not result['tracking_evaluated']:
        continue
    record = root / (name + '.npz')
    if not record.exists():
        record = root / (name + '.failure.npz')
    with np.load(record, allow_pickle=False) as a:
        assert np.all(a['history_qp_native_solve_count'] == 1), name
        assert not np.any(a['history_priority_active']), name
        assert not np.any(a['history_fabric_task_requests_cached']), name
        assert not np.any(a['history_joint_tube_recovery_used']), name
        if 'command_acc_correction' in a:
            assert not np.any(a['command_acc_correction']), name
        assert not np.any(a['history_torque_pd_estimate']), name
        names = a['joint_names'] if 'joint_names' in a else a['final_joint_order']
        order = [a['servo_joint_order'].tolist().index(n) for n in names]
        indices = a['servo_qp_update_index'].astype(int)
        np.testing.assert_array_equal(a['servo_torque'][:, :, order], a['history_torque_qp_total'][indices])
        np.testing.assert_array_equal(a['history_qdd_qp_reference'], a['history_qdd_projected'])
        if 'diagnostic_qp_native_solve_count' in a:
            assert np.all(a['diagnostic_qp_native_solve_count'] == 1), name
        options = json.loads(str(a['controller_options'].item()))
        row = dict(name=name, options=result['options'], control_states=len(a['history_time']),
            applied_effort_ticks=len(a['servo_time']), single_qp=True, task_cache=False,
            reference_is_projected=True, applied_torque_equals_qp=True, motion_speed=options['motion_speed'],
            clock_final_speed=options['clock_final_speed'])
    tracking = json.loads((root / (name + '.tracking.json')).read_text())
    steps = json.loads((root / (name + '.steps.json')).read_text())
    assert tracking['tolerances'] == dict(joint_rmse_rad=.2, root_position_m=.2, root_orientation_rad=.35)
    next_step = steps['next_step_diagnostics']
    if next_step is not None:
        assert next_step['required_clearance_m'] == .02
        assert next_step['required_landing_hold_s'] == .05
        assert next_step['landing_position_tolerance_m'] == .04
    row.update(measured_steps=steps['measured_steps'], landings=steps['landings'],
        next_step=next_step, tracking=tracking, stop_reason=tracking.get('stop_reason', steps.get('stop_reason')),
        stop_time_s=tracking.get('failed_time_s', steps.get('last_time_s')))
    rows.append(row)
baseline = root / 'baseline_traced.failure.npz'
previous = root.parent / 'tuning/projection1_attitude.failure.npz'
if baseline.exists():
    with np.load(baseline) as a, np.load(previous) as b:
        fields = ['history_time', 'history_q', 'history_qd', 'history_qdd_qp', 'servo_torque']
        fields = [k for k in fields if k in a and k in b]
        for k in fields:
            np.testing.assert_array_equal(a[k], b[k])
        baseline_match = dict(bitwise_equal_fields=fields, previous=str(previous))
else:
    baseline_match = None
snapshots = root / 'baseline_snapshots.failure.npz'
if baseline.exists() and snapshots.exists():
    with np.load(baseline) as a, np.load(snapshots) as b:
        for k in baseline_match['bitwise_equal_fields']:
            np.testing.assert_array_equal(a[k], b[k])
        baseline_match['snapshot_recording_changes_trajectory'] = False
(root / 'results.json').write_text(json.dumps(dict(baseline_reproduction=baseline_match, trials=rows), indent=2) + '\n')
for row in rows:
    print(row['name'], row['measured_steps'], row['stop_time_s'], row['stop_reason'])
