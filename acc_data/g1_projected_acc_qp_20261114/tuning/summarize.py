"""Audit saved trial counts and unchanged source-fidelity tolerances."""
import json
from pathlib import Path
import numpy as np

root = Path(__file__).resolve().parent
rows = []
for path in sorted(root.glob('*.result.json')):
    row = json.loads(path.read_text())
    name = row['name']
    recording = root/(name+'.npz')
    if not recording.exists():
        recording = root/(name+'.failure.npz')
    if not recording.exists():
        continue
    with np.load(recording, allow_pickle=False) as a:
        assert np.all(a['history_qp_native_solve_count'] == 1), name
        assert not np.any(a['history_priority_active']), name
        assert not np.any(a['history_fabric_task_requests_cached']), name
        np.testing.assert_array_equal(a['history_com_acc_qp_reference'], a['history_com_acc_projected'])
        if 'diagnostic_qp_native_solve_count' in a:
            assert np.all(a['diagnostic_qp_native_solve_count'] == 1), name
        row.update(control_states=len(a['history_time']), applied_effort_ticks=len(a['servo_time']),
                   single_qp=True, task_cache=False, priority=False)
        options = json.loads(str(a['controller_options'].item()))
        row['motion_speed'] = options['motion_speed']
        row['clock_final_speed'] = options['clock_final_speed']
        if 'control_time' in a:
            row['failure_time_s'] = float(a['control_time'])
    tracking = json.loads((root/(name+'.tracking.json')).read_text())
    assert tracking['tolerances'] == dict(joint_rmse_rad=.2, root_position_m=.2, root_orientation_rad=.35)
    steps = json.loads((root/(name+'.steps.json')).read_text())
    row['landings'] = steps['landings']
    row['next_step'] = steps['next_step_diagnostics']
    rows.append(row)
(root/'results.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2)+'\n')
for row in rows:
    print(row['name'], row['measured_steps'], row.get('failure_time_s', row.get('last_time_s')), row['stop_reason'])
