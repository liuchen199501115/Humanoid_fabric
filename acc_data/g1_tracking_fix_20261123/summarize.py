"""Audit completed trials; startup errors and crashes are not step results."""
import json
from pathlib import Path
from collections import Counter
import numpy as np
OUT = Path(__file__).resolve().parent
rows, incomplete = [], []
for path in sorted(OUT.glob('*.result.json')):
    record = json.loads(path.read_text())
    name = record['name']
    if 'tracking' not in record:
        incomplete.append(dict(name=name, returncode=record['returncode'],
                               timed_out=record['timed_out'], options=record['options']))
        continue
    t, s = record['tracking'], record['steps']
    assert t['tolerances'] == dict(joint_rmse_rad=.2, root_position_m=.2, root_orientation_rad=.35)
    data = OUT/(name+'.npz')
    if not data.exists(): data = OUT/(name+'.failure.npz')
    with np.load(data) as a:
        assert (a['history_qp_native_solve_count'] == 1).all(), name
        for key in ('history_priority_active', 'history_fabric_task_requests_cached',
                    'history_joint_tube_recovery_used', 'history_torque_pd_estimate'):
            assert not a[key].any(), (name, key)
        names = a['joint_names'] if 'joint_names' in a else a['final_joint_order']
        order = [a['servo_joint_order'].tolist().index(n) for n in names]
        np.testing.assert_array_equal(a['servo_torque'][:, :, order],
            a['history_torque_qp_total'][a['servo_qp_update_index'].astype(int)])
        rows.append(dict(name=name, options=record['options'],
            stop_time_s=t.get('failed_time_s', s['last_time_s']),
            landings=s['measured_steps'], expected_landings=s['expected_steps'],
            last_source_frame=s['last_source_frame'], source_frames=t['source_frame_count'],
            full_tracking_passed=t['full_motion_tracking_passed'], stop_reason=t['stop_reason'],
            peak_joint_rmse_rad=t['peak_joint_rmse'], peak_root_position_m=t['peak_root_position_error'],
            peak_root_orientation_rad=t['peak_root_orientation_error'],
            max_tube_slack=a['history_tracking_tube_soft_slack'].max(axis=(0,1)).tolist(),
            statuses=dict(Counter(a['history_qp_solver_status'].ravel().tolist())),
            invariants=dict(one_native_attempt_per_update=True, no_task_cache_at_qp=True,
                no_second_qp=True, exact_applied_qp_torque=True, original_fidelity_limits=True)))
report = dict(scope='Same complete original PKL and acceptance limits. Slower playback trials are labeled in options. All runs use one inverse-dynamics QP and no MPC. Verified steps are observations, not a theoretical maximum.',
              trials=rows, incomplete_runs=incomplete)
(OUT/'results.json').write_text(json.dumps(report, indent=2)+'\n')
for r in rows:
    print(r['name'], r['landings'], r['stop_time_s'], r['stop_reason'])
