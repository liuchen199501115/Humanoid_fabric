"""Compare strict full-source trials, including execution and solver invariants."""
import json
from collections import Counter
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
rows = []
files = [OUT.parent / 'g1_both_paths_20261123' / (n + '.result.json')
         for n in ('baseline', 'fabric_metric')]
files += sorted(OUT.glob('*.result.json'))
for file in files:
    result = json.loads(file.read_text())
    name = result['name']
    if 'tracking' not in result:
        rows.append(dict(name=name, tracking_evaluated=False, returncode=result['returncode']))
        continue
    tracking, steps = result['tracking'], result['steps']
    assert tracking['tolerances'] == dict(joint_rmse_rad=.2, root_position_m=.2, root_orientation_rad=.35)
    base = file.name.removesuffix('.result.json')
    data_path = file.parent / (base + '.npz')
    if not data_path.exists():
        data_path = file.parent / (base + '.failure.npz')
    with np.load(data_path) as a:
        assert np.all(a['history_qp_native_solve_count'] == 1)
        for field in ('history_priority_active', 'history_fabric_task_requests_cached',
                      'history_joint_tube_recovery_used', 'history_torque_pd_estimate'):
            assert not np.any(a[field]), (name, field)
        joint_names = a['joint_names'] if 'joint_names' in a else a['final_joint_order']
        order = [a['servo_joint_order'].tolist().index(n) for n in joint_names]
        np.testing.assert_array_equal(a['servo_torque'][:, :, order],
            a['history_torque_qp_total'][a['servo_qp_update_index'].astype(int)])
        com_error = (a['history_com'] - a['history_com_target'])[:, 0]
        rows.append(dict(name=name, tracking_evaluated=True,
            stop_time_s=tracking.get('failed_time_s', steps.get('last_time_s')),
            stop_reason=tracking.get('stop_reason'), measured_landings=steps['measured_steps'],
            expected_landings=steps['expected_steps'], full_tracking_passed=tracking['full_motion_tracking_passed'],
            peak_joint_rmse_rad=tracking['peak_joint_rmse'],
            peak_root_position_m=tracking['peak_root_position_error'],
            final_com_error_m=com_error[-1].tolist(),
            statuses=dict(Counter(a['history_qp_solver_status'].ravel().tolist())),
            bridge_active_samples=int(a['history_contact_reference_bridge_active'].any(axis=-1).sum())
                if 'history_contact_reference_bridge_active' in a else 0,
            next_step=steps.get('next_step_diagnostics'),
            invariants=dict(one_native_attempt=True, no_task_cache=True, no_priority_stage=True,
                            applied_torque_equals_qp=True, original_fidelity_limits=True)))
report = dict(scope='Independent closed-loop starts; same original PKL, source clock and acceptance limits. No ALIP or foot-placement planner. Certificate polish is an explicit extra linear KKT refinement, not a second native QP attempt.', trials=rows)
(OUT / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
for row in rows:
    print(json.dumps({k: v for k, v in row.items() if k not in ('next_step', 'invariants')}))
