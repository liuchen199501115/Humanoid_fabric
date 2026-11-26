"""Strict original-PKL comparison; a surviving QP is not a tracking pass."""
import json
from pathlib import Path
from collections import Counter
import numpy as np
OUT=Path(__file__).resolve().parent
rows=[]
for f in sorted(OUT.glob('*.result.json')):
    r=json.loads(f.read_text());name=r['name']
    if 'tracking' not in r:
        continue
    t,s=r['tracking'],r['steps']
    assert t['tolerances']==dict(joint_rmse_rad=.2,root_position_m=.2,root_orientation_rad=.35)
    p=OUT/(name+'.npz')
    if not p.exists():p=OUT/(name+'.failure.npz')
    with np.load(p) as a:
        assert (a['history_qp_native_solve_count']==1).all()
        for k in ('history_priority_active','history_fabric_task_requests_cached','history_joint_tube_recovery_used','history_torque_pd_estimate'):
            assert not a[k].any(),(name,k)
        names=a['joint_names'] if 'joint_names' in a else a['final_joint_order']
        order=[a['servo_joint_order'].tolist().index(n) for n in names]
        np.testing.assert_array_equal(a['servo_torque'][:,:,order],a['history_torque_qp_total'][a['servo_qp_update_index'].astype(int)])
        rows.append(dict(name=name,stop_time_s=t.get('failed_time_s',s['last_time_s']),
            last_source_frame=s['last_source_frame'],source_frames=t['source_frame_count'],
            landings=s['measured_steps'],expected_landings=s['expected_steps'],
            full_tracking_passed=t['full_motion_tracking_passed'],stop_reason=t['stop_reason'],
            first_fidelity_exceeded=t.get('first_fidelity_exceeded'),
            peak_joint_rmse_rad=t['peak_joint_rmse'],peak_root_position_m=t['peak_root_position_error'],
            peak_root_orientation_rad=t['peak_root_orientation_error'],
            max_normalized_tube_slack=a['history_tracking_tube_soft_slack'].max(axis=(0,1)).tolist(),
            statuses=dict(Counter(a['history_qp_solver_status'].ravel().tolist())),
            fabric_priority_mode=np.unique(a['history_fabric_priority_mode']).tolist(),
            max_nullspace_error=float(a['history_fabric_priority_nullspace_error'].max()),
            max_idempotence_error=float(a['history_fabric_priority_idempotence_error'].max()),
            invariants=dict(one_native_attempt_per_update=True,no_task_cache=True,no_priority_stage=True,
                applied_torque_equals_qp=True,original_fidelity_limits=True,no_mpc=True)))
with np.load(OUT/'baseline.failure.npz') as a,np.load(OUT.parent/'g1_soft_tube_trials_20261123/baseline_gpu.failure.npz') as b:
    compared=[]
    for k in a.files:
        if k.startswith('history_') and k in b and k!='history_qp_wall_seconds':
            if a[k].dtype.kind in 'OUS':np.testing.assert_array_equal(a[k],b[k])
            else:assert np.array_equal(a[k],b[k],equal_nan=True),k
            compared.append(k)
report=dict(scope='Independent full-source starts, identical original PKL, clock and acceptance thresholds. MPC and solver retries remain disabled. Default baseline history matches the archived baseline exactly except solver wall time.',baseline_identical_history_fields=compared,trials=rows)
(OUT/'results.json').write_text(json.dumps(report,indent=2)+'\n')
for r in rows:print({k:r[k] for k in ('name','stop_time_s','landings','stop_reason','peak_joint_rmse_rad','peak_root_position_m','peak_root_orientation_rad')})
