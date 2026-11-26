"""Compare the saved acceleration stages against the control reference velocity."""
import json
from pathlib import Path
import numpy as np

root = Path(__file__).resolve().parent
with np.load(root / 'baseline_traced.failure.npz') as a:
    t = a['history_time'].reshape(-1)
    v = a['history_qd'][:, 0, 6:]
    rv = a['history_whole_body_reference_v'][:, 0, 6:]
    ra = a['history_whole_body_reference_a'][:, 0, 6:]
    error = v - rv
    rmse = np.sqrt(np.mean(error ** 2, axis=1))
    stages = {name: a['history_qdd_' + name][:, 0, 6:] for name in ('fabric', 'projected', 'qp')}
    rates = {name: np.mean(error * (acc - ra), axis=1) for name, acc in stages.items()}
    rows = []
    for time in (.2, .4, .6, .8, 1., 1.2, 1.25, 1.3, 1.32, 1.34, 1.354):
        i = int(np.argmin(abs(t-time)))
        rows.append(dict(time_s=float(t[i]), velocity_error_rms_rad_s=float(rmse[i]),
            half_squared_velocity_error_rate={name: float(value[i]) for name, value in rates.items()},
            acceleration_rms_rad_s2={name: float(np.sqrt(np.mean(value[i] ** 2))) for name, value in stages.items()},
            joint_position_rmse_rad=float(a['history_pkl_joint_rmse'][i].item())))
    last_predicted = v[-1] + (float(a['control_time']) - t[-1]) * stages['qp'][-1]
    final_difference = a['qd'][0, 6:] - last_predicted
    report = dict(source='baseline_traced.failure.npz', reference='recorded whole_body_reference_v/a, not original-source acceptance',
        interpretation='Negative rate means that acceleration instantaneously reduces squared velocity error; it does not prove closed-loop stability.',
        samples=rows, final_joint_velocity_prediction_error_rms_rad_s=float(np.sqrt(np.mean(final_difference ** 2))),
        final_joint_velocity_prediction_error_max_rad_s=float(np.max(abs(final_difference))))
(root / 'velocity_audit.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
