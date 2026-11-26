"""Compare QP acceleration with measured motion near physical joint stops."""
import json
import sys
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
for name in sys.argv[1:] or ['ten_verify']:
    with np.load(OUT / (name + '.npz')) as a:
        t = a['history_time']
        q = a['history_q'][:, 0, 7:]
        v = a['history_qd'][:, 0]
        predicted = a['history_qdd_qp'][:, 0]
        observed = np.diff(v, axis=0) / np.diff(t)[:, None]
        limits = a['joint_position_limits'][0]
        names = a['final_joint_order'].tolist()
        records = []
        for j, joint in enumerate(names):
            lower, upper = limits[j]
            near_upper = q[:-1, j] >= upper - .001
            near_lower = q[:-1, j] <= lower + .001
            outward = ((near_upper & (predicted[:-1, j+6] > 10))
                       | (near_lower & (predicted[:-1, j+6] < -10)))
            stopped = np.abs(v[:-1, j+6]) < .1
            hits = np.flatnonzero(outward & stopped)
            if not len(hits):
                continue
            late_hits = hits[t[hits] >= 3.2]
            records.append(dict(joint=joint, count=len(hits),
                first_time_s=float(t[hits[0]]),
                late_first_time_s=float(t[late_hits[0]]) if len(late_hits) else None,
                peak_outward_predicted_acceleration_rad_s2=float(
                    np.abs(predicted[hits, j+6]).max()),
                observed_acceleration_rms_rad_s2=float(
                    np.sqrt(np.mean(observed[hits, j+6]**2)))))
        intervals = []
        for lo, hi in [(3.2, 3.6), (3.65, 3.75), (3.79, 3.82), (3.82, 3.848)]:
            select = (t[:-1] >= lo) & (t[:-1] < hi)
            if not select.any():
                continue
            intervals.append(dict(start_s=lo, end_s=hi,
                mean_predicted_root_angular_acceleration=predicted[:-1][select, 3:6].mean(0).tolist(),
                mean_observed_root_angular_acceleration=observed[select, 3:6].mean(0).tolist()))
        report = dict(trial=name, joint_stops=records, root_angular_intervals=intervals,
            note='Outward QP predictions while a joint is stationary at its physical stop identify unmodeled stop reactions. Interval differences can also include foot impacts; this is diagnostic evidence, not a causal proof in isolation.')
        (OUT / (name + '.joint_stop_audit.json')).write_text(json.dumps(report, indent=2)+'\n')
        print(name, [(r['joint'], r['late_first_time_s'], r['count']) for r in records])
