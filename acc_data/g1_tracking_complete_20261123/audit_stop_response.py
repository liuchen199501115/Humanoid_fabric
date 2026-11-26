"""Offline fixed-torque response to a measured waist-roll joint stop.

This is a diagnostic linear dynamics calculation, never an additional runtime
QP or a change to the simulated joints. The extra constraint is applied only
to the acceleration discrepancy at a saved state.
"""
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
name = 'ten_momentum_late'
with np.load(OUT / (name + '.snapshot_3.800.npz')) as a, np.load(OUT / (name + '.npz')) as h:
    E = a['soft_problem_A_eq'][0]
    M = E[:35, :35]
    J = E[35:, :35]
    J = J[np.linalg.norm(J, axis=1) > 1e-8]
    qdd = a['soft_problem_solution'][0, :35]
    j = h['final_joint_order'].tolist().index('waist_roll_joint')
    coordinate = 6 + j
    i = np.abs(h['history_time'] - float(a['time'])).argmin()
    assert abs(h['history_q'][i, 0, 7+j] - h['joint_position_limits'][0, j, 1]) < .001
    assert abs(h['history_qd'][i, 0, coordinate]) < .1
    assert qdd[coordinate] > 0
    C = np.r_[J, np.eye(35)[coordinate:coordinate+1]]
    K = np.block([[M, C.T], [C, np.zeros((len(C), len(C)))]])
    rhs = np.r_[np.zeros(35+len(J)), -qdd[coordinate]]
    response = np.linalg.solve(K, rhs)
    np.testing.assert_allclose(K @ response, rhs, atol=1e-9)
    dt = h['history_time'][i+1] - h['history_time'][i]
    observed = (h['history_qd'][i+1, 0] - h['history_qd'][i, 0]) / dt
    report = dict(trial=name, time_s=float(a['time']),
        waist_qdd_predicted=float(qdd[coordinate]),
        waist_qdd_observed=float(observed[coordinate]),
        root_qdd_predicted=qdd[3:6].tolist(),
        root_qdd_observed=observed[3:6].tolist(),
        root_qdd_with_stop_reaction=(qdd+response[:35])[3:6].tolist(),
        stop_reaction_multiplier=float(response[-1]),
        note='Offline fixed-torque constrained mass response to zero waist-roll acceleration. Same observed foot constraints; no change to runtime controller or physics.')
    (OUT / 'waist_stop_mass_response.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
