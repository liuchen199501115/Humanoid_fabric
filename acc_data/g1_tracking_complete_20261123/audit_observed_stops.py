"""Check that passive reactions are used only at measured stationary stops."""
import json
import sys
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
for name in sys.argv[1:]:
    with np.load(OUT/(name+'.npz')) as a:
        direction = a['history_observed_joint_stop_directions'][:, 0]
        reaction = a['history_joint_stop_reaction'][:, 0]
        active = direction != 0
        q = a['history_q'][:, 0, 7:]
        v = a['history_qd'][:, 0, 6:]
        limits = a['joint_position_limits'][0]
        distance = np.where(direction > 0, abs(q-limits[:, 1]), abs(q-limits[:, 0]))
        assert (distance[active] <= 1.001e-4).all()
        assert (abs(v[active]) <= .050001).all()
        assert (direction*reaction <= 1e-5).all()
        assert (reaction[~active] == 0).all()
        assert (a['history_qp_native_solve_count'] == 1).all()
        error = abs(a['history_qdd_qp'][:, 0, 6:]*active)
        assert error.max() <= .05
        order = [a['servo_joint_order'].tolist().index(n) for n in a['final_joint_order']]
        np.testing.assert_array_equal(a['servo_torque'][:, :, order],
            a['history_torque_qp_total'][a['servo_qp_update_index'].astype(int)])
        first = np.flatnonzero(active.any(1))
        report = dict(trial=name, first_active_time_s=float(a['history_time'][first[0]]) if len(first) else None,
            active_joint_updates=int(active.sum()), max_reaction_nm=float(abs(reaction).max()),
            max_locked_joint_acceleration_rad_s2=float(error.max()),
            observed_position_and_velocity_verified=True, unilateral_reactions_verified=True,
            one_native_QP_per_update=True, only_QP_motor_torque_applied=True,
            note='This audits controller mode selection and QP passivity, not a claim that the entire trajectory passed.')
        (OUT/(name+'.observed_stop_audit.json')).write_text(json.dumps(report,indent=2)+'\n')
        print(report)
