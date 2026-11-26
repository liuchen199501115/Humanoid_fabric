"""Frozen-state feasibility ablations; does not modify controller constraints."""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
source = ROOT / 'acc_data/g1_both_paths_20261123/baseline.failure.npz'
with np.load(source) as data:
    p = {k: data['problem_' + k][0].astype(float)
         for k in ('G', 'b', 'A_eq', 'b_eq', 'lb', 'ub')}
    C = data['task_acceleration_C'][0]
G, b, E, e, lo, hi = [p[k] for k in ('G', 'b', 'A_eq', 'b_eq', 'lb', 'ub')]
assert G.shape == (631, 76)
np.testing.assert_array_equal(G[40:, :35], C)
speed = np.zeros((58, 35))
speed[:29, 6:] = .002 * np.eye(29)
speed[29:, 6:] = -.002 * np.eye(29)
np.testing.assert_allclose(G[570:628, :35], speed, rtol=1e-6, atol=0)
groups = dict(wrench=list(range(40)), collision=list(range(40, 570)),
              velocity=list(range(570, 628)), joint_tube=[628],
              root_position_tube=[629], root_orientation_tube=[630],
              all_tubes=[628, 629, 630])
cases = []
for name in ['none', *groups, 'torque', 'contact_equalities']:
    mask = np.ones(len(b), bool)
    equalities = np.ones(len(e), bool)
    lower, upper = lo.copy(), hi.copy()
    if name in groups:
        mask[groups[name]] = False
    if name == 'torque':
        lower[47:], upper[47:] = -np.inf, np.inf
    if name == 'contact_equalities':
        equalities[35:] = False
    result = linprog(np.zeros(76), A_ub=G[mask], b_ub=b[mask],
                     A_eq=E[equalities], b_eq=e[equalities],
                     bounds=list(zip(lower, upper)), method='highs')
    row = dict(removed=name, status=int(result.status), message=result.message)
    if result.success:
        z = result.x
        row['max_retained_violation'] = float(max(0., (G[mask] @ z - b[mask]).max(),
            abs(E[equalities] @ z - e[equalities]).max(), (lower - z).max(), (z - upper).max()))
        row['tube_residuals'] = (G[-3:] @ z - b[-3:]).tolist()
        assert row['max_retained_violation'] < 1e-6
    cases.append(row)
    print(name, result.status)
report = dict(source=str(source.relative_to(ROOT)),
              scope='Frozen failure state only. Removed constraints are diagnostic ablations, not controller changes or full-tracking demonstrations.',
              groups=groups, cases=cases)
(OUT / 'group_feasibility.json').write_text(json.dumps(report, indent=2) + '\n')
