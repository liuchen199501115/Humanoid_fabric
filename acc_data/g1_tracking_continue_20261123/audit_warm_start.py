"""Offline one-attempt comparisons on the same saved QP coefficients.

Warm state is reconstructed from logged outputs (which may have lower precision
than the solver's internal cache). This is an audit, not a runtime retry path.
"""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from fabrics_sim.solver.daqp_backend import DAQPBackend

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
data = np.load(ROOT/'acc_data/g1_tracking_fix_20261123/align15_soft.failure.npz')
qp = SimpleNamespace(**{k: torch.tensor(data['soft_problem_'+k], dtype=torch.float64)
    for k in ['H', 'h', 'G', 'b', 'lb', 'ub', 'A_eq', 'b_eq']})
warm = np.concatenate([data[k][-1].ravel() for k in ['history_qdd_qp',
    'history_contact_wrench', 'history_torque_qp_total', 'history_tracking_tube_soft_slack']])
report = {}
for mode in ['warm', 'cold']:
    solver = DAQPBackend(single_solve=True, single_proximal=1e-6)
    if mode == 'warm':
        solver._previous_solutions[('tracking', 0)] = warm
    result = solver.solve(qp)
    report[mode] = dict(status=solver.last_status, valid=result.converged.tolist(),
                       native_calls=solver.last_solve_counts)
(OUT/'warm_cold_audit.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2))
