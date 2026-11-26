"""Offline LP lower bound on acceleration required by the saved hard constraints."""
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import linprog

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'acc_data/amp_walk_20261103/feedback_followup'))
from test_g1_two_tasks import load_motion_trajectory
from fabrics_sim.reference.interpolation import MotionInterpolator
from blended_spline import BlendedSplineReference

with np.load(OUT / 'baseline_traced.failure.npz') as baseline:
    options = json.loads(str(baseline['controller_options'].item()))
    trajectory, fps = load_motion_trajectory(ROOT / options['motion_pkl'])
    trajectory[:, 2] += float(baseline['motion_ground_shift'])
sampler = MotionInterpolator(trajectory, method=options['tube_reference_method'])
sampler = BlendedSplineReference(sampler, trajectory, stride=options['tube_smooth_stride'],
    start_frame=options['smooth_start'] * fps, ramp_frames=options['smooth_ramp'] * fps)

def minimum_peak(p, keep, acceleration_indices):
    n = len(p['h'])
    c = np.zeros(n+1)
    c[-1] = 1
    selector = np.eye(n)[acceleration_indices]
    G = np.vstack((np.c_[p['G'][keep], np.zeros(np.sum(keep))],
                   np.c_[selector, -np.ones(len(selector))],
                   np.c_[-selector, -np.ones(len(selector))]))
    b = np.r_[p['b'][keep], np.zeros(2*len(selector))]
    result = linprog(c, A_ub=G, b_ub=b, A_eq=np.c_[p['A_eq'], np.zeros(len(p['b_eq']))],
        b_eq=p['b_eq'], bounds=[*zip(p['lb'], p['ub']), (0, None)], method='highs')
    violation = None
    if result.success:
        x = result.x[:-1]
        violation = max(0., float(np.max(G @ result.x - b)),
            float(np.max(abs(p['A_eq'] @ x - p['b_eq']))),
            float(np.max(p['lb'] - x)), float(np.max(x - p['ub'])), -float(result.x[-1]))
        assert violation <= 1e-6, violation
    return dict(feasible=bool(result.success), status=int(result.status),
                max_constraint_violation=violation,
                minimum_max_abs_acceleration=float(result.fun) if result.success else None)

rows = []
paths = sorted(OUT.glob('baseline_snapshots.qp_*.npz'))
assert len(paths) == 4, paths
for path in paths:
    with np.load(path) as a:
        p = {k: a['problem_' + k][0] for k in ('H','h','G','b','lb','ub','A_eq','b_eq')}
        keep = np.ones(len(p['b']), dtype=bool)
        full = minimum_peak(p, keep, np.arange(35))
        joint = minimum_peak(p, keep, np.arange(6,35))
        reference = sampler(float(a['motion_frame_position'])).numpy()
        C = np.zeros(p['G'].shape[1])
        C[6:35] = 2 * (a['q'][0, 7:] - reference[0, 7:]) / 29
        matching = np.flatnonzero(np.max(abs(p['G'] - C), axis=1) < 2e-6)
        removed = None
        if len(matching):
            assert len(matching) == 1, matching
            keep[matching[0]] = False
            removed = dict(identified_joint_tube_row=int(matching[0]),
                           **minimum_peak(p, keep, np.arange(35)))
        rows.append(dict(snapshot=path.name, time_s=float(a['time']), full_constraints=full,
            joint_acceleration_only=joint, without_joint_tube=removed,
            actual_qp_max_abs_acceleration=float(np.max(abs(a['problem_solution'][0,:35])))))
report = dict(kind='offline_LP_only', tracking_evaluated=False,
    acceleration_units='base translation m/s^2; base angular and joints rad/s^2',
    purpose='Lower bounds imposed by hard constraints, independent of QP objective weights. Removing a row here is diagnostic only.',
    results=rows)
(OUT / 'acceleration_bounds.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2))
