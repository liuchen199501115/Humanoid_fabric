"""Offline task comparisons under the exact saved physical constraints.

Only direct-WBC snapshots are supported. This script never sends torques and
does not relax physical constraints. A feasible instantaneous task request is
not proof that its resulting trajectory can be tracked.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
from scipy import sparse
import osqp


def diagnose(path):
    with np.load(path, allow_pickle=False) as data:
        options = json.loads(str(data['controller_options']))
        if not options.get('direct_wbc'):
            raise ValueError('This diagnostic requires a direct-WBC snapshot')
        prefix = ('priority_level_0_' if 'priority_level_0_H' in data else
                  'priority_problem_' if 'priority_problem_H' in data else 'problem_')
        problem = {k: data[prefix+k][0].astype(float) for k in
                   ('H', 'h', 'G', 'b', 'lb', 'ub', 'A_eq', 'b_eq')}
        tracking_H = data['problem_H'][0].astype(float)
        tracking_h = data['problem_h'][0].astype(float)
        correction = data['correction'][0].astype(float)
        names = [k[5:-2] for k in data.files if k.startswith('task_') and k.endswith('_A')]
        tasks = {}
        for name in names:
            A, bias, desired, metric, active = [data['task_'+name+'_'+field][0] for field in
                                               ('A', 'bias', 'desired', 'metric', 'active')]
            tasks[name] = (A.astype(float), desired-bias+A@correction,
                           metric*active[:, None]*active[None, :], active)
        stamp = float(data['time'])
    if 'center_of_mass' not in tasks:
        raise ValueError('No serialized task requests; record a new --diagnose-qp-time snapshot')
    n = len(problem['h'])
    bounds = list(zip(problem['lb'], problem['ub']))
    def feasible(extra=False):
        eq, rhs = problem['A_eq'], problem['b_eq']
        if extra:
            A, goal, _, _ = tasks['center_of_mass']
            row = np.zeros((3, n)); row[:, :35] = A
            eq, rhs = np.vstack((eq, row)), np.r_[rhs, goal]
        result = linprog(np.zeros(n), A_ub=problem['G'], b_ub=problem['b'],
                         A_eq=eq, b_eq=rhs, bounds=bounds, method='highs')
        return dict(feasible=bool(result.success), status=result.message)
    result = dict(time=stamp, unchanged_physical_constraints=feasible(),
                  exact_requested_com=feasible(True), comparisons={})
    matrix = sparse.csc_matrix(np.vstack((problem['A_eq'], problem['G'], np.eye(n))))
    lo = np.r_[problem['b_eq'], np.full(len(problem['b']), -np.inf), problem['lb']]
    hi = np.r_[problem['b_eq'], problem['b'], problem['ub']]
    groups = {'saved_priority': None, 'com': ['center_of_mass'],
              'com_and_swing': ['center_of_mass', 'feet_pose'],
              'com_swing_torso': ['center_of_mass', 'feet_pose', 'body_orientation']}
    if 'source_pelvis_orientation' in tasks:
        groups['com_swing_torso_pelvis'] = ['center_of_mass', 'feet_pose',
                                           'body_orientation', 'source_pelvis_orientation']
    for label, selected in groups.items():
        H, h = problem['H'].copy(), problem['h'].copy()
        if selected is not None:
            H, h = np.eye(n)*1e-6, np.zeros(n)
            for name in selected:
                A, goal, W, _ = tasks[name]
                H[:35, :35] += A.T@W@A
                h[:35] -= A.T@W@goal
        solver = osqp.OSQP()
        solver.setup(P=sparse.triu(sparse.csc_matrix(H), format='csc'), q=h,
                     A=matrix, l=lo, u=hi, verbose=False, max_iter=100000,
                     eps_abs=1e-6, eps_rel=1e-6, eps_prim_inf=1e-8, eps_dual_inf=1e-8,
                     polish=True, adaptive_rho_interval=25)
        solved = solver.solve()
        entry = dict(status=solved.info.status)
        if solved.info.status_val in (1, 2):
            x = solved.x
            entry['maximum_physical_violation'] = float(max(0., (lo-matrix@x).max(), (matrix@x-hi).max()))
            entry['task_residuals'] = {name: ((A@x[:35]-goal)*active).tolist()
                                      for name, (A, goal, _, active) in tasks.items()
                                      if name not in ('collision', 'joint_limit_avoidance')}
            if selected is not None and entry['maximum_physical_violation'] <= 1e-3:
                # Reproduce the actual two-stage policy: lock achieved task
                # accelerations, then optimize ALL original tracking tasks.
                # The first-stage minimum-norm posture alone says little about
                # the final pelvis/arm trajectory because of redundancy.
                rows = np.vstack([tasks[name][0][tasks[name][3] &
                                     (np.diag(tasks[name][2]) > 0)] for name in selected])
                lock = np.zeros((len(rows), n)); lock[:, :35] = rows
                achieved = lock@x
                full_matrix = sparse.vstack((matrix, sparse.csc_matrix(lock)), format='csc')
                lower, upper = np.r_[lo, achieved-.02], np.r_[hi, achieved+.02]
                tracking_solver = osqp.OSQP()
                tracking_solver.setup(P=sparse.triu(sparse.csc_matrix(tracking_H), format='csc'),
                    q=tracking_h, A=full_matrix, l=lower, u=upper, verbose=False,
                    max_iter=100000, eps_abs=1e-6, eps_rel=1e-6,
                    eps_prim_inf=1e-8, eps_dual_inf=1e-8, polish=True, adaptive_rho_interval=25)
                final = tracking_solver.solve()
                second = dict(status=final.info.status)
                if final.info.status_val in (1, 2):
                    second['maximum_constraint_violation'] = float(max(0.,
                        (lower-full_matrix@final.x).max(), (full_matrix@final.x-upper).max()))
                    second['task_residuals'] = {name: ((A@final.x[:35]-goal)*active).tolist()
                        for name, (A, goal, _, active) in tasks.items()
                        if name not in ('collision', 'joint_limit_avoidance')}
                entry['tracking_stage'] = second
        result['comparisons'][label] = entry
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('snapshot', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = json.dumps(diagnose(args.snapshot), indent=2)+'\n'
    if args.output:
        args.output.write_text(result)
    print(result, end='')
