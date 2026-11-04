"""Read-only COM reachability under every constraint of an online QP snapshot."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linprog


def closest_task(G, b, E, e, lower, upper, C, target):
    """Minimize maximum absolute task error without relaxing any saved constraint."""
    n = len(lower)
    bounds = list(zip(lower, upper))
    exact = linprog(np.zeros(n), A_ub=G, b_ub=b,
        A_eq=np.vstack((E, C)), b_eq=np.r_[e, target], bounds=bounds, method='highs')
    A = np.vstack((np.c_[G, np.zeros(len(b))],
                   np.c_[C, -np.ones(len(target))],
                   np.c_[-C, -np.ones(len(target))]))
    rhs = np.r_[b, target, -target]
    result = linprog(np.r_[np.zeros(n), 1.], A_ub=A, b_ub=rhs,
        A_eq=np.c_[E, np.zeros(len(e))], b_eq=e,
        bounds=bounds+[(0., None)], method='highs')
    report = dict(exact_target_status=int(exact.status),
        exact_target_feasible=True if exact.success else False if exact.status == 2 else None,
        closest_status=int(result.status))
    if result.success:
        x = result.x[:n]
        violation = max(0., float(np.max(G@x-b, initial=0.)), float(np.max(abs(E@x-e), initial=0.)),
                        float(np.max(lower-x)), float(np.max(x-upper)))
        report.update(minimum_max_error=float(np.max(abs(C@x-target))),
                      best_task_output=(C@x).tolist(), constraint_violation=violation)
        if violation > 1e-4:
            raise RuntimeError(f'Offline solution violates saved constraints: {violation}')
    return report


def analyze(data):
    p = {k: np.asarray(data['problem_'+k][0], dtype=float)
         for k in ('G', 'b', 'A_eq', 'b_eq', 'lb', 'ub')}
    # Recover the EXACT masked contact Jacobian from the dynamics block,
    # including released contact rows / observed contact-frame coordinates.
    J = -p['A_eq'][:35, 35:47].T
    mass = data['mass'][0].astype(float)
    inverse = np.linalg.inv(mass)
    inactive = np.max(abs(J), axis=1) == 0
    mobility = J@inverse@J.T+np.diag(inactive.astype(float))
    force_response = -np.linalg.solve(mobility, (J@inverse)[:, 6:])
    response = inverse[:, 6:]+inverse@J.T@force_response
    transform = np.zeros((35, len(p['lb'])))
    transform[:, :35] = np.eye(35)
    transform[:, 6:35] += response*data['servo_acc_gain'][0][None, :]
    offset = response@data['servo_bias'][0]-data['correction'][0]
    com_jac = data['com_jacobian'][0].astype(float)
    C = com_jac@transform
    com_offset = com_jac@offset+data['diagnostic_com_curvature'][0]
    original = data['problem_solution'][0].astype(float)
    predicted = C@original+com_offset
    saved = data['diagnostic_com_acc_qp_predicted'][0]
    mapping_error = float(np.max(abs(predicted-saved)))
    if mapping_error > 1e-3:
        raise ValueError(f'Affine replay mismatch; snapshot not supported: {mapping_error}')
    requested = data['diagnostic_com_acc_requested'][0].astype(float)
    result = closest_task(p['G'], p['b'], p['A_eq'], p['b_eq'],
                          p['lb'], p['ub'], C, requested-com_offset)
    if 'best_task_output' in result:
        result['closest_com_acceleration'] = (np.asarray(result.pop('best_task_output'))+com_offset).tolist()
    result.update(time_s=float(data['time']), requested_com_acceleration=requested.tolist(),
                  controller_com_acceleration=predicted.tolist(),
                  controller_max_error=float(np.max(abs(predicted-requested))),
                  affine_replay_error=mapping_error,
                  scope='Offline instantaneous COM-only optimum under ALL saved hard constraints. No controller change or closed-loop stability certificate.')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('snapshot', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.snapshot, allow_pickle=False) as data:
        report = analyze(data)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
