"""Full-body static load lower bound at a recorded pose; not a controller.

Assume v=a=0 only for this offline equilibrium question. Online floating-base
acceleration remains a WBC variable. Full soles are an optimistic contact
assumption; this does not certify a moving transfer or the observed patches.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
import torch
import warp as wp

from fabrics_sim.fabrics.g1_pose_fabric import G1PoseFabric
from fabrics_sim.reference.inverse_dynamics import required_single_support
from fabrics_sim.reference.whole_body import WholeBodyReference
from fabrics_sim.solver.AccelerationQP import build_single_support_wrench_constraints
from test_g1_two_tasks import G1_JOINT_NAMES, unitree_g1_effort_limits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('record', type=Path)
    parser.add_argument('--outgoing', choices=('left', 'right'), default='right')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    wp.config.kernel_cache_dir = '/tmp/g1_warp_cache'
    wp.init()
    torch.set_grad_enabled(False)
    fabric = G1PoseFabric(1, 'cpu', robot_name='g1_29dof_rev_1_0', task_profile='dynamics-aware')
    with np.load(args.record) as record:
        q = torch.tensor(record['history_q'][-1])
        time = float(record['history_time'][-1])
        frame = float(record['history_frame'][-1])
        measured = record['history_foot_forces'][-1].reshape(2, 3)[:, 2].tolist()
    model = fabric.dynamics_tasks.model
    feet = [model.kinematics.get_link_index(side+'_ankle_roll_link') for side in ('left', 'right')]
    zero = q.new_zeros((1, 35))
    result = required_single_support(model, WholeBodyReference(q, zero, zero),
                                     fabric.integrate_root_position, feet[0])
    state = result['state']
    jac = torch.cat((state['origin_jac'][:, feet], state['angular_jac'][:, feet]), 2).flatten(1, 2)[0]
    generalized = torch.cat((q.new_zeros(6), result['torque'][0])) + jac[:6].T @ result['wrench'][0]
    cone, bound = build_single_support_wrench_constraints(
        contact_mode=torch.tensor([3]), foot_rotation_world=state['rotations'][:, feet],
        toe_length=.12, heel_length=.05, half_width=.025, sole_depth=.035, coupled_yaw=True)
    effort = .8*unitree_g1_effort_limits().astype(float)
    j = jac.numpy().astype(float)
    g = generalized.numpy().astype(float)
    G, b = cone[0].numpy().astype(float), bound[0].numpy().astype(float)
    objective = np.zeros(12)
    objective[6*('left', 'right').index(args.outgoing)+2] = 1.
    cases = {}
    for name, torque_bounded in (('full_model_with_motor_limits', True),
                                  ('contact_only_diagnostic', False)):
        A = np.concatenate((G, -j[:, 6:].T, j[:, 6:].T)) if torque_bounded else G
        rhs = np.concatenate((b, effort-g[6:], effort+g[6:])) if torque_bounded else b
        solution = linprog(objective, A_ub=A, b_ub=rhs, A_eq=j[:, :6].T, b_eq=g[:6],
                           bounds=[(None, None)]*12, method='highs')
        case = dict(feasible=bool(solution.success), solver_message=solution.message)
        if solution.success:
            f = solution.x
            tau = g[6:]-j[:, 6:].T@f
            case.update(minimum_outgoing_world_vertical_force_n=float(objective@f),
                        minimum_outgoing_fraction=float((objective@f)/(f[2]+f[8])),
                        world_ankle_wrench=f.reshape(2, 6).tolist(),
                        peak_motor_limit_ratio=float(np.max(np.abs(tau)/effort)),
                        base_equilibrium_residual=float(np.max(np.abs(j[:, :6].T@f-g[:6]))),
                        inequality_violation=float(np.maximum(A@f-rhs, 0).max()))
        cases[name] = case
    report = dict(scope='full_body_static_equilibrium_at_recorded_pose', record=str(args.record),
                  time=time, source_frame=frame, outgoing=args.outgoing, measured_vertical_load_n=measured,
                  assumptions=['zero velocity and acceleration for offline equilibrium only',
                               'full rectangular soles, not observed contact patches',
                               'motor limits at 0.8 times online simulated effort table',
                               'contact-only case omits motor limits solely to isolate constraints'],
                  effort_joint_names=G1_JOINT_NAMES, effort_limits=effort.tolist(),
                  cases=cases, dynamic_transfer_certified=False, full_four_step_tracking_passed=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
