"""Sample necessary double-support dynamics conditions in a G1 recording.

Uses optimistic full soles and the runner's 80% motor budget. Feasible force
allocation does not certify contact consistency, tracking, collision freedom,
or the unsampled portions of a trajectory.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import warp as wp
from scipy.optimize import linprog

from fabrics_sim.fabrics.g1_pose_fabric import G1PoseFabric
from fabrics_sim.reference.whole_body import WholeBodyReference
from fabrics_sim.reference.inverse_dynamics import required_single_support
from fabrics_sim.solver.AccelerationQP import build_single_support_wrench_constraints
from test_g1_two_tasks import G1_JOINT_NAMES, unitree_g1_effort_limits
from g1_armature import unitree_armatures


@torch.no_grad()
def audit(data, times):
    time = data['history_time'].ravel()
    if not np.isfinite(times).all() or min(times) < time[0] or max(times) > time[-1]:
        raise ValueError('Sample times must lie inside the recorded history')
    fabric = G1PoseFabric(1, 'cpu', robot_name='g1_29dof_rev_1_0',
                         task_profile='dynamics-aware')
    model = fabric.dynamics_tasks.model
    feet = [model.kinematics.get_link_index(side+'_ankle_roll_link')
            for side in ('left', 'right')]
    effort = .8 * unitree_g1_effort_limits().astype(float)
    rows = []
    for requested_time in times:
        i = int(abs(time-requested_time).argmin())
        if int(data['history_commanded_mode'][i].item()) != 3:
            raise ValueError('This audit only supports commanded double support')
        q, v, a = [torch.tensor(data['history_whole_body_reference_'+key][i])
                   for key in ('q', 'v', 'a')]
        result = required_single_support(model, WholeBodyReference(q, v, a),
            fabric.integrate_root_position, feet[0],
            q.new_tensor(unitree_armatures(G1_JOINT_NAMES)))
        state = result['state']
        jac = torch.cat((state['origin_jac'][:, feet],
                         state['angular_jac'][:, feet]), 2).flatten(1, 2)[0]
        # Reconstruct M*a+h from the single-support decomposition; then
        # allocate BOTH feet's wrenches with the floating-base equations.
        generalized = (torch.cat((q.new_zeros(6), result['torque'][0]))
                       + jac[:6].T @ result['wrench'][0])
        cone, bound = build_single_support_wrench_constraints(
            contact_mode=torch.tensor([3]), foot_rotation_world=state['rotations'][:, feet],
            toe_length=.12, heel_length=.05, half_width=.025,
            sole_depth=.035, coupled_yaw=True)
        j, g = jac.numpy().astype(float), generalized.numpy().astype(float)
        upper = np.concatenate((cone[0].numpy(), -j[:, 6:].T, j[:, 6:].T))
        rhs = np.r_[bound[0].numpy(), effort-g[6:], effort+g[6:]]
        solution = linprog(np.zeros(12), A_ub=upper, b_ub=rhs,
            A_eq=j[:, :6].T, b_eq=g[:6], bounds=[(None, None)]*12, method='highs')
        bias = torch.cat((state['origin_jac_bias'][:, feet],
                          state['angular_jac_bias'][:, feet]), 2).flatten(1, 2)[0]
        velocity = (jac @ v[0]).reshape(2, 6)
        acceleration = (jac @ a[0]+bias).reshape(2, 6)
        row = dict(time_s=float(time[i]), wrench_allocation_feasible=bool(solution.success),
            solver_status=int(solution.status),
            max_stance_linear_speed_m_s=float(velocity[:, :3].abs().max()),
            max_stance_angular_speed_rad_s=float(velocity[:, 3:].abs().max()),
            max_stance_linear_acceleration_m_s2=float(acceleration[:, :3].abs().max()),
            max_stance_angular_acceleration_rad_s2=float(acceleration[:, 3:].abs().max()))
        if solution.success:
            row['peak_torque_budget_ratio'] = float((abs(g[6:]-j[:, 6:].T@solution.x)/effort).max())
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('recording', type=Path)
    parser.add_argument('--times', type=float, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    wp.config.kernel_cache_dir = '/tmp/g1_warp_cache'
    wp.init()
    with np.load(args.recording, allow_pickle=False) as data:
        rows = audit(data, args.times)
    report = dict(recording=str(args.recording),
        scope='Necessary sampled conditions; optimistic full soles, 0.8 real motor budget, motor armature. Not a closed-loop certificate.',
        rows=rows)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
