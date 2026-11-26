"""Replay the saved failing QP on CPU; this does not measure tracking steps."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from scipy.optimize import linprog

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'acc_data/amp_walk_20261103/feedback_followup'))
from test_g1_two_tasks import load_motion_trajectory
from fabrics_sim.reference.interpolation import MotionInterpolator
from fabrics_sim.solver.admm_qp import QPData
from fabrics_sim.solver.daqp_backend import DAQPBackend
from blended_spline import BlendedSplineReference
from smooth_clock import smooth_clock
from tracking_tube import tracking_tube_constraints

source = ROOT / 'acc_data/g1_projected_acc_qp_20261114/tuning/projection1_attitude.failure.npz'
with np.load(source) as saved:
    a = {k: saved[k] for k in saved.files}
o = json.loads(str(a['controller_options'].item()))
trajectory, fps = load_motion_trajectory(ROOT / o['motion_pkl'])
trajectory[:, 2] += float(a['motion_ground_shift'])
sampler = MotionInterpolator(trajectory, method=o['tube_reference_method'])
sampler = BlendedSplineReference(sampler, trajectory, stride=o['tube_smooth_stride'],
    start_frame=o['smooth_start'] * fps, ramp_frames=o['smooth_ramp'] * fps)
_, rate, acceleration = smooth_clock(float(a['control_time']),
    *[o[k] for k in ('clock_transition_start', 'clock_transition_ramp', 'clock_final_speed',
                    'clock_second_start', 'clock_second_ramp', 'clock_second_speed',
                    'clock_third_start', 'clock_third_ramp', 'clock_third_speed')])
reference = sampler.kinematics(float(a['motion_frame_position']), fps * o['motion_speed'] * rate,
                              fps * o['motion_speed'] * acceleration)
C, d, diagnostics = tracking_tube_constraints(torch.from_numpy(a['q']), torch.from_numpy(a['qd']),
    *reference, bandwidth=o['tracking_tube_bandwidth'])
np.testing.assert_allclose(C.numpy(), a['task_acceleration_C'][:, -3:], atol=2e-6, rtol=2e-5)
np.testing.assert_allclose(d.numpy(), a['task_acceleration_d'][:, -3:], atol=2e-5, rtol=2e-5)
fields = ('H', 'h', 'G', 'b', 'lb', 'ub', 'A_eq', 'b_eq')
p = {k: a['problem_' + k][0].copy() for k in fields}
matching = np.flatnonzero(np.max(np.abs(p['G'][:, :35] - C.numpy()[0, 0]), axis=1) < 1e-7)
assert len(matching) == 1, matching
row = int(matching[0])
assert abs(float(p['b'][row]) - float(d[0, 0])) < 2e-5
keep = np.arange(len(p['b'])) != row
minimum = linprog(p['G'][row], A_ub=p['G'][keep], b_ub=p['b'][keep],
    A_eq=p['A_eq'], b_eq=p['b_eq'], bounds=list(zip(p['lb'], p['ub'])), method='highs')
assert minimum.success, minimum.message
results = []
for bandwidth in (4, 10, 15, 20, 30, 40):
    _, bound, diag = tracking_tube_constraints(torch.from_numpy(a['q']), torch.from_numpy(a['qd']),
        *reference, bandwidth=o['tracking_tube_bandwidth'], joint_bandwidth=bandwidth)
    trial = {k: v.copy() for k, v in p.items()}
    trial['b'][row] = float(bound[0, 0])
    lp = linprog(np.zeros(len(p['h'])), A_ub=trial['G'], b_ub=trial['b'],
        A_eq=trial['A_eq'], b_eq=trial['b_eq'], bounds=list(zip(p['lb'], p['ub'])), method='highs')
    solver = DAQPBackend(single_solve=True)
    solution = solver.solve(QPData(**{k: torch.from_numpy(v[None]) for k, v in trial.items()}))
    assert solver.last_solve_counts == [1]
    results.append(dict(joint_bandwidth=bandwidth, bound=float(bound[0, 0]),
        lp_feasible=bool(lp.success), single_qp_accepted=bool(solution.status_ok.item()),
        solver_status=solver.last_status, native_calls=solver.last_solve_counts,
        barrier_diagnostics=diag[0, 0].tolist()))
report = dict(kind='offline_saved_state_only', tracking_evaluated=False,
    source=str(source.relative_to(ROOT)), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    source_time_s=float(a['control_time']), reconstructed_tube_matches=True,
    joint_tube_row=row, minimum_feasible_bound=float(minimum.fun),
    original_bound=float(p['b'][row]), original_diagnostics=diagnostics[0, 0].tolist(), results=results)
error = torch.from_numpy(a['q'][:, 7:]) - reference[0][:, 7:]
velocity = torch.from_numpy(a['qd'][:, 6:]) - reference[1][:, 6:]
h, hdot, _ = diagnostics[0, 0].tolist()
report['joint_bound_components'] = dict(
    reference_acceleration=float(2 * (error * reference[2][:, 6:]).sum() / 29),
    relative_velocity=float(-2 * velocity.square().sum() / 29),
    barrier_rate_term=2 * o['tracking_tube_bandwidth'] * hdot,
    barrier_position_term=o['tracking_tube_bandwidth'] ** 2 * h,
    relative_velocity_rms_rad_s=float(velocity.square().mean().sqrt()))
previous_velocity = a['history_qd'][-1, 0, 6:]
dt = float(a['control_time']) - float(a['history_time'][-1])
predicted_velocity = previous_velocity + dt * a['history_qdd_qp'][-1, 0, 6:]
report['largest_relative_joint_velocities'] = [dict(
    joint=str(a['joint_names'][j]), actual_rad_s=float(a['qd'][0, j+6]),
    reference_rad_s=float(reference[1][0, j+6]),
    previous_tick_prediction_rad_s=float(predicted_velocity[j]),
    error_rad_s=float(velocity[0, j]))
    for j in np.argsort(-np.abs(velocity.numpy()[0]))[:6]]
report['last_interval_s'] = dt
Path(__file__).with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
