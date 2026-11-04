"""Audit a bounded first-transfer recording; never certify complete PKL tracking."""
import argparse
import json
from pathlib import Path

import numpy as np


def sole_clearances(feet, points=None, radii=None):
    q = feet[..., 3:7]
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    x, y, z, w = np.moveaxis(q, -1, 0)
    rotation_z = np.stack((2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)), axis=-1)
    if points is not None or radii is not None:
        points, radii = np.asarray(points), np.asarray(radii)
        if (points.ndim != 3 or points.shape[0] != 2 or points.shape[-1] != 3
                or radii.shape != points.shape[:2] or not np.isfinite(points).all()
                or not np.isfinite(radii).all() or (radii < 0).any()):
            raise ValueError('Expected native collision points [2,N,3] and radii [2,N]')
        # Sphere radius is measured along world Z, independent of foot rotation.
        return (feet[...,2,None]+np.einsum('...fj,fkj->...fk',rotation_z,points)-radii).min(-1)
    corners = np.array([[-.05, -.025, -.035], [-.05, .025, -.035],
                        [.12, -.025, -.035], [.12, .025, -.035]])
    return (feet[..., 2, None] + rotation_z @ corners.T).min(-1)


def summarize_transfer(data):
    t = data['history_time'].ravel()
    mode = data['history_commanded_mode'].ravel()
    single = np.flatnonzero((mode == 1) | (mode == 2))
    result = dict(scope='first_transfer_and_lift', full_pkl_tracking=False,
                  duration=float(t[-1]), final_source_frame=float(data['history_frame'][-1]),
                  control_updates=len(t), simulation_control_hz=float(1/np.median(np.diff(t))),
                  transfer_passed=False)
    # Keep useful measurements even when the first release never happens.
    # These are diagnostics, not a replacement for the manager's sustained gates.
    recent = t >= t[-1]-1.+1e-7
    forces = data['history_foot_forces'].reshape(-1, 2, 3)[..., 2]
    positive = np.maximum(forces, 0.)
    total = positive.sum(axis=1)
    fractions = np.divide(positive, total[:, None], out=np.zeros_like(positive),
                          where=total[:, None] > 0.)
    result['load_diagnostics'] = dict(
        foot_order=['left', 'right'], final_load_n=forces[-1].tolist(),
        final_second_min_load_n=forces[recent].min(axis=0).tolist(),
        final_second_max_load_n=forces[recent].max(axis=0).tolist(),
        final_second_min_load_fraction=fractions[recent].min(axis=0).tolist(),
        final_second_max_load_fraction=fractions[recent].max(axis=0).tolist())
    for key in ('reference_phase', 'reference_rate'):
        field = 'history_'+key
        if field in data:
            result['final_'+key] = np.asarray(data[field]).ravel()[-1].item()
    com_error = np.linalg.norm((data['history_com']-data['history_com_target']).reshape(-1, 3), axis=1)
    result['final_second_max_com_error_m'] = float(com_error[recent].max())
    result['constraint_diagnostics'] = {}
    for key in ('joint_velocity_violation', 'measured_joint_velocity_violation',
                'unload_base_error', 'unload_contact_error', 'unload_friction_error',
                'unload_bound_error'):
        field = 'history_'+key
        if field in data:
            result['constraint_diagnostics']['peak_'+key] = float(np.abs(data[field]).max())
    if 'history_unload_qp_validated' in data:
        result['constraint_diagnostics']['all_qp_validated'] = bool(
            np.all(data['history_unload_qp_validated']))
    if not len(single):
        result['reason'] = 'No commanded single support'
        return result
    support = int(mode[single[0]])-1
    release = float(t[single[0]])
    recent = t >= t[-1]-1.+1e-7
    feet = data['history_feet'].reshape(-1, 2, 7)
    forces = data['history_foot_forces'].reshape(-1, 2, 3)[..., 2]
    clearance = sole_clearances(feet)
    error = np.linalg.norm((data['history_com']-data['history_com_target']).reshape(-1, 3), axis=-1)
    result.update(support='left' if support == 0 else 'right', release_time=release,
                  post_release_duration=float(t[-1]-release),
                  final_second=dict(min_swing_clearance_m=float(clearance[recent, 1-support].min()),
                                    max_swing_force_n=float(forces[recent, 1-support].max()),
                                    min_stance_force_n=float(forces[recent, support].min()),
                                    max_com_error_m=float(error[recent].max())))
    result['transfer_passed'] = bool(t[-1]-release >= 3. and recent.sum() >= .99*result['simulation_control_hz']
        and np.all(mode[recent] == support+1) and clearance[recent, 1-support].min() >= .02
        and forces[recent, 1-support].max() < 10. and forces[recent, support].min() >= 100.
        and error[recent].max() <= .03 and bool(np.all(data.get('motion_tracking_valid', False))))
    return result


def summarize_unloading(data):
    """Require a completed transfer and a stable last second; never count a step."""
    result = dict(scope='double_support_unloading_only', full_pkl_tracking=False,
                  unloading_passed=False, measured_steps=0)
    required = ('time', 'unload_elapsed', 'unload_support', 'commanded_mode',
                'foot_forces', 'com', 'com_target', 'unload_com_velocity',
                'contact_wrench_world_ankle', 'unload_effort_ratio', 'unload_qp_validated')
    if any('history_'+key not in data for key in required):
        return dict(result, reason='Missing unloading diagnostic fields')
    arrays = {key: np.asarray(data['history_'+key]) for key in required}
    if any(not np.isfinite(a).all() for a in arrays.values()):
        return dict(result, reason='Nonfinite diagnostic data')
    t = arrays['time'].ravel()
    if len(t) < 2 or np.any(np.diff(t) <= 0):
        return dict(result, reason='Insufficient or nonmonotonic recording')
    active = arrays['unload_elapsed'].ravel() >= 0
    if not active.any():
        return dict(result, reason='Transfer did not start')
    ids = arrays['unload_support'].ravel()[active]
    if ids[0] not in (0, 1) or np.any(ids != ids[0]):
        return dict(result, reason='Invalid or changing support foot')
    support = int(ids[0]); outgoing = 1-support
    recent = (t >= t[-1]-1.+1e-7) & active
    forces = arrays['foot_forces'].reshape(-1, 2, 3)[..., 2]
    predicted = arrays['contact_wrench_world_ankle'].reshape(-1, 2, 6)[..., 2]
    error = np.linalg.norm((arrays['com']-arrays['com_target']).reshape(-1, 3), axis=1)
    speed = np.linalg.norm(arrays['unload_com_velocity'].reshape(-1, 3), axis=1)
    dt = float(np.median(np.diff(t)))
    checks = dict(completed=bool(arrays['unload_elapsed'].ravel()[-1] >= 3.5-1.5*dt),
                  full_window=bool(t[recent][-1]-t[recent][0] >= 1.-1.5*dt and np.diff(t[recent]).max(initial=0) <= 1.5*dt),
                  double_support=bool(np.all(arrays['commanded_mode'].ravel()[active] == 3)),
                  unloaded=bool(np.max(np.abs(forces[recent, outgoing])) < 10.),
                  support_loaded=bool(np.min(forces[recent, support]) >= 100.),
                  com_tracking=bool(error[recent].max() <= .03),
                  body_settled=bool(speed[recent].max() <= .04),
                  qp_valid=bool(np.all(arrays['unload_qp_validated'])),
                  controller_valid=bool(np.all(data.get('motion_tracking_valid', False))))
    result.update(support=('left', 'right')[support], checks=checks,
                  initial_load_n=forces[np.flatnonzero(active)[0]].tolist(),
                  final_load_n=forces[-1].tolist(),
                  final_second=dict(max_outgoing_load_n=float(np.abs(forces[recent, outgoing]).max()),
                                    min_support_load_n=float(forces[recent, support].min()),
                                    max_com_error_m=float(error[recent].max()),
                                    max_com_speed_m_s=float(speed[recent].max()),
                                    max_prediction_error_n=float(np.abs(predicted[recent]-forces[recent]).max())),
                  peak_effort_ratio=float(arrays['unload_effort_ratio'][active].max()),
                  unloading_passed=all(checks.values()))
    for key in ('base_error', 'contact_error', 'friction_error', 'bound_error'):
        field = 'history_unload_'+key
        if field in data:
            result['peak_'+key] = float(np.abs(data[field]).max())
    return result


def plot_unloading(data, target):
    """Plot recorded measurements and model predictions without certifying them."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    t = np.asarray(data['history_time']).ravel()
    forces = data['history_foot_forces'].reshape(-1, 2, 3)[..., 2]
    predicted = data['history_contact_wrench_world_ankle'].reshape(-1, 2, 6)[..., 2]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for i, label in enumerate(('Left', 'Right')):
        axes[0, 0].plot(t, forces[:, i], label=label+' measured')
        axes[0, 0].plot(t, predicted[:, i], '--', label=label+' WBC')
    axes[0, 0].set_ylabel('World vertical force [N]')
    axes[0, 0].legend()
    error = np.linalg.norm((data['history_com']-data['history_com_target']).reshape(-1, 3), axis=1)
    axes[0, 1].plot(t, error); axes[0, 1].axhline(.03, color='r', linestyle=':')
    axes[0, 1].set_ylabel('COM error [m]')
    speed = np.linalg.norm(data['history_unload_com_velocity'].reshape(-1, 3), axis=1)
    axes[1, 0].plot(t, speed); axes[1, 0].axhline(.04, color='r', linestyle=':')
    axes[1, 0].set_ylabel('COM speed [m/s]')
    axes[1, 1].plot(t, data['history_unload_effort_ratio'].ravel())
    axes[1, 1].axhline(1., color='r', linestyle=':')
    axes[1, 1].set_ylabel('WBC torque / limit (maximum)')
    active = np.flatnonzero(data['history_unload_elapsed'].ravel() >= 0)
    for axis in axes.flat:
        axis.set_xlabel('Simulation time [s]'); axis.grid(alpha=.25)
        if len(active): axis.axvline(t[active[0]], color='k', linestyle=':', alpha=.5)
    fig.suptitle('Double-support unloading diagnostic — not step tracking')
    fig.tight_layout(); fig.savefig(target, dpi=150); plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('recording', type=Path)
    parser.add_argument('--unloading', action='store_true', help='Audit double-support unloading instead of a lift')
    parser.add_argument('--plot', action='store_true', help='Save unloading force and tracking plots')
    args = parser.parse_args()
    with np.load(args.recording) as data:
        report = summarize_unloading(data) if args.unloading else summarize_transfer(data)
        if args.plot:
            plot_unloading(data, args.recording.with_suffix('.unloading.png'))
    target = args.recording.with_suffix('.unloading.json' if args.unloading else '.transfer.json')
    target.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
