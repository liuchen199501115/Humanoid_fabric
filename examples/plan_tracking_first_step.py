"""Audit the handoff from a validated walk1 transfer to source frames 114 -> 129.

This checks a nominal candidate; it does not reset measured state or execute it.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from analyze_contact_transfer import summarize_transfer


def handoff_errors(data, names):
    q = np.asarray(data['history_q'])[-1].reshape(36)
    v = np.asarray(data['history_qd'])[-1].reshape(35)
    rq = np.asarray(data['history_whole_body_reference_q'])[-1].reshape(36)
    rv = np.asarray(data['history_whole_body_reference_v'])[-1].reshape(35)
    if len(names) != 29 or not all(np.isfinite(x).all() for x in (q, v, rq, rv)):
        raise ValueError('Handoff requires finite measured and reference G1 states')
    joint = np.abs(q[7:]-rq[7:])
    worst = int(joint.argmax())
    qa, qb = q[3:7], rq[3:7]
    if min(np.linalg.norm(qa), np.linalg.norm(qb)) < 1e-8:
        raise ValueError('Invalid handoff quaternion')
    angle = 2*np.arccos(np.clip(abs(np.dot(qa, qb))/(np.linalg.norm(qa)*np.linalg.norm(qb)), 0., 1.))
    return dict(max_joint_error_rad=float(joint[worst]), worst_joint=names[worst],
                root_position_error_m=float(np.linalg.norm(q[:3]-rq[:3])),
                root_orientation_error_rad=float(angle),
                max_velocity_error=float(np.max(abs(v-rv))),
                measured_max_velocity=float(np.max(abs(v))),
                scope='Measured state minus nominal handoff; no state reset or feasibility certificate')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('recording', type=Path)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--measured-upper-body', action='store_true',
                        help='Experimental offline candidate from measured q/v; hold measured waist/arms')
    parser.add_argument('--landing-height', type=float)
    parser.add_argument('--stop-duration', type=float, default=.25)
    args = parser.parse_args()
    if not np.isfinite(args.stop_duration) or args.stop_duration < 0:
        raise ValueError('Stop duration must be finite and nonnegative')
    if args.landing_height is not None and (not np.isfinite(args.landing_height) or args.landing_height <= 0):
        raise ValueError('Landing height must be finite and positive')
    if args.output.exists():
        raise FileExistsError(args.output)
    import torch
    import warp as wp
    from fabrics_sim.fabrics.g1_pose_fabric import G1PoseFabric
    from fabrics_sim.reference.whole_body import WholeBodyReference
    from fabrics_sim.reference.source_step_rollout import SourceStepRollout
    from fabrics_sim.reference.collision_audit import collision_reference_check
    from test_g1_two_tasks import G1_JOINT_NAMES, load_motion_trajectory
    from g1_armature import unitree_armatures
    report = dict(full_pkl_tracking=False, simulation_validated=False,
                  recording=str(args.recording), first_step_preflight_passed=False)
    with np.load(args.recording, allow_pickle=False) as data:
        report['transfer_audit'] = summarize_transfer(data)
        options = json.loads(str(data['controller_options'].item()))
        if not report['transfer_audit']['transfer_passed']:
            raise ValueError('A validated transfer recording is required')
        if (options.get('source_arm_roll_offset', 0) or options.get('convex_wrist_collision')
                or not all(options.get(k) for k in ('wrist_hip_avoidance', 'wrist_body_avoidance', 'wrist_hip_barrier'))):
            raise ValueError('This preflight requires the unchanged sphere-collision baseline')
        if not 114 <= report['transfer_audit']['final_source_frame'] < 115:
            raise ValueError('Expected the first walk1 source takeoff at frame 114')
        report['handoff'] = handoff_errors(data, G1_JOINT_NAMES)
        initial = WholeBodyReference(*[torch.from_numpy(data['history_whole_body_reference_'+k][-1].copy()) for k in ('q', 'v', 'a')])
        measured_q = torch.from_numpy(data['history_q'][-1].copy())
        measured_v = torch.from_numpy(data['history_qd'][-1].copy())
        shift = float(data['motion_ground_shift'])
    wp.config.kernel_cache_dir='/tmp/g1_warp_cache'
    wp.init(); torch.set_grad_enabled(False)
    fabric = G1PoseFabric(1, 'cpu', robot_name='g1_29dof_rev_1_0', task_profile='dynamics-aware')
    fabric.dynamics_tasks.enable_wrist_hip_avoidance(include_upper_body=True)
    fabric.dynamics_tasks.wrist_hip_barrier = True
    source, _ = load_motion_trajectory(Path(options['motion_pkl']))
    source[:, 2] += shift
    report['nominal_handoff_collision'] = collision_reference_check(fabric, initial)
    report['measured_pose_static_collision'] = collision_reference_check(fabric,
        WholeBodyReference(measured_q, torch.zeros_like(initial.v), torch.zeros_like(initial.a)))
    report['measured_pose_static_collision']['scope'] = 'Static pose check; measured acceleration is not assumed zero in physics'
    candidate_options = json.loads(args.profile.read_text())['first']
    if args.landing_height is not None:
        candidate_options['landing_height'] = args.landing_height
    if args.measured_upper_body:
        initial = WholeBodyReference(measured_q, measured_v, torch.zeros_like(measured_v))
        source = source.clone()
        source[:, 19:] = measured_q[:, 19:]
    report['candidate_initialization'] = dict(measured_qv=args.measured_upper_body,
        measured_waist_arms_held=args.measured_upper_body,
        acceleration_assumption='zero (not measured)' if args.measured_upper_body else 'nominal reference',
        stop_duration=args.stop_duration, candidate_options=candidate_options,
        executable_handoff_validated=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    candidate_path = args.output.with_suffix('.candidate.json')
    report['candidate_report'] = str(candidate_path)
    try:
        rollout = SourceStepRollout(fabric, initial, G1_JOINT_NAMES, source, 114, 129, 0, .002,
            source.new_tensor(unitree_armatures(G1_JOINT_NAMES)), candidate_path,
            candidate_options=candidate_options, stop_duration=args.stop_duration)
        report['planned_start_handoff'] = handoff_errors(dict(
            history_q=measured_q.numpy()[None], history_qd=measured_v.numpy()[None],
            history_whole_body_reference_q=rollout.references[0].q.numpy()[None],
            history_whole_body_reference_v=rollout.references[0].v.numpy()[None]), G1_JOINT_NAMES)
        report['first_step_preflight_passed'] = True
        np.savez_compressed(args.output.with_suffix('.candidate.npz'),
            **{k: np.asarray([r[k] for r in rollout.rows]) for k in ('q', 'v', 'a', 'frame', 'time')})
    except (RuntimeError, ValueError) as exc:
        report['rejection'] = str(exc)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    return 0 if report['first_step_preflight_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
