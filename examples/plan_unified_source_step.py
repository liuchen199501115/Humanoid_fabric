"""Offline source-sized first-step candidate. This command never runs a robot."""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import torch
import warp as wp

from fabrics_sim.fabrics.g1_pose_fabric import G1PoseFabric
from fabrics_sim.reference.footstep import source_grounded_step
from fabrics_sim.reference.source_step import SourceStepCandidate, centroidal_support_check
from fabrics_sim.reference.whole_body import WholeBodyReference
from test_g1_two_tasks import load_motion_trajectory, G1_JOINT_NAMES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recording', type=Path, required=True, help='Successful unified transfer recording')
    parser.add_argument('--motion-pkl', type=Path, default=Path('retarget/lafan/walk1_g1.pkl'))
    parser.add_argument('--start-frame', type=int, default=114)
    parser.add_argument('--end-frame', type=int, default=129)
    parser.add_argument('--duration', type=float, default=.8)
    parser.add_argument('--landing-height', type=float, default=.62)
    parser.add_argument('--swing-phase-blend', type=float, default=0., help='Blend source swing phase toward uniform physical time')
    parser.add_argument('--body-interpolation', choices=('pchip','cubic'), default='pchip',
                        help='Cubic gives C2 body reference but requires overshoot/limit audit')
    parser.add_argument('--landing-search-depth', type=float, default=0.,
                        help='Bounded virtual sole penetration in final .2 s; requires measured touchdown, max .01 m')
    parser.add_argument('--landing-fraction', type=float, default=1.,
                        help='Explicit bridge step: fraction of source planar landing displacement, in (0,1]')
    parser.add_argument('--landing-com-fraction', type=float, default=.5,
                        help='COM endpoint fraction from stance to landing foot')
    parser.add_argument('--terminal-capture', choices=('landing', 'double_support'), default='landing',
                        help='Double support requires a subsequent load-transfer plan')
    parser.add_argument('--pelvis-pitch-offset', type=float, default=0.,
                        help='Explicit local pelvis pitch adaptation at source endpoint, radians')
    parser.add_argument('--body-smoothing-passes', type=int, default=0, help='Explicit source upper-body reference smoothing')
    parser.add_argument('--qp-acceleration-limit-scale', type=float, default=5., help='Match the runtime software QP acceleration bounds')
    parser.add_argument('--full-dynamics', action='store_true', help='Include torque and full contact wrench bounds in correction merit')
    parser.add_argument('--momentum-iterations', type=int, default=0, help='Offline whole-body COM correction iterations')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.duration <= 0 or not np.isfinite(args.duration):
        parser.error('--duration must be finite and positive')
    if not np.isfinite(args.qp_acceleration_limit_scale) or args.qp_acceleration_limit_scale <= 0:
        parser.error('--qp-acceleration-limit-scale must be finite and positive')
    if args.momentum_iterations < 0:
        parser.error('--momentum-iterations must be nonnegative')
    if args.output.suffix != '.npz':
        parser.error('--output must end in .npz')
    wp.config.kernel_cache_dir='/tmp/g1_warp_cache'; wp.init()
    torch.set_grad_enabled(False)
    fabric = G1PoseFabric(1, 'cpu', robot_name='g1_29dof_rev_1_0', task_profile='dynamics-aware')
    model = fabric.dynamics_tasks.model
    with np.load(args.recording) as record:
        mode = int(record['history_commanded_mode'][-1].item())
        if mode not in (1, 2) or not record['motion_tracking_valid'].all():
            raise ValueError('Require a passed transfer recording with single support')
        initial = WholeBodyReference(*[torch.from_numpy(record['history_whole_body_reference_'+key][-1]) for key in ('q', 'v', 'a')])
        shift = float(record['motion_ground_shift'])
    source, fps = load_motion_trajectory(args.motion_pkl)
    if not 0 <= args.start_frame < args.end_frame < len(source):
        parser.error('Expected a source interval inside the supplied PKL')
    source[:, 2] += shift
    support = mode-1
    feet = [model.kinematics.get_link_index(side+'_ankle_roll_link') for side in ('left', 'right')]
    goal = source_grounded_step(model.snapshot(source[args.end_frame:args.end_frame+1])['transforms'][0, feet[1-support]].numpy())
    goal = source.new_tensor(goal)[None]
    plan = SourceStepCandidate(fabric, initial, G1_JOINT_NAMES, source,
                               args.start_frame, args.end_frame, support, goal,
                               duration=args.duration, landing_height=args.landing_height,
                               body_smoothing_passes=args.body_smoothing_passes, swing_phase_blend=args.swing_phase_blend,
                               landing_com_fraction=args.landing_com_fraction, terminal_capture=args.terminal_capture,
                               pelvis_pitch_offset=args.pelvis_pitch_offset, body_interpolation=args.body_interpolation, landing_fraction=args.landing_fraction, landing_search_depth=args.landing_search_depth)
    limits = {node.attrib['name']:node.find('limit') for node in ET.parse(fabric.urdf_path).getroot().findall('joint')}
    velocity_limits = source.new_tensor([float(limits[name].attrib['velocity']) for name in G1_JOINT_NAMES])
    dynamics_limits = None
    if args.full_dynamics:
        from g1_armature import unitree_armatures
        dynamics_limits = dict(armature=torch.from_numpy(unitree_armatures(G1_JOINT_NAMES)),
            effort=source.new_tensor([float(limits[name].attrib['effort']) for name in G1_JOINT_NAMES]),
            acceleration=args.qp_acceleration_limit_scale*fabric.accel_limits,
            joint_bounds=dict(lower=source.new_tensor([float(limits[n].attrib['lower']) for n in G1_JOINT_NAMES]),
                upper=source.new_tensor([float(limits[n].attrib['upper']) for n in G1_JOINT_NAMES]),dt=.002,bandwidth=8.))
    times = np.linspace(0., args.duration, max(2, int(np.ceil(args.duration*500))+1))
    rows = []; error = None; optimization = None
    if args.momentum_iterations or args.full_dynamics:
        from fabrics_sim.reference.source_step_optimization import correct_candidate
        rows, optimization = correct_candidate(plan, times, velocity_limits, args.momentum_iterations,
            progress=lambda item: print(json.dumps(item), flush=True), dynamics_limits=dynamics_limits)
    for time in ([] if optimization is not None else times):
        try:
            reference = plan.sample(float(time))
        except (ValueError, RuntimeError) as exc:
            error = dict(time=float(time), reason=str(exc)); break
        state, _, _ = model.evaluate(reference.q, reference.v, fabric.integrate_root_position,
                                     fabric.dynamics_tasks.config['difference_step'])
        force, cop, wrench_passed = centroidal_support_check(model, reference, state, feet[support])
        rows.append(dict(time=float(time), q=reference.q.numpy(), v=reference.v.numpy(), a=reference.a.numpy(),
                         frame=args.start_frame+(args.end_frame-args.start_frame)*plan.progress(float(time))[3][0],
                         com=state['com'].numpy(), com_velocity=(state['com_jac']@reference.v[..., None]).squeeze(-1).numpy(),
                         cop=cop.numpy(), force=force.numpy(), wrench_passed=bool(wrench_passed.all()),
                         velocity_ratio=float((reference.v[:, 6:].abs()/velocity_limits).max())))
    report = dict(landing_search_depth=args.landing_search_depth,
                  contact_search_requires_measured_load=True, landing_fraction=args.landing_fraction, source_landing_xy=goal[0,:2].tolist(),
                  landing_deviation_m=float((plan.landing_pose[0,:2]-goal[0,:2]).norm()), body_interpolation=args.body_interpolation, landing_com_fraction=args.landing_com_fraction, terminal_capture=args.terminal_capture,
                  pelvis_pitch_offset=args.pelvis_pitch_offset, qp_acceleration_limit_scale=args.qp_acceleration_limit_scale,full_dynamics_in_optimization=args.full_dynamics,momentum_optimization=optimization, scope='offline_first_source_step_candidate', simulation_validated=False,
                  ready_for_tracking=False, full_pkl_tracking=False,
                  source=str(args.motion_pkl), start_frame=args.start_frame, end_frame=args.end_frame,
                  duration=args.duration, landing_height=args.landing_height, body_smoothing_passes=args.body_smoothing_passes, swing_phase_blend=args.swing_phase_blend,
                  sampled_kinematics_passed=error is None, sample_count=len(rows), requested_samples=len(times), error=error,
                  landing_xy=plan.landing_pose[0, :2].tolist(), source_fps=fps,
                  requires_landing_and_next_step=True)
    if rows:
        cop_values = np.asarray([r['cop'] for r in rows]).reshape(-1, 2)
        report.update(centroidal_wrench_pass_fraction=float(np.mean([r['wrench_passed'] for r in rows])),
                      outside_physical_sole_fraction=float(np.mean(
                          (cop_values[:, 0] < -.05) | (cop_values[:, 0] > .12) | (abs(cop_values[:, 1]) > .025))),
                      max_joint_velocity_ratio=max(r['velocity_ratio'] for r in rows),
                      final_com_velocity=rows[-1]['com_velocity'].ravel().tolist(),
                      cop_min=np.min([r['cop'] for r in rows], axis=0).ravel().tolist(),
                      cop_max=np.max([r['cop'] for r in rows], axis=0).ravel().tolist())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **{k:np.asarray([row[k] for row in rows]) for k in (rows[0] if rows else {})},
                        ready_for_tracking=False, full_pkl_tracking=False, metadata=json.dumps(report))
    args.output.with_suffix('.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
