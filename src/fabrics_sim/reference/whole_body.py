"""One full-body state supplies every motion-tracking target.

Contact anchors describe the physical environment and remain separate constraints.
This projection guarantees a common kinematic reference, not dynamic feasibility.
"""
from dataclasses import dataclass

import torch


def validate_reference_options(args):
    """Fail explicitly instead of silently ignoring independent-reference flags."""
    enabled = args.motion_pkl is not None and args.reference_mode == 'unified'
    if not enabled:
        return False
    if args.task_profile != 'dynamics-aware' or args.motion_interpolation not in ('pchip-analytic', 'bspline-analytic'):
        raise ValueError('Unified references require --task-profile dynamics-aware --motion-interpolation pchip-analytic or bspline-analytic')
    if args.reference_manager and args.reference_footsteps != 'source':
        raise ValueError('Unified managed references require --reference-footsteps source')
    independent = (
        'dynamic_reference', 'source_swing_reference', 'source_transfer_reference',
        'align_balance_pelvis', 'centroidal_preview',
        'variable_height_preview', 'source_com_endpoint', 'source_com_path',
        'support_aware_height', 'finish_com_preview', 'double_support_preview',
        'roll_aware_landing', 'fixed_joint_test', 'weight_transfer_test',
        'single_support_test', 'contact_cycle_test',
    )
    conflicts = ['--' + name.replace('_', '-') for name in independent if getattr(args, name, False)]
    if conflicts:
        raise ValueError('Independent reference modifiers require --reference-mode legacy-managed: '
                         + ', '.join(conflicts))
    return True


def support_orientation_request(feet_request, foot, contact_frame, point_support, polygon_support):
    """Reuse the same foot target for rotations freed by an observed contact.

The stance foot's ordinary task rows are masked out. A point frees all angular
coordinates, a line only its local y rotation; a polygon frees none.
"""
    from fabrics_sim.fabrics.task_request import TaskRequest
    rows = slice(6 * foot + 3, 6 * foot + 6)
    inverse = contact_frame.transpose(-1, -2)
    desired = (inverse @ feet_request.desired[:, rows, None]).squeeze(-1)
    active = torch.full_like(desired, bool(point_support), dtype=torch.bool)
    active[:, 1] = True
    if polygon_support:
        active.zero_()
    metric = inverse @ feet_request.metric[:, rows, rows] @ contact_frame
    # Congruence preserves symmetry mathematically; remove float32 matmul
    # roundoff before the strict task-contract check (without changing gains).
    metric = .5*(metric+metric.transpose(-1,-2))
    return TaskRequest(
        f'support_orientation_{foot}', inverse @ feet_request.A[:, rows],
        (inverse @ feet_request.bias[:, rows, None]).squeeze(-1), desired,
        metric, active, 'medium')


@dataclass(frozen=True)
class WholeBodyReference:
    q: torch.Tensor
    v: torch.Tensor
    a: torch.Tensor

    def __post_init__(self):
        q, v, a = self.q, self.v, self.a
        if (q.ndim != 2 or q.shape[1] != 36 or v.shape != (q.shape[0], 35)
                or a.shape != v.shape or not q.is_floating_point()
                or any(x.device != q.device or x.dtype != q.dtype
                       or not torch.isfinite(x).all() for x in (q, v, a))
                or not torch.allclose(q[:, 3:7].norm(dim=-1),
                                      q.new_ones(q.shape[0]), atol=1e-5, rtol=1e-5)):
            raise ValueError('Expected finite G1 q[B,36], v/a[B,35] and unit xyzw quaternions')
        # Neither caller mutation nor task-map buffers may change the source state.
        for name in ('q', 'v', 'a'):
            object.__setattr__(self, name, getattr(self, name).detach().clone())

    @torch.no_grad()
    def apply(self, fabric):
        """Replace all tracking references from this state, using world-frame rates."""
        tasks = fabric.dynamics_tasks
        if tasks is None:
            raise ValueError('Whole-body references require dynamics-aware tasks')
        q, v, a = self.q, self.v, self.a
        state, _, _ = tasks.model.evaluate(
            q, v, fabric.integrate_root_position, tasks.config['difference_step'])
        fabric.set_joint_posture_target(q[:, 7:].clone(), v[:, 6:].clone(), a[:, 6:].clone())
        fabric.set_pelvis_target(q[:, :3].clone(), q[:, 3:7].clone(),
                                v[:, :3].clone(), v[:, 3:6].clone(),
                                a[:, :3].clone(), a[:, 3:6].clone())
        for taskmap, setter in ((fabric.feet_pose_taskmap, fabric.set_feet_targets),
                                (fabric.hand_pose_taskmap, fabric.set_hand_targets)):
            indices = [tasks.model.kinematics.get_link_index(taskmap.frame_groups[name][0])
                       for name in taskmap.end_effector_order]
            pose = state['transforms'][:, indices]
            jv, jw = state['origin_jac'][:, indices], state['angular_jac'][:, indices]
            setter(pose[..., :3].clone(), pose[..., 3:7].clone(),
                   (jv @ v[:, None, :, None]).squeeze(-1),
                   (jw @ v[:, None, :, None]).squeeze(-1),
                   (jv @ a[:, None, :, None]).squeeze(-1) + state['origin_jac_bias'][:, indices],
                   (jw @ a[:, None, :, None]).squeeze(-1) + state['angular_jac_bias'][:, indices])
        index = tasks.model.kinematics.get_link_index('torso_link')
        jw = state['angular_jac'][:, index]
        fabric.set_body_orientation_target(
            state['transforms'][:, index, 3:7].clone(),
            (jw @ v[..., None]).squeeze(-1),
            (jw @ a[..., None]).squeeze(-1) + state['angular_jac_bias'][:, index],
            link='torso_link')
        fabric.set_center_of_mass_target(
            state['com'].clone(), (state['com_jac'] @ v[..., None]).squeeze(-1),
            (state['com_jac'] @ a[..., None]).squeeze(-1) + state['com_jac_bias'])
        fabric.set_centroidal_momentum_target(
            state['momentum'].clone(),
            (state['momentum_matrix'] @ a[..., None]).squeeze(-1) + state['momentum_matrix_bias'])
        return state
