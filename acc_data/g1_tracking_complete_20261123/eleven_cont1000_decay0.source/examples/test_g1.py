"""Track motion PKLs using Fabric tasks and a constrained dynamics QP.

Full AMP attempt from the repository root:
    PYTHONPATH=src:examples:.runtime/qp_solvers python examples/test_g1.py \
        --tracking-profile amp-full
The historical two-stage baseline verified 12/29 landings. The current
amp-full profile projects the returned Fabric acceleration onto contact
constraints, then tracks that projected acceleration in one QP,
without caching task requests, with one native solve attempt per tick;
its tracking performance must be measured separately. This entry retains the
complete source, native contacts and original error limits.
Historical experiment helpers under acc_data are required by this profile.

With --g1-md, use natural-form task summation as the dynamics objective,
persistent joint integration and MIT effort output at 1 kHz / 2 kHz PhysX.
Contact feasibility remains checked; see docs/g1_md_isaac.md.

Default data flow:
    qdd_fabric [1, 35]
        -> Isaac Sim M [1, 35, 35] and foot J [1, 12, 35]
        -> dynamically consistent projection
        -> qdd_contact_consistent [1, 35]
        -> floating-base dynamics acceleration QP
        -> qdd_dynamics_feasible [1, 35]
        -> actuated-joint semi-implicit integration [1, 29]

For a clip that begins mid-motion, --motion-start-in-motion initializes
source q/v and near-ground soles, primes fresh PhysX contacts, and follows
the source clock without a stationary warmup. It does not certify tracking.

With a motion PKL, the closed-loop simulation reconciles planned contacts
with measured foot forces. PKL poses provide task references; joint commands
come from integrating the QP command acceleration without hard resets.
Its bounded synchronization correction is excluded from physical feedforward
acceleration, but remains inside the QP's command and total-effort constraints.
The QP also predicts the contact-consistent response to explicit feedback.
Explicit 500 Hz joint PD
and QP feedforward are combined and torque-limited before being sent to
PhysX, whose internal joint drives remain disabled. Numerical feasibility does not establish
motion tracking success; the final state and sampled history are saved for
separate tracking validation.
"""

from __future__ import annotations

import argparse
from functools import partial
import json
import math
from pathlib import Path
import sys
import time
import traceback
import warnings

# Keep the tested reference/contact helpers shared with the recorded baseline.
_EXPERIMENT_HELPERS = (Path(__file__).resolve().parents[1]
    / "acc_data/amp_walk_20261103/feedback_followup")
sys.path.append(str(_EXPERIMENT_HELPERS))

import numpy as np
import torch
import warp as wp

# Warp 1.8.1 probes ``tensor.grad`` while wrapping Torch tensors. PyTorch
# warns when that tensor is non-leaf even though this controller never calls
# backward(). Suppress only this known bridge warning; all other warnings stay
# visible.
warnings.filterwarnings(
    "ignore",
    message=(
        r"The \.grad attribute of a Tensor that is not a leaf Tensor "
        r"is being accessed\..*"
    ),
    category=UserWarning,
    module=r"warp\.torch",
)

from fabrics_sim.fabrics.g1_pose_fabric import G1PoseFabric
from g1_acceleration_cost import linear_acceleration_weight, preserve_com_acceleration, track_com_feedback, com_feedback_scale
from fabrics_sim.solver.contact_observation import ground_support_forces
from fabrics_sim.solver.AccelerationQP import split_foot_wrench_constraints
from fabrics_sim.solver.com_diagnostics import com_acceleration_snapshot, complete_observed_interval
from fabrics_sim.solver.swing_clearance import clearance_active, sole_clearance_constraints
from fabrics_sim.solver.rigid_body_frames import (
    origin_to_com_map, dynamics_at_origin, jacobians_at_origins,
)
from fabrics_sim.solver.AccelerationQP import (
    G1DynamicsAccelerationQP,
    build_single_support_wrench_constraints,
)
from fabrics_sim.solver.joint_control import (
    explicit_pd_gains, integration_acceleration_bounds, integrated_servo_model, servo_health,
    synchronize_command_acceleration,
    bounded_reference_step,
    measured_joint_acceleration_bounds,
    joint_velocity_constraints,
    joint_limit_avoidance_request,
    contact_servo_gain_bank,
    implicit_servo_response,
)
from fabrics_sim.solver.dynamic_consistent_gpu import (
    BatchedFootContactDetector,
    ContactDetectorConfig,
    G1FabricContactProjector,
    ProjectionConfig,
)
from fabrics_sim.taskmaps.g1_end_effector_pose_taskmap import (
    _quat_xyzw_to_matrix,
    _so3_log,
)

from test_g1_two_tasks import (
    FABRIC_TIMESTEP,
    G1_JOINT_NAMES,
    G1_USD_PATH,
    PD_TIMESTEP,
    load_motion_trajectory,
    load_motion_contact_sequence,
    unitree_g1_effort_limits,
    unitree_g1_pd_gains,
)


LEFT_FOOT_BODY = "left_ankle_roll_link"
RIGHT_FOOT_BODY = "right_ankle_roll_link"
_active_simulation_app = None
OUTPUT_PATH = (
    Path(__file__).resolve().parents[1]
    / "acc_data/g1_two_task_dynamics_feasible.npz"
)


def rolling_sphere_bias_correction(foot_rotation, angular_velocity, patch_frame,
                                   local_points, collider_centers, collider_radii,
                                   tolerance=.002):
    """Account for migration of spherical floor contacts, not impact impulses.

    A sphere contact offset from its center is -radius * world_z, so its
    world derivative is zero. A rigid material-point Jdot*v instead includes
    omega x (omega x offset). Subtract that term; the instantaneous contact
    Jacobian and wrench map remain unchanged. Apply only when every support
    vertex matches a native spherical collider's current lowest point.
    """
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('Contact matching tolerance must be finite and positive')
    points = (foot_rotation[:, None] @ local_points[..., None]).squeeze(-1)
    centers = (foot_rotation[:, None] @ collider_centers[None, ..., None]).squeeze(-1)
    vertical = angular_velocity.new_tensor([0., 0., 1.])
    bottoms = centers - collider_radii[None, :, None] * vertical
    distance = torch.linalg.vector_norm(points[:, :, None] - bottoms[:, None], dim=-1)
    distance = distance.masked_fill(collider_radii[None, None] <= 0, float('inf'))
    nearest_distance, nearest = distance.min(dim=-1)
    matched = (nearest_distance <= tolerance).all(dim=-1)
    radius = collider_radii[nearest].mean(dim=-1)
    correction_world = radius[:, None] * torch.linalg.cross(
        angular_velocity, torch.linalg.cross(angular_velocity,
            vertical.expand_as(angular_velocity), dim=-1), dim=-1)
    correction = (patch_frame.transpose(-1, -2) @ correction_world[..., None]).squeeze(-1)
    return torch.where(matched[:, None], correction, torch.zeros_like(correction)), matched

def step_physics_interval(world, torque_dt, substeps, apply_effort=None):
    """Hold the existing effort command over one interval of native physics steps."""
    if (not isinstance(substeps, int) or isinstance(substeps, bool) or substeps < 1
            or not math.isfinite(torque_dt) or torque_dt <= 0):
        raise ValueError('Expected a positive torque interval and integer physics substeps')
    native_dt = torque_dt / substeps
    if not math.isclose(world.get_physics_dt(), native_dt, rel_tol=1e-7, abs_tol=1e-10):
        world.set_simulation_dt(physics_dt=native_dt, rendering_dt=world.get_rendering_dt())
    before = world.current_time
    for _ in range(substeps):
        if apply_effort is not None:
            apply_effort()
        world.step(render=False)
    elapsed = world.current_time - before
    if not math.isclose(elapsed, torque_dt, rel_tol=1e-5, abs_tol=1e-8):
        raise RuntimeError(f'Physics/control clock mismatch: expected {torque_dt:.6f}s '
                           f'per torque update, observed {elapsed:.6f}s')
    return native_dt


def priority_error_diagnostics(errors):
    """Keep history fields stable when a priority stage turns on or off."""
    return dict(priority_active=np.asarray(errors is not None),
                priority_stage_constraint_error=(np.full(1, np.nan, dtype=np.float32)
                    if errors is None else _to_numpy(errors[0]).copy()),
                priority_lock_error=(np.full(1, np.nan, dtype=np.float32)
                    if errors is None else _to_numpy(errors[1]).copy()))


def blend_secondary_task_metrics(requests, priority_names, priority_rows, blend):
    """Continuously add secondary row costs to a single physical QP objective."""
    if not math.isfinite(blend) or not 0 <= blend <= 1:
        raise ValueError('Secondary task blend must be in [0,1]')
    for request in requests:
        factors = torch.full_like(request.desired, math.sqrt(blend))
        if request.name in priority_names:
            selected = priority_rows.get(request.name)
            if selected is None:
                factors[:] = 1.
            else:
                factors[:, list(selected)] = 1.
        request.metric = request.metric*factors[:, :, None]*factors[:, None, :]


def root_position_request(q, velocity, reference_q, reference_v, reference_a,
                          metric, stiffness=100., damping=20., feedback_limit=30.,axes='xyz'):
    """Track the moving root origin in world coordinates through the physical QP.

    This is a soft acceleration objective. It cannot apply forces to the
    unactuated base; the existing dynamics and measured contacts still govern
    what joint torque can realize. Feedforward is not feedback-clipped.
    """
    from fabrics_sim.fabrics.task_request import TaskRequest
    if (any(not math.isfinite(x) or x <= 0 for x in
            (metric, stiffness, damping, feedback_limit)) or axes not in ('y','xy','xyz')):
        raise ValueError('Root task weights and gains must be finite and positive')
    feedback = (stiffness*(reference_q[:, :3]-q[:, :3])
                + damping*(reference_v[:, :3]-velocity[:, :3]))
    desired = reference_a[:, :3]+feedback.clamp(-feedback_limit, feedback_limit)
    A = q.new_zeros(q.shape[0], 3, 35)
    A[:, :, :3] = torch.eye(3, device=q.device, dtype=q.dtype)
    active=torch.ones_like(desired,dtype=torch.bool)
    if axes!='xyz':active[:,2]=False
    if axes=='y':active[:,0]=False
    return TaskRequest('root_position', A, torch.zeros_like(desired), desired,
                       metric*torch.eye(3, device=q.device, dtype=q.dtype)[None].expand(q.shape[0], -1, -1),
                       active)


def blend_root_position_weight(request, weight, axes, blend):
    """Convexly blend the existing root task's axis costs; targets stay fixed."""
    if (not math.isfinite(weight) or weight <= 0 or axes not in ('y','xy','xyz')
            or not math.isfinite(blend) or not 0 <= blend <= 1):
        raise ValueError('Invalid late root-task weights')
    if blend == 0:
        return
    mask=request.active.to(request.metric.dtype)
    old=request.metric*mask[..., :, None]*mask[..., None, :]
    target=request.metric.new_zeros(request.metric.shape)
    for axis in axes:
        index='xyz'.index(axis);target[:,index,index]=weight
    request.metric=(1-blend)*old+blend*target
    request.active=request.metric.diagonal(dim1=-2,dim2=-1)>0


def scale_stance_posture_metric(metric, active_contacts, scale):
    """Keep stance posture in the same QP stage, with a smaller soft weight.

    A congruence scales cross terms consistently for non-diagonal metrics.
    Swing legs and upper-body weights are retained; no task target is changed.
    """
    if not math.isfinite(scale) or not 0. < scale <= 1.:
        raise ValueError('Stance posture scale must be finite and in (0, 1]')
    if (metric.ndim != 3 or metric.shape[1:] != (29, 29)
            or active_contacts.shape != (len(metric), 2)
            or active_contacts.dtype != torch.bool
            or active_contacts.device != metric.device):
        raise ValueError('Expected joint metric [B,29,29] and boolean contacts [B,2]')
    factors = metric.new_ones((len(metric), 29))
    stance_rows = active_contacts.repeat_interleave(6, dim=1).to(metric.dtype)
    factors[:, :12] = 1. + (math.sqrt(scale)-1.)*stance_rows
    return factors[:, :, None] * metric * factors[:, None, :]


def control_timesteps(control_frequency, physics_frequency=None, *, g1_md=False):
    """Choose commensurate simulation periods before constructing Isaac."""
    rate = physics_frequency if physics_frequency is not None else (2000 if g1_md else 500)
    if (control_frequency <= 0 or rate < control_frequency
            or rate % control_frequency != 0):
        raise ValueError('Physics frequency must be an integer multiple of control frequency')
    return 1. / control_frequency, 1. / rate


def solve_single_tracking_qp(solver, **kwargs):
    """Track the supplied projected acceleration with no task-level objective."""
    if solver.solver_backend != 'daqp':
        raise ValueError('Single-solve tracking requires the DAQP backend')
    kwargs.update(task_requests=None, priority_tasks=(), priority_requests=None,
                  priority_levels=None, priority_rows=None, priority_projections=None,
                  command_acc_correction=None)
    return solver.solve(**kwargs)


def configure_full_tracking_profile(parser):
    """Run the AMP reference with a single acceleration-tracking QP per tick.

    Full source coverage, original-source fidelity and measured landing checks
    remain separate acceptance conditions. The preset is not a success claim.
    """
    parser.add_argument('--tracking-profile', choices=('amp-full',), default=None,
                        help='Attempt all 2000 AMP source frames and all 29 landings')
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument('--tracking-profile')
    if probe.parse_known_args()[0].tracking_profile is None:
        parser.set_defaults(
            anticipate_unload_start=float('inf'),
            body_feedback_start=float('inf'),
            centroidal_preview_start=float('inf'),
            clock_final_speed=1.0,
            clock_second_start=float('inf'),
            clock_third_start=float('inf'),
            clock_transition_start=float('inf'),
            collision_barrier_bandwidth_start=float('inf'),
            com_feedback_metric_start=float('inf'),
            com_feedback_start=float('inf'),
            configuration_tracking_start=float('inf'),
            contact_friction_refine_start=float('inf'),
            contact_pause_start=float('inf'),
            convex_wrist_collision_start=float('inf'),
            flight_feedback_start=float('inf'),
            foot_feedback_starts=[float('inf'), float('inf')],
            full_body_preview_start=float('inf'),
            full_body_preview_warm_start=float('inf'),
            full_swing_pose_start=float('inf'),
            global_posture_feedback_start=float('inf'),
            horizontal_feedback_start=float('inf'),
            joint_limit_priority_start=float('inf'),
            landing_gated_unload_start=float('inf'),
            late_balance_hierarchy_start=float('inf'),
            late_body_angular_metric_start=float('inf'),
            late_contact_stabilization_start=float('inf'),
            late_foot_angular_metric_start=float('inf'),
            late_foot_metric_start=float('inf'),
            late_joint_tube_start=float('inf'),
            late_leg_feedback_start=float('inf'),
            late_momentum_priority_start=float('inf'),
            late_pelvis_priority_start=float('inf'),
            late_reference_smooth_start=float('inf'),
            late_tracking_tube_start=float('inf'),
            late_tube_smooth_start=float('inf'),
            late_unload_start=float('inf'),
            leg_posture_secondary_start=float('inf'),
            measured_stance_reference_start=float('inf'),
            minimum_support_acceleration_start=float('inf'),
            model_effort_start=float('inf'),
            native_contact_plan_start=float('inf'),
            pelvis_orientation_replace_start=float('inf'),
            physical_braking_start=float('inf'),
            position_tube_start=float('inf'),
            posture_total_accel_start=float('inf'),
            retain_planned_support_start=float('inf'),
            smooth_start=float('inf'),
            stance_posture_secondary_start=float('inf'),
            support_load_floor_start=float('inf'),
            support_orientation_priority_start=float('inf'),
            swing_balance_priority_start=float('inf'),
            synchronous_effort_budget_start=float('inf'),
            tracking_margin_start=float('inf'),
            tracking_tube_start=float('inf'),
            transition_swing_priority_start=float('inf'),
            upper_posture_metric_start=float('inf'),
            vertical_feedback_start=float('inf'),
            waist_feedback_start=float('inf'),
        )
        return parser.parse_args()
    preset = ['--control-frequency', '500', '--motion-start-in-motion', '--tracking-steps', '0',
     '--track-full-motion', '--motion-pkl',
     'acc_data/amp_walk_20261102/amp_walk_native_contacts.pkl', '--motion-interpolation',
     'pchip-analytic', '--reference-mode', 'unified', '--motion-feedback-scale', '9',
     '--preserve-reference-acceleration', '--balance-priority', '--swing-position-priority',
     '--whole-body-posture-priority', '--dynamic-unload-weight', '.05',
     '--swing-midpoint-clearance', '.056', '--observed-contact-patches', '--source-pelvis-weight',
     '1', '--device', 'cpu', '--duration', '18', '--joint-velocity-limits',
     '--joint-limit-avoidance-weight', '20', '--joint-limit-barrier-bandwidth', '8', '--qp-solver',
     'daqp', '--wrist-hip-avoidance', '--wrist-hip-barrier', '--wrist-body-avoidance',
     '--report-self-contacts', '--diagnose-com', '--diagnose-contacts', '--motor-armature',
     '--foot-feedback-starts', '1.3', '.88', '--foot-feedback-limits', '150', '150',
     '--foot-feedback-joints', 'leg', 'leg', '--flight-feedback-ramp', '.04',
     '--waist-feedback-start', '100', '--late-leg-feedback-start', '100',
     '--global-posture-feedback-start', '.70', '--global-posture-feedback-limit', '150',
     '--smooth-start', '.65', '--smooth-ramp', '.1', '--smooth-stride', '6',
     '--tube-smooth-stride', '2', '--reference-correction-rmse-limit', '.08',
     '--reference-pose-candidate',
     'acc_data/amp_walk_20261103/feedback_followup/wbc_convex_catch15_25_clear35.npz',
     '--landing-gated-unload-start', '1.48', '--gate-contact-transitions',
     '--horizontal-feedback-start', '1.4', '--horizontal-feedback-limit', '80',
     '--horizontal-feedback-metric-scale', '4', '--vertical-feedback-start', '1.4',
     '--full-swing-pose-start', '1.4',
     '--clock-transition-start', '.90', '--clock-transition-ramp', '.35', '--clock-final-speed',
     '1.2', '--stop-at-fidelity-loss', '--anticipate-unload-hysteresis',
     '--anticipate-unload-start', '3.7', '--anticipate-unload-horizon', '.1',
     '--late-unload-start', '3.7', '--late-unload-ramp', '.02', '--late-unload-weight', '.2',
     '--late-body-angular-metric-start', '4.2', '--late-body-angular-metric-scale', '.1',
     '--anticipate-unload-overdue', '--convex-wrist-collision', '--convex-wrist-collision-start',
     '3.5', '--model-effort-start', '4.15', '--tracking-margin-start', '4.4',
     '--late-foot-metric-start', '4.5', '--late-foot-metric-scale', '4',
     '--full-body-preview-start', 'inf', '--centroidal-preview-start', 'inf', '--direct-wbc',
     '--tracking-servo', '--contact-aware-servo', '--nominal-implicit-servo']
    # Verified single-QP continuation: three measured source landings.
    # Preserve the physical constraints and all auxiliary tracking barriers.
    preset += ['--posture-metric-scale', '10', '--joint-feedback-limit', '150',
               '--projection-joint-correction-weight', '1',
               '--qp-base-acceleration-weight', '100', '--source-pelvis-weight', '10',
               '--fabric-angular-feedback-limit', '100',
               '--output-path', 'acc_data/g1_projected_acc_qp_20261114/continued_tracking.npz']
    args = parser.parse_args(preset + sys.argv[1:])
    if (args.tracking_steps or not args.track_full_motion
            or not args.stop_at_fidelity_loss or not args.direct_wbc or args.g1_md):
        parser.error('amp-full requires direct WBC, the whole source and original fidelity checks')
    if args.full_body_preview_start != float('inf') or args.centroidal_preview_start != float('inf'):
        parser.error('amp-full uses a single nonpredictive WBC QP')
    if (args.relax_joint_tube or args.recover_orientation_tube or args.shoulder_roll_guard
            or args.qp_solver != 'daqp'):
        parser.error('amp-full requires one DAQP attempt: tube retries and shoulder guard are disabled')
    return args


def offset_left_shoulder_reference(trajectory: torch.Tensor, offset: float) -> torch.Tensor:
    """Return a reference-only clearance variant; never mutate source poses."""
    if trajectory.ndim != 2 or trajectory.shape[1] != 36:
        raise ValueError("Expected motion poses [frames,36].")
    if not math.isfinite(offset):
        raise ValueError("Shoulder reference offset must be finite.")
    result = trajectory.clone()
    result[:, 7 + G1_JOINT_NAMES.index("left_shoulder_roll_joint")] += offset
    return result


def configure_motion_targets(
    fabric: G1PoseFabric,
    q: torch.Tensor,
    previous_q: torch.Tensor | None = None,
    next_q: torch.Tensor | None = None,
    reference_dt: float | None = None,
    *,
    reference_momentum: bool = True,
    reference_kinematics: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> dict | None:
    """Set task references using ``a_ref - Kp*e - Kd*e_dot``.

    The three consecutive PKL poses make trajectory time explicit in each
    Fabric task.  At a trajectory endpoint the caller passes ``None`` on the
    missing side and the reference acceleration is conservatively zero.
    A manager supplying its own momentum target can skip the unused source
    momentum evaluation; all pose and posture references are still updated.
    """
    if reference_kinematics is not None:
        velocity, acceleration = reference_kinematics
        if (fabric.dynamics_tasks is None or velocity.shape != (q.shape[0],35)
                or acceleration.shape != velocity.shape
                or any(x.dtype != q.dtype or x.device != q.device or not torch.isfinite(x).all()
                       for x in (velocity,acceleration))):
            raise ValueError('Analytic reference derivatives require matching physical-model states')
        model = fabric.dynamics_tasks.model
        state, _, _ = model.evaluate(q,velocity,fabric.integrate_root_position,
                                     fabric.dynamics_tasks.config['difference_step'])
        fabric.set_joint_posture_target(q[:,7:],velocity[:,6:],acceleration[:,6:])
        fabric.set_pelvis_target(q[:,:3].clone(),q[:,3:7].clone(),
            velocity[:,:3],velocity[:,3:6],acceleration[:,:3],acceleration[:,3:6])
        for taskmap, setter in ((fabric.feet_pose_taskmap,fabric.set_feet_targets),
                                 (fabric.hand_pose_taskmap,fabric.set_hand_targets)):
            indices = [model.kinematics.get_link_index(taskmap.frame_groups[name][0])
                       for name in taskmap.end_effector_order]
            pose = state['transforms'][:,indices]
            Jv,Jw = state['origin_jac'][:,indices],state['angular_jac'][:,indices]
            linear_v = (Jv@velocity[:,None,:,None]).squeeze(-1)
            angular_v = (Jw@velocity[:,None,:,None]).squeeze(-1)
            linear_a = (Jv@acceleration[:,None,:,None]).squeeze(-1)+state['origin_jac_bias'][:,indices]
            angular_a = (Jw@acceleration[:,None,:,None]).squeeze(-1)+state['angular_jac_bias'][:,indices]
            setter(pose[:,:,:3],pose[:,:,3:7],linear_v,angular_v,linear_a,angular_a)
        if reference_momentum:
            fabric.set_centroidal_momentum_target(state['momentum'],
                (state['momentum_matrix']@acceleration[...,None]).squeeze(-1)+state['momentum_matrix_bias'])
        return state
    fabric.feet_pose_taskmap(q, None)
    feet = fabric.feet_pose_taskmap.robot_kinematics.batch_link_transforms_torch[
        :, fabric.feet_pose_taskmap.link_indices
    ].clone()
    fabric.hand_pose_taskmap(q, None)
    hands = fabric.hand_pose_taskmap.robot_kinematics.batch_link_transforms_torch[
        :, fabric.hand_pose_taskmap.link_indices
    ].clone()

    pelvis_linear_velocity = None
    pelvis_angular_velocity = None
    pelvis_linear_acceleration = None
    pelvis_angular_acceleration = None
    feet_linear_velocity = None
    feet_angular_velocity = None
    feet_linear_acceleration = None
    feet_angular_acceleration = None
    hands_linear_velocity = None
    hands_angular_velocity = None
    hands_linear_acceleration = None
    hands_angular_acceleration = None
    joint_reference_velocity = None
    joint_reference_acceleration = None
    if previous_q is not None or next_q is not None:
        if reference_dt is None or reference_dt <= 0.0:
            raise ValueError("reference_dt must be positive for a motion reference.")
        previous_q = q if previous_q is None else previous_q
        next_q = q if next_q is None else next_q
        dt = reference_dt
        joint_reference_velocity, joint_reference_acceleration = finite_difference_reference(
            previous_q[:, 7:], q[:, 7:], next_q[:, 7:], dt,
            previous_q is q, next_q is q,
        )
        fabric.feet_pose_taskmap(previous_q, None)
        previous_feet = (
            fabric.feet_pose_taskmap.robot_kinematics
            .batch_link_transforms_torch[:, fabric.feet_pose_taskmap.link_indices]
            .clone()
        )
        fabric.hand_pose_taskmap(previous_q, None)
        previous_hands = (
            fabric.hand_pose_taskmap.robot_kinematics
            .batch_link_transforms_torch[:, fabric.hand_pose_taskmap.link_indices]
            .clone()
        )
        fabric.feet_pose_taskmap(next_q, None)
        next_feet = (
            fabric.feet_pose_taskmap.robot_kinematics
            .batch_link_transforms_torch[
                :, fabric.feet_pose_taskmap.link_indices
            ]
            .clone()
        )
        fabric.hand_pose_taskmap(next_q, None)
        next_hands = (
            fabric.hand_pose_taskmap.robot_kinematics
            .batch_link_transforms_torch[
                :, fabric.hand_pose_taskmap.link_indices
            ]
            .clone()
        )
        pelvis_linear_velocity, pelvis_linear_acceleration = finite_difference_reference(
            previous_q[:, :3], q[:, :3], next_q[:, :3], dt,
            previous_q is q, next_q is q,
        )
        pelvis_angular_velocity, pelvis_angular_acceleration = angular_reference(
            previous_q[:, 3:7], q[:, 3:7], next_q[:, 3:7], dt,
            previous_q is q, next_q is q,
        )
        feet_linear_velocity, feet_linear_acceleration = finite_difference_reference(
            previous_feet[..., :3], feet[..., :3], next_feet[..., :3], dt,
            previous_q is q, next_q is q,
        )
        feet_angular_velocity, feet_angular_acceleration = angular_reference(
            previous_feet[..., 3:7], feet[..., 3:7], next_feet[..., 3:7], dt,
            previous_q is q, next_q is q,
        )
        hands_linear_velocity, hands_linear_acceleration = finite_difference_reference(
            previous_hands[..., :3], hands[..., :3], next_hands[..., :3], dt,
            previous_q is q, next_q is q,
        )
        hands_angular_velocity, hands_angular_acceleration = angular_reference(
            previous_hands[..., 3:7], hands[..., 3:7], next_hands[..., 3:7], dt,
            previous_q is q, next_q is q,
        )
    fabric.set_joint_posture_target(q[:, 7:], joint_reference_velocity)
    fabric.set_pelvis_target(
        pelvis_position_target=q[:, :3].clone(),
        pelvis_quaternion_target_xyzw=q[:, 3:7].clone(),
        pelvis_linear_velocity_target=pelvis_linear_velocity,
        pelvis_angular_velocity_target=pelvis_angular_velocity,
        pelvis_linear_acceleration_target=pelvis_linear_acceleration,
        pelvis_angular_acceleration_target=pelvis_angular_acceleration,
    )
    fabric.set_feet_targets(
        position_target=feet[..., :3],
        quaternion_target_xyzw=feet[..., 3:7],
        linear_velocity_target=feet_linear_velocity,
        angular_velocity_target=feet_angular_velocity,
        linear_acceleration_target=feet_linear_acceleration,
        angular_acceleration_target=feet_angular_acceleration,
    )
    fabric.set_hand_targets(
        position_target=hands[..., :3],
        quaternion_target_xyzw=hands[..., 3:7],
        linear_velocity_target=hands_linear_velocity,
        angular_velocity_target=hands_angular_velocity,
        linear_acceleration_target=hands_linear_acceleration,
        angular_acceleration_target=hands_angular_acceleration,
    )
    if fabric.dynamics_tasks is not None:
        fabric.set_joint_posture_target(q[:, 7:], joint_reference_velocity, joint_reference_acceleration)
        if not reference_momentum:
            return
        reference_v = torch.zeros(q.shape[0], 35, device=q.device, dtype=q.dtype)
        reference_a = torch.zeros_like(reference_v)
        if pelvis_linear_velocity is not None:
            reference_v = torch.cat((pelvis_linear_velocity, pelvis_angular_velocity, joint_reference_velocity), -1)
            reference_a = torch.cat((pelvis_linear_acceleration, pelvis_angular_acceleration, joint_reference_acceleration), -1)
        reference_state, _, _ = fabric.dynamics_tasks.model.evaluate(
            q, reference_v, fabric.integrate_root_position, fabric.dynamics_tasks.config["difference_step"])
        fabric.set_centroidal_momentum_target(reference_state["momentum"],
            (reference_state["momentum_matrix"] @ reference_a[..., None]).squeeze(-1)
            + reference_state["momentum_matrix_bias"])


def align_pelvis_balance_reference(fabric, q, previous_q, next_q, dt,
                                   com_position, com_velocity, com_acceleration):
    """Give pelvis and COM the same horizontal balance displacement.

    Preserve PKL pelvis height and orientation. Differentiate the displacement
    as well as position, so the two tasks do not request conflicting velocities.
    """
    reference_com = fabric.compute_center_of_mass(q)
    previous = q if previous_q is None else previous_q
    following = q if next_q is None else next_q
    root_v, root_a = finite_difference_reference(previous[:, :3], q[:, :3], following[:, :3],
                                                dt, previous_q is None, next_q is None)
    com_v, com_a = finite_difference_reference(
        fabric.compute_center_of_mass(previous), reference_com,
        fabric.compute_center_of_mass(following), dt, previous_q is None, next_q is None)
    angular_v, angular_a = angular_reference(previous[:, 3:7], q[:, 3:7], following[:, 3:7],
                                             dt, previous_q is None, next_q is None)
    position = q[:, :3].clone()
    position[:, :2] += (com_position - reference_com)[:, :2]
    root_v[:, :2] += (com_velocity - com_v)[:, :2]
    root_a[:, :2] += (com_acceleration - com_a)[:, :2]
    fabric.set_pelvis_target(position, q[:, 3:7], root_v, angular_v, root_a, angular_a)


def quaternion_angular_velocity(
    quaternion: torch.Tensor,
    next_quaternion: torch.Tensor,
    dt: float,
) -> torch.Tensor:
    """Return the world-frame angular velocity between two xyzw poses."""
    rotation = _quat_xyzw_to_matrix(quaternion)
    next_rotation = _quat_xyzw_to_matrix(next_quaternion)
    return _so3_log(torch.matmul(next_rotation, rotation.transpose(-1, -2))) / dt


def finite_difference_reference(
    previous: torch.Tensor,
    current: torch.Tensor,
    next_value: torch.Tensor,
    dt: float,
    at_start: bool,
    at_end: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return position-reference velocity and acceleration for uniform samples."""
    if at_start:
        velocity = (next_value - current) / dt
    elif at_end:
        velocity = (current - previous) / dt
    else:
        velocity = (next_value - previous) / (2.0 * dt)
    acceleration = torch.zeros_like(current) if (at_start or at_end) else (
        (next_value - 2.0 * current + previous) / (dt * dt)
    )
    return velocity, acceleration


def angular_reference(
    previous: torch.Tensor,
    current: torch.Tensor,
    next_value: torch.Tensor,
    dt: float,
    at_start: bool,
    at_end: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return world-frame angular velocity and finite-difference acceleration."""
    omega_before = quaternion_angular_velocity(previous, current, dt)
    omega_after = quaternion_angular_velocity(current, next_value, dt)
    velocity = omega_after if at_start else omega_before if at_end else 0.5 * (omega_before + omega_after)
    acceleration = torch.zeros_like(velocity) if (at_start or at_end) else (omega_after - omega_before) / dt
    return velocity, acceleration


def ramped_motion_time(elapsed, ramp_duration):
    """Integrate a smoothstep playback speed from zero to one."""
    if ramp_duration < 0 or not math.isfinite(ramp_duration):
        raise ValueError("Motion ramp duration must be finite and nonnegative")
    elapsed = max(0., elapsed)
    if ramp_duration == 0:
        return elapsed
    if elapsed >= ramp_duration:
        return elapsed - .5 * ramp_duration
    phase = elapsed / ramp_duration
    return ramp_duration * (phase**3 - .5*phase**4)


def interpolate_motion_trajectory(
    trajectory: torch.Tensor,
    frame_position: float,
) -> torch.Tensor:
    """Sample a trajectory at a fractional frame using lerp and quaternion slerp."""
    last_frame = trajectory.shape[0] - 1
    position = min(max(float(frame_position), 0.0), float(last_frame))
    lower = int(math.floor(position))
    upper = min(lower + 1, last_frame)
    alpha = position - lower
    if upper == lower or alpha == 0.0:
        return trajectory[lower:lower + 1].clone()

    first = trajectory[lower:lower + 1]
    second = trajectory[upper:upper + 1]
    result = (1.0 - alpha) * first + alpha * second
    q0 = first[:, 3:7]
    q1 = second[:, 3:7]
    dot = torch.sum(q0 * q1, dim=-1, keepdim=True)
    q1 = torch.where(dot < 0.0, -q1, q1)
    dot = torch.sum(q0 * q1, dim=-1, keepdim=True).clamp(-1.0, 1.0)
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)
    slerp = (
        torch.sin((1.0 - alpha) * theta) / sin_theta.clamp_min(1.0e-7) * q0
        + torch.sin(alpha * theta) / sin_theta.clamp_min(1.0e-7) * q1
    )
    lerp = (1.0 - alpha) * q0 + alpha * q1
    quaternion = torch.where(sin_theta.abs() < 1.0e-5, lerp, slerp)
    result[:, 3:7] = quaternion / torch.linalg.vector_norm(
        quaternion, dim=-1, keepdim=True
    ).clamp_min(1.0e-8)
    return result


def reference_feet_transforms(
    fabric: G1PoseFabric,
    reference_q: torch.Tensor,
) -> torch.Tensor:
    """Return left/right reference foot transforms for one PKL frame."""
    fabric.feet_pose_taskmap(reference_q, None)
    return (
        fabric.feet_pose_taskmap.robot_kinematics
        .batch_link_transforms_torch[
            :, fabric.feet_pose_taskmap.link_indices
        ]
        .clone()
    )


@torch.no_grad()
def ground_motion_reference(fabric, trajectory, start_frame):
    """Place the lowest modeled sole corner on the flat floor, once at spawn.

    Translate the whole reference by the same amount; never reset live states.
    Uses the same sole rectangle and depth as the contact-wrench model.
    """
    feet = reference_feet_transforms(fabric, trajectory[start_frame:start_frame+1])
    corners = trajectory.new_tensor([[-.05,-.025,-.035],[-.05,.025,-.035],
                                     [.12,-.025,-.035],[.12,.025,-.035]])
    rotation = _quat_xyzw_to_matrix(feet[...,3:7])
    world = feet[...,:3].unsqueeze(-2) + (rotation.unsqueeze(-3) @ corners[...,None]).squeeze(-1)
    shift = -world[...,2].amin()
    result = trajectory.clone()
    result[:,2] += shift
    return result, float(shift)


def source_start_contacts(feet: torch.Tensor, tolerance: float = .002) -> torch.Tensor:
    """Seed only touching soles before the first delayed PhysX force sample.

    This is an initialization assumption, never a replacement for measured
    contact updates. The normal detector may subsequently remove support.
    """
    corners = feet.new_tensor([[-.05, -.025, -.035], [-.05, .025, -.035],
                              [.12, -.025, -.035], [.12, .025, -.035]])
    rotation = _quat_xyzw_to_matrix(feet[..., 3:7])
    points = feet[..., :3].unsqueeze(-2) + (rotation.unsqueeze(-3) @ corners[..., None]).squeeze(-1)
    height = points[..., 2].amin(-1)
    return height.abs() <= tolerance


def contact_pose_error(
    current_feet: torch.Tensor,
    contact_anchors: torch.Tensor,
) -> torch.Tensor:
    """Return per-foot world-frame SE(3) error from fixed contact anchors."""
    position_error = current_feet[..., :3] - contact_anchors[..., :3]
    current_rotation = _quat_xyzw_to_matrix(current_feet[..., 3:7])
    anchor_rotation = _quat_xyzw_to_matrix(contact_anchors[..., 3:7])
    orientation_error = _so3_log(
        torch.matmul(current_rotation, anchor_rotation.transpose(-1, -2))
    )
    return torch.cat((position_error, orientation_error), dim=-1)


def planned_contact_mode_from_reference_frame(
    fabric: G1PoseFabric,
    reference_q: torch.Tensor,
    reference_ground_heights: torch.Tensor,
    height_tolerance: float,
) -> torch.Tensor:
    """Infer support order from the same PKL frame used by Fabric.

    A foot close to its initial reference ground height is planned as support.
    If numerical/root-height motion marks both feet as airborne, retain the
    lower-clearance foot as support to avoid an unintended flight phase.
    """
    reference_feet = reference_feet_transforms(fabric, reference_q)
    clearance = reference_feet[..., 2] - reference_ground_heights
    contacts = clearance <= height_tolerance
    neither = ~contacts.any(dim=1)
    lower_foot = clearance.argmin(dim=1)
    fallback = torch.nn.functional.one_hot(
        lower_foot, num_classes=2
    ).to(dtype=torch.bool)
    contacts = torch.where(neither.unsqueeze(-1), fallback, contacts)
    return (
        contacts[:, 0].to(torch.int64)
        + 2 * contacts[:, 1].to(torch.int64)
    )


def contact_mode_from_sequence_frame(
    contact_sequence: torch.Tensor,
    frame: int,
) -> torch.Tensor:
    """Encode a precomputed [left, right] contact mask as mode 0..3."""
    contacts = contact_sequence[frame:frame + 1]
    return (
        contacts[:, 0].to(torch.int64)
        + 2 * contacts[:, 1].to(torch.int64)
    )


def planned_contact_mode_for_frame(
    fabric: G1PoseFabric,
    reference_q: torch.Tensor,
    reference_ground_heights: torch.Tensor,
    height_tolerance: float,
    contact_sequence: torch.Tensor | None,
    frame: int,
    allow_flight: bool,
) -> torch.Tensor:
    """Combine precomputed phase labels with reference-foot plausibility.

    Existing motion files can contain long false flight phases because their
    labels were inferred from toe vertical velocity alone.  Unless flight was
    explicitly enabled, use reference-foot height to replace flight labels and
    contacts assigned to a foot that is visibly above the floor.
    """
    height_mode = planned_contact_mode_from_reference_frame(
        fabric,
        reference_q,
        reference_ground_heights,
        height_tolerance,
    )
    if contact_sequence is None:
        return height_mode

    sequence_mode = contact_mode_from_sequence_frame(contact_sequence, frame)
    if allow_flight:
        return sequence_mode

    left_height_support = (height_mode == 1) | (height_mode == 3)
    right_height_support = (height_mode == 2) | (height_mode == 3)
    sequence_support_is_plausible = (
        ((sequence_mode == 1) & left_height_support)
        | ((sequence_mode == 2) & right_height_support)
        | ((sequence_mode == 3) & (height_mode == 3))
    )
    return torch.where(
        sequence_support_is_plausible,
        sequence_mode,
        height_mode,
    )


def motion_generalized_velocity(
    trajectory: torch.Tensor,
    frame: int,
    fps: float,
) -> torch.Tensor:
    """Finite-difference a motion frame into the 35-D velocity convention."""
    if fps <= 0.0:
        raise ValueError("fps must be positive")
    previous_frame = max(frame - 1, 0)
    next_frame = min(frame + 1, trajectory.shape[0] - 1)
    previous_q = trajectory[previous_frame:previous_frame + 1]
    q = trajectory[frame:frame + 1]
    next_q = trajectory[next_frame:next_frame + 1]
    at_start = previous_frame == frame
    at_end = next_frame == frame
    linear_velocity, _ = finite_difference_reference(
        previous_q[:, :3],
        q[:, :3],
        next_q[:, :3],
        1.0 / fps,
        at_start,
        at_end,
    )
    angular_velocity, _ = angular_reference(
        previous_q[:, 3:7],
        q[:, 3:7],
        next_q[:, 3:7],
        1.0 / fps,
        at_start,
        at_end,
    )
    joint_velocity, _ = finite_difference_reference(
        previous_q[:, 7:36],
        q[:, 7:36],
        next_q[:, 7:36],
        1.0 / fps,
        at_start,
        at_end,
    )
    return torch.cat(
        (linear_velocity, angular_velocity, joint_velocity), dim=-1
    )


def single_support_ready(com, velocity, target, foot, foot_speed, forces, support):
    """Conservative quasi-static diagnostic gate, not a dynamic walking certificate."""
    height = (com[:, 2] - foot[:, 2]).clamp_min(.1)
    capture = com[:, :2] + velocity[:, :2] / torch.sqrt(9.81 / height)[:, None]
    load = forces[:, support] / forces.sum(dim=1).clamp_min(1.)
    return ((load >= .9) & (forces[:, support] >= 20.)
            & (torch.linalg.vector_norm(com - target, dim=1) <= .03)
            & (torch.linalg.vector_norm(velocity, dim=1) <= .05)
            & (foot_speed <= .02)
            & ((capture - foot[:, :2]).abs() <= com.new_tensor([.06, .04])).all(dim=1))


def support_aware_com_target(
    fabric: G1PoseFabric,
    reference_q: torch.Tensor,
    support_reference_q: torch.Tensor,
    support_mode: torch.Tensor,
    support_anchors: torch.Tensor | None = None,
    anchored_contacts: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build a COM target that shifts toward the foot used for support.

    ``support_reference_q`` may be a future frame. This starts the weight
    transfer before the reference foot actually leaves the ground.
    """
    reference_com = fabric.compute_center_of_mass(reference_q).detach()
    support_feet = reference_feet_transforms(
        fabric, support_reference_q
    )[..., :3]
    if support_anchors is not None:
        if anchored_contacts is None or support_anchors.shape != support_feet.shape or anchored_contacts.shape != support_feet.shape[:2]:
            raise ValueError("Support anchors/mask must have shape [B,2,3]/[B,2].")
        support_feet = torch.where(anchored_contacts[:, :, None], support_anchors, support_feet)
    left_support = support_mode == 1
    right_support = support_mode == 2
    support_xy = 0.5 * (support_feet[:, 0, :2] + support_feet[:, 1, :2])
    support_xy = torch.where(
        left_support[:, None], support_feet[:, 0, :2], support_xy
    )
    support_xy = torch.where(
        right_support[:, None], support_feet[:, 1, :2], support_xy
    )

    # Retain part of the retargeted motion's natural COM progression. During
    # single support, bias strongly toward the support ankle so the other foot
    # can unload. During double support, track the reference COM directly.
    single_support = (left_support | right_support)[:, None]
    target_xy = torch.where(
        single_support,
        0.75 * support_xy + 0.25 * reference_com[:, :2],
        reference_com[:, :2],
    )
    return torch.cat((target_xy, reference_com[:, 2:3]), dim=-1)


def select_balance_support(
    commanded_mode: torch.Tensor,
    measured_mode: torch.Tensor,
    reference_q: torch.Tensor,
    lookahead_reference_q: torch.Tensor,
    lookahead_mode: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select the support reference used by the COM controller.

    Lookahead is useful only while both feet are constrained: it shifts the
    COM toward the foot that will remain after lift-off.  Once the controller
    has entered single support, looking through to a later opposite-foot phase
    would pull the COM toward the current swing foot and destabilize tracking.
    An unexpected measured single-support state still takes precedence over
    the lookahead plan so balance follows the physical support.
    """
    commanded_single = (commanded_mode == 1) | (commanded_mode == 2)
    measured_single = (measured_mode == 1) | (measured_mode == 2)
    use_current_reference = commanded_single | measured_single
    current_mode = torch.where(
        commanded_single,
        commanded_mode,
        measured_mode,
    )
    balance_mode = torch.where(
        use_current_reference,
        current_mode,
        lookahead_mode,
    )
    balance_reference_q = torch.where(
        use_current_reference[:, None],
        reference_q,
        lookahead_reference_q,
    )
    return balance_mode, balance_reference_q


def advance_contact_phase(frame: float, stride: float, modes: torch.Tensor,
                          commanded_mode: int) -> tuple[float, int, bool]:
    """Advance without crossing an unprepared single-support phase.

    Return the held reference frame and pending mode separately, so the contact
    state machine can prepare support even while task poses stop advancing.
    Inspect intermediate frames to avoid skipping a short support phase.
    """
    if modes.ndim != 1 or modes.numel() == 0 or not math.isfinite(frame) or not math.isfinite(stride) or stride < 0:
        raise ValueError("Expected nonempty 1-D modes and finite frame/nonnegative stride.")
    if not 0 <= frame <= modes.numel() - 1:
        raise ValueError("Motion frame is outside the trajectory.")
    candidate = min(frame + stride, modes.numel() - 1)
    for index in range(int(round(frame)), int(round(candidate)) + 1):
        requested = int(modes[index])
        if requested in (1, 2) and requested != commanded_mode:
            return max(frame, float(index - 1)), requested, True
    return candidate, int(modes[int(round(candidate))]), False


def update_contact_mode_state_machine(
    commanded_mode: torch.Tensor,
    planned_mode: torch.Tensor,
    measured_contacts: torch.Tensor,
    ready_counter: torch.Tensor,
    confirm_updates: int,
    foot_normal_forces: torch.Tensor | None = None,
    minimum_support_load_fraction: float = 0.8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Safely reconcile planned phases with measured foot contacts.

    Ground-to-flight follows a debounced explicit flight request.  Flight-to-
    ground and single-to-double transitions are gated by actual contact, while
    double-to-single waits until the requested support foot is loaded. When
    forces are supplied, it must carry the specified fraction of total load;
    merely touching the ground is not sufficient to release the other foot.
    """
    commanded_mode = commanded_mode.clone()
    ready_counter = ready_counter.clone()
    measured_mode = (
        measured_contacts[:, 0].to(torch.int64)
        + 2 * measured_contacts[:, 1].to(torch.int64)
    )

    # Constrain a measured touchdown immediately. This also recovers safely
    # from an unplanned collision during a nominal flight phase.
    in_flight = commanded_mode == 0
    landed = in_flight & (measured_mode != 0)
    commanded_mode = torch.where(landed, measured_mode, commanded_mode)
    ready_counter = torch.where(
        landed, torch.zeros_like(ready_counter), ready_counter
    )

    # Never keep a kinematic constraint on a foot whose debounced physical
    # contact has disappeared.  Planned modes decide when a loaded foot may
    # be released; measured modes decide whether a claimed support actually
    # exists.  This prevents the QP from balancing on a virtual contact.
    commanded_left = (commanded_mode == 1) | (commanded_mode == 3)
    commanded_right = (commanded_mode == 2) | (commanded_mode == 3)
    lost_commanded_contact = (
        (commanded_left & ~measured_contacts[:, 0])
        | (commanded_right & ~measured_contacts[:, 1])
    )
    planned_single = (planned_mode == 1) | (planned_mode == 2)
    planned_release_is_ready = (
        (commanded_mode == 3)
        & planned_single
        & (measured_mode == planned_mode)
    )
    physical_override = (
        lost_commanded_contact
        & (planned_mode != 0)
        & ~planned_release_is_ready
    )
    commanded_mode = torch.where(
        physical_override,
        measured_mode,
        commanded_mode,
    )
    ready_counter = torch.where(
        physical_override,
        torch.zeros_like(ready_counter),
        ready_counter,
    )

    double_support = commanded_mode == 3
    requests_single = (planned_mode == 1) | (planned_mode == 2)
    planned_support_index = torch.clamp(planned_mode - 1, min=0, max=1)
    support_is_measured = torch.gather(
        measured_contacts, 1, planned_support_index[:, None]
    ).squeeze(1)
    support_load_ready = support_is_measured
    if foot_normal_forces is not None:
        if foot_normal_forces.shape != measured_contacts.shape or not torch.isfinite(foot_normal_forces).all():
            raise ValueError("foot_normal_forces must be finite [B,2].")
        if not 0.5 < minimum_support_load_fraction <= 1.0:
            raise ValueError("minimum_support_load_fraction must be in (0.5,1].")
        upward_load = foot_normal_forces.clamp_min(0.)
        total_load = upward_load.sum(dim=1)
        support_load = upward_load.gather(1, planned_support_index[:, None]).squeeze(1)
        support_load_ready = (support_is_measured & (total_load > 0.) &
                              (support_load >= minimum_support_load_fraction * total_load))
    requests_flight = (commanded_mode != 0) & (planned_mode == 0)
    preparing_single = (
        double_support
        & requests_single
        & support_load_ready
    )
    preparing = requests_flight | preparing_single
    ready_counter = torch.where(
        preparing,
        ready_counter + 1,
        torch.zeros_like(ready_counter),
    )
    transition_ready = preparing & (ready_counter >= confirm_updates)
    take_off = transition_ready & requests_flight
    lift_off = transition_ready & preparing_single
    commanded_mode = torch.where(
        take_off, torch.zeros_like(commanded_mode), commanded_mode
    )
    commanded_mode = torch.where(
        lift_off, planned_mode, commanded_mode
    )
    ready_counter = torch.where(
        take_off | lift_off,
        torch.zeros_like(ready_counter),
        ready_counter,
    )

    # A swing foot must physically land before it is constrained. Opposite-side
    # requests pass through double support, preventing an instantaneous swap.
    in_single = (commanded_mode == 1) | (commanded_mode == 2)
    swing_index = torch.where(commanded_mode == 1, 1, 0)
    swing_is_measured = torch.gather(
        measured_contacts, 1, swing_index[:, None]
    ).squeeze(1)
    landing_or_opposite = (
        in_single
        & (planned_mode != 0)
        & (planned_mode != commanded_mode)
        & swing_is_measured
    )
    if foot_normal_forces is not None:
        # A substantial unexpected touchdown is already a physical constraint,
        # even if the motion still requests single support. Do not wait for
        # the planned phase or the low-force contact debounce to catch up.
        swing_load = upward_load.gather(1, swing_index[:, None]).squeeze(1)
        unexpected_loaded_contact = in_single & (swing_load >= 20.) & (swing_load > .25 * total_load)
        landing_or_opposite |= unexpected_loaded_contact
    commanded_mode = torch.where(
        landing_or_opposite,
        torch.full_like(commanded_mode, 3),
        commanded_mode,
    )
    return commanded_mode, ready_counter


def _to_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _clone_contiguous(value):
    """Own a contiguous dynamics buffer for either Torch or NumPy backends."""
    if isinstance(value, torch.Tensor):
        return value.clone().contiguous()
    return np.ascontiguousarray(np.asarray(value).copy())


def inverse_dynamics_joint_feedforward(
    mass_matrix: torch.Tensor,
    bias_force: torch.Tensor,
    contact_jacobian: torch.Tensor,
    acceleration: torch.Tensor,
    contact_wrench: torch.Tensor,
) -> torch.Tensor:
    """Map QP acceleration and contact wrench to 29-D joint torque.

    From M*qdd + h = S^T*tau + Jc^T*lambda, the actuated rows give
    tau = Mj*qdd + hj - Jcj^T*lambda.
    """
    inertial_and_bias = torch.bmm(
        mass_matrix[:, 6:35, :], acceleration.unsqueeze(-1)
    ).squeeze(-1) + bias_force[:, 6:35]
    contact_torque = torch.bmm(
        contact_jacobian[:, :, 6:35].transpose(1, 2),
        contact_wrench.unsqueeze(-1),
    ).squeeze(-1)
    return inertial_and_bias - contact_torque


def isaac_frame_data(robot):
    """PhysX exposes COM twists; Fabric uses the root/ankle link origins."""
    view = getattr(robot, "root_physx_view", None)
    if view is None:
        view = robot._physics_view
    poses = torch.as_tensor(_to_numpy(view.get_link_transforms()).copy())
    local_coms = torch.as_tensor(_to_numpy(view.get_coms()).copy())[..., :3]
    offsets = (_quat_xyzw_to_matrix(poses[..., 3:7]) @ local_coms[..., None]).squeeze(-1)
    velocity = torch.as_tensor(_to_numpy(robot.get_velocities()).copy())
    transform, bias = origin_to_com_map(offsets[:, 0], velocity[:, 3:6])
    return transform, bias, offsets


def read_isaac_dynamics(robot):
    """Read full G1 dynamics in root-link-origin coordinates (world axes)."""
    # Isaac Lab assets expose ``root_physx_view``. Isaac Sim Core's
    # ``Articulation`` stores the same PhysX tensor view as ``_physics_view``.
    view = getattr(robot, "root_physx_view", None)
    if view is None:
        view = getattr(robot, "_physics_view", None)
    if view is None:
        raise RuntimeError(
            "Isaac articulation has no initialized PhysX view. "
            "Call world.reset() before reading dynamics."
        )

    mass_matrix = _clone_contiguous(
        view.get_generalized_mass_matrices()
    )
    # PhysX's generalized mass API excludes joint armature (verified with
    # paired free-flight acceleration measurements). The integrator includes
    # it, so add the measured reflected inertia once in joint coordinates.
    armature = _clone_contiguous(view.get_dof_armatures())
    joint_indices = list(range(6, 35))
    mass_matrix[:, joint_indices, joint_indices] += armature
    coriolis = _clone_contiguous(
        view.get_coriolis_and_centrifugal_compensation_forces()
    )
    gravity = _clone_contiguous(
        view.get_gravity_compensation_forces()
    )
    bias_force = coriolis + gravity
    batch_size = mass_matrix.shape[0]

    if mass_matrix.shape != (batch_size, 35, 35):
        raise RuntimeError(
            "Expected floating-base mass matrix "
            f"[B,35,35], got {tuple(mass_matrix.shape)}"
        )
    if coriolis.shape != (batch_size, 35):
        raise RuntimeError(
            "Expected floating-base Coriolis force "
            f"[B,35], got {tuple(coriolis.shape)}"
        )
    if gravity.shape != (batch_size, 35):
        raise RuntimeError(
            "Expected floating-base gravity force "
            f"[B,35], got {tuple(gravity.shape)}"
        )
    if bias_force.shape != (batch_size, 35):
        raise RuntimeError(
            "Expected floating-base bias force "
            f"[B,35], got {tuple(bias_force.shape)}"
        )

    transform, acceleration_bias, _ = isaac_frame_data(robot)
    mass, bias = dynamics_at_origin(torch.as_tensor(_to_numpy(mass_matrix)),
                                   torch.as_tensor(_to_numpy(bias_force)), transform, acceleration_bias)
    return mass.numpy(), bias.numpy()


def read_live_isaac_snapshot(
    robot,
    isaac_joint_names: list[str],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Read the live state and dynamics in FABRICS generalized order."""
    isaac_to_fabric_joint = np.asarray(
        [isaac_joint_names.index(name) for name in G1_JOINT_NAMES],
        dtype=np.int64,
    )
    coordinate_index = np.asarray(
        list(range(6))
        + [6 + index for index in isaac_to_fabric_joint],
        dtype=np.int64,
    )

    positions, orientations_wxyz = robot.get_world_poses()
    positions = _to_numpy(positions)
    orientations_wxyz = _to_numpy(orientations_wxyz)
    joint_positions_isaac = _to_numpy(robot.get_joint_positions())
    base_velocity = _to_numpy(robot.get_velocities())
    joint_velocity_isaac = _to_numpy(robot.get_joint_velocities())

    q_numpy = np.concatenate(
        (
            positions,
            orientations_wxyz[:, 1:4],
            orientations_wxyz[:, 0:1],
            joint_positions_isaac[:, isaac_to_fabric_joint],
        ),
        axis=1,
    )
    qd_numpy = np.concatenate(
        (
            base_velocity[:, :6],
            joint_velocity_isaac[:, isaac_to_fabric_joint],
        ),
        axis=1,
    )

    mass_raw, bias_raw = read_isaac_dynamics(robot)
    transform, _, offsets = isaac_frame_data(robot)
    qd_numpy[:, :3] -= (transform[:, :3, 3:6] @ torch.as_tensor(base_velocity[:, 3:6])[..., None]).squeeze(-1).numpy()
    mass_isaac = _to_numpy(mass_raw)
    bias_isaac = _to_numpy(bias_raw)
    mass_fabric = mass_isaac[
        :, coordinate_index[:, None], coordinate_index[None, :]
    ]
    bias_fabric = bias_isaac[:, coordinate_index]

    jacobians = jacobians_at_origins(torch.as_tensor(_to_numpy(robot.get_jacobians())), offsets, transform).numpy()
    body_names = list(robot.body_names)
    if jacobians.shape[1] == len(body_names):
        jacobian_body_names = body_names
    elif jacobians.shape[1] == len(body_names) - 1:
        jacobian_body_names = body_names[1:]
    else:
        raise RuntimeError(
            f"Cannot align {jacobians.shape[1]} Jacobian rows with "
            f"{len(body_names)} Isaac body names."
        )
    left_index = jacobian_body_names.index(LEFT_FOOT_BODY)
    right_index = jacobian_body_names.index(RIGHT_FOOT_BODY)
    left_jacobian = jacobians[
        :, left_index, :, :
    ][:, :, coordinate_index]
    right_jacobian = jacobians[
        :, right_index, :, :
    ][:, :, coordinate_index]

    def tensor(value) -> torch.Tensor:
        return torch.as_tensor(value, device=device, dtype=dtype)

    return (
        tensor(q_numpy),
        tensor(qd_numpy),
        tensor(mass_fabric),
        tensor(bias_fabric),
        tensor(left_jacobian),
        tensor(right_jacobian),
    )


def initialize_isaac_dynamics(
    q: torch.Tensor,
    qd: torch.Tensor,
    headless: bool,
    enable_self_collisions: bool | None = None,
    report_contacts: bool = False,
    physics_iterations: list[int] | None = None,
    diagnose_contacts: bool = False,
    motor_armature: bool = False,
    prime_source_contacts: bool = False,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[str],
    object,
    object,
    object,
    object,
]:
    """Read M, h, and both foot Jacobians at exactly the supplied state."""
    if not G1_USD_PATH.is_file():
        raise FileNotFoundError(G1_USD_PATH)

    # SimulationApp must be constructed before importing other Isaac modules.
    from isaacsim import SimulationApp

    global _active_simulation_app
    simulation_app = SimulationApp({"headless": headless})
    _active_simulation_app = simulation_app
    completed = False
    try:
        from isaacsim.core.api import World
        from isaacsim.core.api.materials import PhysicsMaterial
        from isaacsim.core.api.objects import GroundPlane
        from isaacsim.core.prims import Articulation, RigidPrim, SingleGeometryPrim
        from isaacsim.core.utils.stage import add_reference_to_stage
        from isaacsim.storage.native import get_assets_root_path
        import omni.usd
        from pxr import UsdLux, UsdPhysics

        if World.instance():
            World.instance().clear_instance()
        world = World(
            physics_dt=PD_TIMESTEP,
            rendering_dt=1.0 / 50.0,
        )
        ground_material = PhysicsMaterial(
            prim_path="/World/Physics_Materials/tracking_ground",
            static_friction=1.0,
            dynamic_friction=0.8,
            restitution=0.0,
        )
        world.scene.add(
            GroundPlane(
                prim_path="/World/GroundPlane",
                name="ground_plane",
                size=20.0,
                z_position=0.0,
                physics_material=ground_material,
            )
        )
        # In this Isaac build addGroundPlane creates collisionPlane, while
        # the legacy GroundPlane wrapper ALSO enables collision on geom.
        # Two coincident floors split impulses between different filter prims
        # and can hide heel touchdown from the controller. Keep the native
        # physical plane; geom remains the visible mesh only.
        stage = omni.usd.get_context().get_stage()
        native_ground=stage.GetPrimAtPath('/World/GroundPlane/collisionPlane')
        ground_collision_path='/World/GroundPlane/geom'
        if native_ground.IsValid():
            ground_collision_path=str(native_ground.GetPath())
            visual_ground=stage.GetPrimAtPath('/World/GroundPlane/geom')
            if visual_ground.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(visual_ground).GetCollisionEnabledAttr().Set(False)
            SingleGeometryPrim(prim_path=ground_collision_path,collision=True).apply_physics_material(ground_material)
        ground_colliders=[str(prim.GetPath()) for prim in stage.Traverse()
            if str(prim.GetPath()).startswith('/World/GroundPlane/')
            and prim.HasAPI(UsdPhysics.CollisionAPI)
            and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()]
        if ground_colliders != [ground_collision_path]:
            raise RuntimeError(f'Expected one physical ground plane, got {ground_colliders}')
        print(f'Physical ground and contact filter: {ground_collision_path}',flush=True)
        add_reference_to_stage(str(G1_USD_PATH), "/World/G1")

        # Match unitreerobotics/unitree_rl_lab's G1-29DOF scene:
        # a 750-intensity dome light with the Kloofendal clear-sky HDRI.
        stage = omni.usd.get_context().get_stage()
        if report_contacts:
            from pxr import PhysxSchema, UsdPhysics
            for prim in stage.Traverse():
                if str(prim.GetPath()).startswith("/World/G1") and prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.0)
        if enable_self_collisions is not None:
            from pxr import PhysxSchema, UsdPhysics
            for prim in stage.Traverse():
                if str(prim.GetPath()).startswith("/World/G1") and prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                    PhysxSchema.PhysxArticulationAPI.Apply(prim).CreateEnabledSelfCollisionsAttr(
                        enable_self_collisions
                    )
        sky_light = UsdLux.DomeLight.Define(stage, "/World/skyLight")
        sky_light.CreateIntensityAttr(750.0)
        if not headless:
            # The remote HDRI is visual only. Avoid asset-download waits in
            # repeated headless physics trials; retain the interactive scene.
            assets_root = get_assets_root_path(skip_check=True)
            sky_light.CreateTextureFileAttr(
                f"{assets_root}/Isaac/Materials/Textures/Skies/PolyHaven/"
                "kloofendal_43d_clear_puresky_4k.hdr"
            )

        robot = world.scene.add(
            Articulation(prim_paths_expr="/World/G1", name="g1")
        )
        foot_paths = {}
        for prim in stage.Traverse():
            name = prim.GetName()
            if name in {LEFT_FOOT_BODY, RIGHT_FOOT_BODY}:
                foot_paths[name] = str(prim.GetPath())
        missing_feet = {
            LEFT_FOOT_BODY,
            RIGHT_FOOT_BODY,
        } - set(foot_paths)
        if missing_feet:
            raise RuntimeError(
                f"Cannot create foot contact view; missing prims: "
                f"{sorted(missing_feet)}"
            )
        foot_contact_view = world.scene.add(
            RigidPrim(
                prim_paths_expr=[
                    foot_paths[LEFT_FOOT_BODY],
                    foot_paths[RIGHT_FOOT_BODY],
                ],
                name="g1_foot_contacts",
                track_contact_forces=True,
                prepare_contact_sensors=True,
                # Isaac Sim 5.1 requires one filter-pattern group per sensor
                # pattern. There are two sensor patterns (left/right foot);
                # the default [] is normalized to only one group and makes
                # PhysX reject the contact view.
                contact_filter_prim_paths_expr=[[ground_collision_path]] * 2,
                max_contact_count=128 if diagnose_contacts else 0,
            )
        )
        world.reset()
        if not headless:
            from isaacsim.core.utils.viewports import set_camera_view
            origin = q[0,:3].detach().cpu().numpy()
            set_camera_view(eye=np.array([origin[0]+3.,origin[1]+3.,2.2]),
                            target=np.array([origin[0]+.6,origin[1],.7]))
        if physics_iterations is not None:
            robot.set_solver_position_iteration_counts(np.asarray([physics_iterations[0]],dtype=np.int32))
            robot.set_solver_velocity_iteration_counts(np.asarray([physics_iterations[1]],dtype=np.int32))
        print(f"PhysX solver iterations: position={_to_numpy(robot.get_solver_position_iteration_counts()).tolist()}, "
              f"velocity={_to_numpy(robot.get_solver_velocity_iteration_counts()).tolist()}",flush=True)

        q_np = q.detach().cpu().numpy()
        qd_np = qd.detach().cpu().numpy()
        isaac_joint_names = list(robot.dof_names)
        missing = sorted(set(G1_JOINT_NAMES) - set(isaac_joint_names))
        if missing:
            raise RuntimeError(f"Isaac articulation is missing joints: {missing}")
        fabric_to_isaac_joint = np.asarray(
            [G1_JOINT_NAMES.index(name) for name in isaac_joint_names],
            dtype=np.int64,
        )
        # Disable USD joint drives. Feedback is computed explicitly by the
        # main loop and applied only through the effort command channel.
        zero_gains = np.zeros(
            (1, len(isaac_joint_names)), dtype=np.float32
        )
        robot.set_gains(kps=zero_gains, kds=zero_gains)
        if motor_armature:
            from g1_armature import unitree_armatures
            robot.set_armatures(unitree_armatures(isaac_joint_names))
        print('Joint armature:', _to_numpy(robot.get_armatures()).tolist(), flush=True)

        robot.set_world_poses(
            positions=q_np[:, :3],
            orientations=q_np[:, [6, 3, 4, 5]],  # xyzw -> wxyz
        )
        local_com = torch.as_tensor(_to_numpy(robot._physics_view.get_coms())[:, 0, :3].copy())
        root_offset = (_quat_xyzw_to_matrix(torch.as_tensor(q_np[:, 3:7])) @ local_com[..., None]).squeeze(-1)
        root_transform, _ = origin_to_com_map(root_offset, torch.as_tensor(qd_np[:, 3:6]), dimension=6)
        root_com_velocity = (root_transform @ torch.as_tensor(qd_np[:, :6])[..., None]).squeeze(-1).numpy()
        robot.set_velocities(root_com_velocity)
        robot.set_joint_positions(
            q_np[:, 7:36][:, fabric_to_isaac_joint]
        )
        robot.set_joint_velocities(
            qd_np[:, 6:35][:, fabric_to_isaac_joint]
        )
        from isaacsim.core.simulation_manager import SimulationManager
        SimulationManager.get_physics_sim_view().update_articulations_kinematic()
        if prime_source_contacts:
            # world.reset() leaves contacts from the USD default pose. After
            # setting the moving source state, obtain fresh physical contacts
            # before any controller uses force reports or contact points.
            robot.set_joint_efforts(np.zeros_like(q_np[:, 7:36]))
            world.step(render=False)
            print(f"Dynamic start contact priming: {PD_TIMESTEP:g}s passive physics, no subsequent state reset", flush=True)

        mass_matrix, bias_force = read_isaac_dynamics(robot)
        mass_isaac = _to_numpy(mass_matrix).copy()
        bias_isaac = _to_numpy(bias_force).copy()
        transform, _, offsets = isaac_frame_data(robot)
        jacobians = jacobians_at_origins(torch.as_tensor(_to_numpy(robot.get_jacobians()).copy()), offsets, transform).numpy()
        body_names = list(robot.body_names)

        expected_mass_shape = (1, 35, 35)
        if mass_isaac.shape != expected_mass_shape:
            raise RuntimeError(
                f"Expected floating-base mass matrix {expected_mass_shape}, "
                f"got {mass_isaac.shape}."
            )
        if bias_isaac.shape != (1, 35):
            raise RuntimeError(
                "Expected floating-base bias force [1, 35], got "
                f"{bias_isaac.shape}."
            )
        if jacobians.shape[0] != 1 or jacobians.shape[-2:] != (6, 35):
            raise RuntimeError(
                "Expected Isaac Jacobians [1, bodies, 6, 35], got "
                f"{jacobians.shape}."
            )

        # A floating-base articulation contains one Jacobian per body. Keep a
        # defensive fixed-base branch so an accidentally fixed USD fails by
        # body name rather than silently selecting the wrong row.
        if jacobians.shape[1] == len(body_names):
            jacobian_body_names = body_names
        elif jacobians.shape[1] == len(body_names) - 1:
            jacobian_body_names = body_names[1:]
        else:
            raise RuntimeError(
                f"Cannot align {jacobians.shape[1]} Jacobian rows with "
                f"{len(body_names)} Isaac body names."
            )
        try:
            left_index = jacobian_body_names.index(LEFT_FOOT_BODY)
            right_index = jacobian_body_names.index(RIGHT_FOOT_BODY)
        except ValueError as exc:
            raise RuntimeError(
                f"Foot body missing; Isaac bodies are {body_names}"
            ) from exc

        # Isaac generalized order is [base6, Isaac DOFs]. Reorder both axes of
        # M and the columns of J into [base6, G1_JOINT_NAMES].
        isaac_joint_index = {
            name: index for index, name in enumerate(isaac_joint_names)
        }
        coordinate_index = np.asarray(
            list(range(6))
            + [
                6 + isaac_joint_index[name]
                for name in G1_JOINT_NAMES
            ],
            dtype=np.int64,
        )
        mass_fabric = mass_isaac[
            :, coordinate_index[:, None], coordinate_index[None, :]
        ]
        bias_fabric = bias_isaac[:, coordinate_index]
        left_jacobian = jacobians[
            :, left_index, :, :
        ][:, :, coordinate_index]
        right_jacobian = jacobians[
            :, right_index, :, :
        ][:, :, coordinate_index]
        result = (
            mass_fabric,
            bias_fabric,
            left_jacobian,
            right_jacobian,
            isaac_joint_names,
            simulation_app,
            world,
            robot,
            foot_contact_view,
        )
        completed = True
        return result
    except BaseException:
        # Isaac Sim 5.1 shutdown can terminate the process before Python gets
        # a chance to display the uncaught exception. Print it first so a
        # failed dynamics query remains diagnosable.
        traceback.print_exc()
        sys.stderr.flush()
        raise
    finally:
        # Isaac Sim 5.1 fast shutdown can terminate the Python process. Keep
        # the app alive after a successful query and close it only after the
        # projection results have been printed and saved.
        if not completed:
            simulation_app.close()


def save_reference_failure(args, error, control_time, q, qd, forces, history, servo_history=()):
    path=args.output_path.with_suffix('.reference_failure.npz')
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,reason=str(error),control_time=control_time,
        source_frame_count=np.asarray(getattr(args,'source_frame_count',0)),
        source_start_frame=np.asarray(getattr(args,'source_start_frame',0)),
        controller_options=np.asarray(json.dumps(vars(args),default=str,sort_keys=True)),
        q=q.detach().cpu().numpy(),qd=qd.detach().cpu().numpy(),
        foot_forces=forces.detach().cpu().numpy(),
        motion_tracking_valid=np.asarray(False),
        simulation_completed=np.asarray(False),
        stopped_at_motion_end=np.asarray(False),
        **{'servo_'+key:np.asarray([row[key] for row in servo_history])
           for key in (servo_history[0] if servo_history else {})},
        **{'history_'+key:np.asarray([row[key] for row in history])
           for key in (history[0] if history else {})})
    print(f'[fail] reference planning; saved: {path}',flush=True)
    # A bound-construction failure occurs before the QP failure handler. Retain
    # the same measured landing audit for its valid prefix, without certifying
    # the failed state or changing the controller's stop decision.
    plan_path = args.output_path.with_suffix('.steps_plan.json')
    if len(history) >= 2 and plan_path.exists():
        try:
            from analyze_bounded_steps import summarize as summarize_steps
            plan = json.loads(plan_path.read_text())
            failure = dict(stop_reason=str(error), failed_time_s=float(control_time))
            with np.load(path) as recording:
                steps = summarize_steps(recording, plan)
                steps.update(failure)
                args.output_path.with_suffix('.steps.json').write_text(
                    json.dumps(steps, indent=2)+'\n')
                if getattr(args, 'track_full_motion', False):
                    from analyze_motion_tracking import summarize_full_motion
                    tracking = summarize_full_motion(recording, plan)
                    tracking.update(failure)
                    args.output_path.with_suffix('.tracking.json').write_text(
                        json.dumps(tracking, indent=2)+'\n')
            print(f"[fail] verified source landings={steps['measured_steps']}/"
                  f"{steps['expected_steps']}; controller stopped: {error}", flush=True)
        except (KeyError, ValueError, OSError) as audit_error:
            print(f'[fail] prefix audit unavailable: {audit_error}', flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run G1 whole-body tracking tasks and directly project their 35-D "
            "C-space acceleration through foot-contact constraints."
        )
    )
    parser.add_argument("--g1-md", action="store_true",
                        help="Use g1.md Fabric composition -> constrained dynamics -> persistent MIT integration in Isaac Sim.")
    parser.add_argument("--motion-pkl", type=Path, default=None)
    parser.add_argument("--motion-start-in-motion", action="store_true",
                        help="Initialize from source q/v and touching soles, then track the source clock without rest warmup or a managed transfer.")
    parser.add_argument("--dynamic-unload-weight", type=float, default=0.,
                        help="Optional outgoing-wrench penalty during measured double support before a planned dynamic foot release.")
    parser.add_argument("--preserve-reference-acceleration", action=argparse.BooleanOptionalAction, default=False,
                        help="Opt in to raw source acceleration beyond task caps in dynamic tracking; physical QP constraints still apply.")
    parser.add_argument("--swing-midpoint-clearance", type=float, default=0.,
                        help="Mid-swing clearance using native foot colliders and leg IK; one smooth bump fills each source midpoint deficit.")

    parser.add_argument("--motion-feedback-scale", type=float, default=1.,
                        help="Scale posture/foot/body error stiffness and sqrt-scale damping; source feedforward and COM gains remain unchanged.")
    parser.add_argument("--track-full-motion", action="store_true",
                        help="Dynamic tracking from frame zero through the last PKL sample; audit all source landings without requiring a stationary endpoint.")
    parser.add_argument("--tracking-steps", type=int, default=0,
                        help="Verify N source landings from frame zero (requires contact labels); managed mode settles, dynamic mode includes an initial partial swing and stops after measured landings.")
    parser.add_argument("--mocap-pipeline", action="store_true",
                        help="Prepare C2 reference in memory, then run managed unified Fabric + WBC in this simulation.")
    parser.add_argument("--joint-velocity-limits", action=argparse.BooleanOptionalAction, default=True,
                        help="Enforce URDF joint speeds on next-step predicted physical velocity.")
    parser.add_argument("--acceleration-and-braking-limits", action=argparse.BooleanOptionalAction, default=False,
                        help="Opt in to software acceleration caps and joint braking bounds (disabled by default).")
    parser.add_argument("--reference-acceleration-cap", type=float, default=50.,
                        help="Planning cap for C2 constant-rate joint acceleration, rad/s^2; not actuator certification.")
    parser.add_argument("--contacts-pkl", type=Path,
                        help="Optional matching pose/fps companion with contact labels.")
    parser.add_argument("--reference-mode", choices=("unified", "legacy-managed"), default="unified",
                        help="Unified: derive every motion target from one q/v/a; legacy-managed: old independent reference adjustments.")
    parser.add_argument("--coordinated-source-start", action="store_true",
                        help="Coordinate first load transfer before source tracking; reject an infeasible or unplanned continuation.")
    parser.add_argument("--source-support-switch", action="store_true",
                        help="After checked landing stop, transfer measured load to the new foot and lift the old foot.")
    parser.add_argument("--source-step-profile", type=Path,
                        help="JSON candidate option overrides with first/following sections; every path still requires preflight.")
    parser.add_argument("--source-contact-events", action="store_true",
                        help="Follow actual PKL support events and audit a final double-support tail.")
    parser.add_argument("--source-tail-duration", type=float, default=1.)
    parser.add_argument("--source-step-count", type=int, default=1,
                        help="Number of checked physical swings, including recovery; each transfer and swing must pass preflight.")
    parser.add_argument("--source-recovery-after-step", type=int, default=0,
                        help="Hold the source frame after this step; remaining requested steps approach that fixed pose (0 disables recovery).")
    parser.add_argument("--source-recovery-step-count", type=int, default=0,
                        help="Even number of recovery swings before resuming source frames; 0 keeps recovery indefinite.")
    parser.add_argument("--recovery-left-transfer-roll-offset", type=float, default=0.,
                        help="Local pelvis roll during left-support transfer after a recovery step, radians.")
    parser.add_argument("--recovery-maximum-step-distance", type=float, default=.42)
    parser.add_argument("--recovery-landing-com-fraction", type=float, default=.4)
    parser.add_argument("--following-step-landing-fraction", type=float, default=.4)
    parser.add_argument("--following-step-landing-height", type=float, default=.66)
    parser.add_argument("--later-step-stop-duration", type=float, default=.4,
                        help="Audited double-support stop duration for source steps 3 onward.")
    parser.add_argument("--later-step-landing-fraction", type=float, default=.3,
                        help="Landing fraction for source steps 3 onward; explicitly adapted reference.")
    parser.add_argument("--later-transfer-height-offset", type=float, default=-.02,
                        help="COM height change for transfers after step 2 onward, metres.")
    parser.add_argument("--later-transfer-lift-height", type=float, default=.03,
                        help="Foot lift for transfers after step 2 onward, metres.")
    parser.add_argument("--mocap-first-step-project-torque", action="store_true",
                        help="Simulation-only bounded probe: permit torque-infeasible nominal reference; WBC motor limits stay hard.")
    parser.add_argument("--mocap-first-step-report", type=Path,
                        help="Bounded first-step probe after measured unloading, rebuilt from live state; not four-step certification.")
    parser.add_argument("--mocap-unload-test", action="store_true",
                        help="Bounded measured-state full-body double-support unloading diagnostic; no steps executed.")
    parser.add_argument("--mocap-unload-weight", type=float, default=0.,
                        help="Experimental direct outgoing-wrench objective weight; disabled by default.")
    parser.add_argument("--source-step-stop-duration", type=float, default=0.,
                        help="Plan a checked double-support stop after first touchdown; seconds, 0 disables.")
    parser.add_argument("--source-step-landing-search-depth", type=float, default=0.,
                        help="Bounded late-swing downward contact search, metres (0 to .01); measured load is required.")
    parser.add_argument("--source-step-landing-fraction", type=float, default=1.,
                        help="Explicit shortened bridge landing in (0,1]; requires coordinated source start.")
    parser.add_argument("--unified-source-step-test", action="store_true",
                        help="Measured first source swing/landing experiment; stops at its endpoint, not full PKL tracking.")
    parser.add_argument("--unified-transfer-test", action="store_true",
                        help="Diagnostic: coordinated full-body first transfer and 4 cm lift; does not complete the PKL.")
    parser.add_argument("--reference-manager", action="store_true",
                        help="Experimental single-owner reference clock/contact scheduler for flat-ground walking.")
    parser.add_argument("--reference-footsteps", choices=("compact", "source"), default="compact",
                        help="Managed foot placement: compact relative steps, or retimed source FK with startup bridges and terminal ground correction.")
    parser.add_argument("--until-motion-end", action="store_true",
                        help="Require the final source frame and 0.5 s stable double support; --duration is the maximum simulation budget.")
    parser.add_argument("--toe-transfer-load-fraction", type=float, default=.8,
                        help="Incoming load fraction required to start toe rolling; the separate takeoff gate stays at 0.8.")
    parser.add_argument("--source-pelvis-weight", type=float, default=0.,
                        help="Additional source pelvis orientation metric relative to torso; requires managed source footsteps.")
    parser.add_argument("--source-momentum-reference", action="store_true",
                        help="Track source angular momentum at the managed motion clock instead of damping it to zero.")
    parser.add_argument("--wrist-body-avoidance", action="store_true",
                        help="Extend wrist/hand mesh barriers to torso and opposite shoulders; requires wrist/hip barriers.")
    parser.add_argument("--align-takeoff-frame", action="store_true",
                        help="Complete PKL preparation frames before releasing the swing foot after balance is ready.")
    parser.add_argument("--takeoff-lead", type=float, default=0.,
                        help="Allow loaded, preview-authorized takeoff up to this many source seconds early (0 to .15).")
    parser.add_argument("--source-swing-reference", action=argparse.BooleanOptionalAction, default=False,
                        help="Use coupled source FK pose and derivatives for swing feet, with a measured-start position bridge.")
    parser.add_argument("--balance-priority", action="store_true",
                        help="Experimental two-stage QP: preserve achieved COM acceleration before optimizing motion tracking.")
    parser.add_argument("--transfer-balance-priority", action="store_true",
                        help="Prioritize COM after a unified transfer starts, or during legacy dynamic double support.")
    parser.add_argument("--swing-clearance-priority", action="store_true",
                        help="Include swing-foot vertical acceleration in the COM priority stage to preserve lift and landing.")
    parser.add_argument("--source-swing-pose-priority", action="store_true",
                        help="For unified source steps, prioritize COM then airborne-foot position and orientation.")
    parser.add_argument("--source-swing-position-priority", action="store_true",
                        help="For unified source steps, prioritize COM then airborne-foot XYZ before posture tracking.")
    parser.add_argument("--swing-position-priority", action="store_true",
                        help="Include all three swing-foot position coordinates in the COM priority stage.")
    parser.add_argument("--swing-pose-priority", action="store_true",
                        help="Include swing-foot position and orientation in the COM priority stage.")
    parser.add_argument("--upper-body-priority", action=argparse.BooleanOptionalAction, default=False,
                        help="Include torso orientation and waist/arm posture in active COM priority stages.")
    parser.add_argument("--whole-body-posture-priority", action="store_true",
                        help="Include all joint references with torso/foot tasks below balance; excludes upper-body-only priority.")
    parser.add_argument("--body-orientation-priority", action="store_true",
                        help="Include torso/pelvis orientation in active balance stages while keeping arm posture in the tracking stage.")
    parser.add_argument("--landing-orientation-priority", action="store_true",
                        help="Prioritize the free angular coordinates of observed incoming contact patches during load transfer.")
    parser.add_argument("--swing-touchdown-lead", type=float, default=.15,
                        help="How many seconds before the COM preview endpoint the foot target reaches the floor.")
    parser.add_argument("--minimum-step-duration", type=float, default=.8,
                        help="Lower bound for dynamic step time; 0 uses source duration and the swing-speed bound.")
    parser.add_argument("--separate-balance-priority", action="store_true",
                        help="Optimize COM before other selected priority tasks, retaining both stages before final tracking.")
    parser.add_argument("--balance-body-priority", action="store_true",
                        help="With separate balance, optimize COM and body/pelvis attitude first; source momentum, feet and posture follow.")
    parser.add_argument("--centroidal-balance-priority", action="store_true",
                        help="Include angular momentum alongside COM in active priority tasks; separate-balance-priority isolates both before pose tracking.")
    parser.add_argument("--posture-last-priority", action="store_true",
                        help="Place all 29 joint posture rows after selected COM, foot and orientation tasks in the QP hierarchy.")
    parser.add_argument("--joint-limit-avoidance-weight", type=float, default=0.,
                        help="Optional soft anticipatory joint braking metric. Simulator joint stops stay enabled; predictive braking requires --acceleration-and-braking-limits or --physical-braking-start.")
    parser.add_argument("--joint-limit-time-constant", type=float, default=None,
                        help="Optional measured-joint velocity damper time in seconds; ramps in over 0.5 s after warmup.")
    parser.add_argument("--joint-limit-barrier-bandwidth", type=float, default=None,
                        help="Joint-distance acceleration barrier bandwidth in 1/s when predictive braking is active; setting this alone does not enable braking bounds.")
    parser.add_argument("--joint-limit-impact-recovery", action="store_true",
                        help="Require measured stopping-distance braking after contact impulses; retain QP effort/acceleration limits.")
    parser.add_argument("--flat-foot-spawn", action="store_true",
                        help="Prepare a nearby flat-foot initial pose before spawning; source PKL and live states remain untouched.")
    parser.add_argument("--early-touchdown", action="store_true",
                        help="Activate measured flat-foot touchdown immediately; keep load confirmation in SETTLE.")
    parser.add_argument("--observed-contact-patches", action="store_true",
                        help="Model both feet's measured heel/toe/point patches and recognize loaded heel touchdown immediately.")
    parser.add_argument("--physics-iterations", type=int, nargs=2, metavar=('POSITION','VELOCITY'),
                        help="Override PhysX articulation solver iterations; default keeps the USD settings.")
    parser.add_argument("--motor-armature", action="store_true",
                        help="Use Unitree G1 MIMIC reflected motor inertia in simulation and QP dynamics.")
    parser.add_argument("--source-bridge-duration", type=float,
                        help="Seconds for the measured-to-source swing transition; default uses the full swing.")
    parser.add_argument("--landing-speed-limit", type=float,
                        help="Optional approach speed envelope in m/s for the actual foot collision shapes, e.g. .05.")
    parser.add_argument("--rolling-contact-kinematics-start", type=float, default=float('inf'),
                        help="Simulation seconds to enable native-sphere contact migration bias; disabled by default.")
    parser.add_argument("--landing-speed-limit-start", type=float, default=0.,
                        help="Simulation seconds before enabling the landing speed envelope; source auditing stays active.")
    parser.add_argument("--horizontal-balance-priority", action="store_true",
                        help="Prioritize COM XY; leave height in the full posture objective. Requires separate balance priority.")
    parser.add_argument("--tracking-servo", action="store_true",
                        help="Direct WBC with 500 Hz feedback around its predicted joint trajectory, reset to measurements at each QP tick.")
    parser.add_argument("--contact-aware-servo", action="store_true",
                        help="Use contact-conditioned servo gains; measured unloading selects free gains at 500 Hz.")
    parser.add_argument("--nominal-implicit-servo", action="store_true",
                        help="Use nominal motor gains with coupled implicit feedback instead of reducing gains on edge contact.")
    parser.add_argument("--control-frequency", type=int, choices=(50,100,250,500,1000),default=500,
                        help="Fabric/QP updates per second; physics/feedback are 500 Hz normally, 2000 Hz with --g1-md.")
    parser.add_argument('--late-physics-substeps-start', type=float, default=float('inf'),
                        help='Simulation seconds to subdivide each torque interval without extra control updates')
    parser.add_argument('--late-physics-substeps', type=int, choices=(1,2,4,8), default=1)
    parser.add_argument("--physics-frequency", type=int, choices=(500,1000,2000,4000), default=None,
                        help="Physics/feedback rate; must be an integer multiple of the control rate. Defaults to 500, or 2000 with --g1-md.")
    parser.add_argument("--qp-solver", choices=("admm", "osqp", "daqp"), default="admm",
                        help="QP backend: batched Torch ADMM, CPU OSQP, or optional native DAQP with KKT checks.")
    parser.add_argument("--qp-reuse-workspace", action=argparse.BooleanOptionalAction, default=False,
                        help="Keep separate OSQP workspaces for priority/tracking stages; retry cold if reuse fails physical checks.")
    parser.add_argument("--qp-active-set-predictor", action="store_true",
                        help="Try the previous active constraints, verify full KKT/physical residuals, and otherwise use OSQP.")
    parser.add_argument("--ground-motion", action=argparse.BooleanOptionalAction, default=True,
                        help="Translate the PKL reference vertically at spawn so the lowest modeled sole touches z=0.")
    parser.add_argument("--motion-ramp-duration", type=float, default=1.0,
                        help="Seconds to smoothly increase playback speed after warmup; 0 disables the ramp.")
    parser.add_argument("--motion-interpolation", choices=("linear", "pchip", "pchip-analytic", "bspline-analytic"), default="pchip-analytic",
                        help="PCHIP preserves samples with C1 interpolation; pchip-analytic also differentiates references analytically.")
    parser.add_argument("--task-profile", choices=("legacy", "dynamics-aware"), default="dynamics-aware",
                        help="Use physical task-level WBC with momentum and stance/swing gating.")
    parser.add_argument("--hand-tracking", choices=("off", "position", "pose"), default="position",
                        help="Dynamics-aware hand task: position only by default; pose also tracks wrist orientation.")
    parser.add_argument("--collision-obstacles", type=Path, default=None,
                        help="JSON array of static world spheres [[x,y,z,r],...], dynamics-aware only.")
    parser.add_argument("--disable-collision-task", action="store_true",
                        help="Disable soft sphere repulsion for a dynamics-aware ablation.")
    parser.add_argument("--wrist-hip-avoidance", action="store_true",
                        help="Add conservative mesh-derived wrist/hip self-collision tasks.")
    parser.add_argument("--source-arm-roll-offset", type=float, default=0.,
                        help="Explicit outward shoulder-roll adaptation in radians; original PKL remains the tracking audit target.")
    parser.add_argument("--convex-wrist-collision", action="store_true",
                        help="Experimental CPU convex-mesh wrist/body distance; closed-loop tracking is not validated and collision margin is retained.")
    parser.add_argument("--wrist-hip-barrier", action="store_true",
                        help="Add distance/approach-speed QP constraints for the extended wrist/hip covers.")
    parser.add_argument("--disable-momentum-task", action="store_true",
                        help="Disable momentum tracking for a dynamics-aware ablation.")
    parser.add_argument("--limit-predicted-acceleration", action="store_true",
                        help="Experimental: also constrain servo-inclusive physical acceleration.")
    parser.add_argument("--task-track-nominal", action="store_true",
                        help="Experimental: task WBC tracks nominal physical acceleration; retain PD response in total-effort constraints.")
    parser.add_argument("--use-sole-wrench-geometry", action=argparse.BooleanOptionalAction, default=True,
                        help="Diagnostic: use a conservative G1 sole rectangle (heel .05, toe .12, half-width .025 m) and shift ankle moments to the sole at -.035 m.")
    parser.add_argument("--diagnose-contacts", action="store_true",
                        help="Record actual PhysX foot-ground contact points and forces at every control tick.")
    parser.add_argument("--swing-metric-scale", type=float, default=None,
                        help="Experimental managed swing task weight; requires --reference-manager. Uses YAML default when omitted.")
    parser.add_argument("--posture-metric-scale", type=float, default=None,
                        help="Experimental managed joint posture task weight; requires --reference-manager.")
    parser.add_argument("--g1-com-priority", action="store_true",
                        help="Experimental g1.md: prioritize COM under all hard constraints, then optimize the composed Fabric root.")
    parser.add_argument("--diagnose-com", action="store_true",
                        help="Record requested/Fabric/projected/QP/observed world COM accelerations without changing objectives.")
    parser.add_argument("--diagnose-qp-time", type=float, action="append", default=[],
                        help="Read-only full QP snapshot at the nearest control tick; repeat for multiple times. Requires --diagnose-com.")
    parser.add_argument("--single-support-test", choices=("left", "right"), default=None,
                        help="Diagnostic: freeze motion, transfer load, gate lift-off and hold one-foot support.")
    parser.add_argument("--contact-cycle-test", choices=("left", "right"), default=None,
                        help="Measured-gated transfer, lift, hold, lower and settle test; selects support foot.")
    parser.add_argument("--support-transfer-duration", type=float, default=3.,
                        help="Duration of the isolated single-support COM transfer reference, seconds.")
    parser.add_argument("--cycle-crouch-depth", type=float, default=0.,
                        help="Optional 3-second preparation: lower COM and bend knees before the contact-cycle transfer, metres.")
    parser.add_argument("--weight-transfer-test", action="store_true",
                        help="Diagnostic: freeze frame zero and shift COM toward the right foot over three seconds after warmup.")
    parser.add_argument("--fast-command-sync", action="store_true",
                        help="Experimental: use 20/8/30 acceleration-level command synchronization instead of 1/2/5.")
    parser.add_argument("--direct-wbc", action="store_true",
                        help="Apply constrained WBC torques with 500 Hz velocity damping, without an integrated joint position servo.")
    parser.add_argument("--align-balance-pelvis", action="store_true",
                        help="Experimental: give pelvis and COM consistent horizontal balance references.")
    parser.add_argument("--jdot-filter-alpha", type=float, default=0.95,
                        help="Contact Jdot*v exponential filter retention in [0,1); 0 disables lag.")
    parser.add_argument("--stabilize-foot-rotation", action=argparse.BooleanOptionalAction, default=None,
                        help="Include support-foot rotation in the bounded contact pose/velocity stabilizer.")
    parser.add_argument("--report-self-contacts", action="store_true",
                        help="Diagnostic: log first penetrating contact for each robot collider pair.")
    parser.add_argument("--require-support-load", action="store_true",
                        help="Experimental 80%% support-load gate with contact-aware motion phase timing.")
    parser.add_argument(
        "--left-shoulder-roll-offset", type=float, default=0.0,
        help="Explicit reference adjustment in rad for clearance experiments; does not modify the PKL.",
    )
    parser.add_argument(
        "--disable-self-collisions", action="store_true",
        help="Diagnostic only: disable self-collisions before creating PhysX; not a tracking success workaround.",
    )
    parser.add_argument(
        "--servo-bandwidth", type=float, default=30.0,
        help="Explicit inertia-aware PD bandwidth in rad/s; coupled step budget still applies.",
    )
    parser.add_argument(
        "--fixed-joint-test", action="store_true",
        help="Servo diagnostic: hold initial joints and initial QP feedforward; do not play motion.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=OUTPUT_PATH,
        help="NPZ path for the final controller snapshot.",
    )
    parser.add_argument(
        "--motion-frame",
        type=int,
        default=0,
        help="Initial motion frame; -1 explicitly selects the middle frame.",
    )
    parser.add_argument(
        "--motion-speed",
        type=float,
        default=1.0,
        help="Motion playback speed multiplier (default: 1.0).",
    )
    parser.add_argument(
        "--motion-warmup-duration",
        type=float,
        default=1.0,
        help="Seconds to stabilize at the initial pose before playback.",
    )
    parser.add_argument(
        "--allow-flight",
        action="store_true",
        help="Honor mode-0 PKL phases instead of retaining near-ground support.",
    )
    parser.add_argument(
        "--contact-mode",
        choices=("flight", "left", "right", "double"),
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--regularization",
        type=float,
        default=1.0e-6,
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Projection/Fabric device: auto, cpu, cuda, or cuda:0.",
    )
    parser.add_argument(
        "--show-isaac-window",
        action="store_true",
        help="Show Isaac while the PD control loop runs.",
    )
    parser.add_argument("--keep-isaac-window", action="store_true",
                        help="Keep the rendered scene paused after tracking until the viewer is closed.")
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="PD/physics simulation duration in seconds (default: 10).",
    )
    parser.add_argument(
        "--reference-contact-height-tolerance",
        type=float,
        default=0.04,
        help="Foot-height tolerance in metres for PKL contact inference.",
    )
    parser.add_argument(
        "--contact-stabilization-kp",
        type=float,
        default=None,
        help="Support-foot pose stabilization gain (default: 20 with ReferenceManager, otherwise 0).",
    )
    parser.add_argument(
        "--contact-stabilization-kd",
        type=float,
        default=None,
        help="Support-foot velocity stabilization gain (default: 8 with ReferenceManager, otherwise 0).",
    )
    parser.add_argument(
        "--contact-stabilization-max-acceleration",
        type=float,
        default=5.0,
        help=(
            "Maximum horizontal contact-stabilization acceleration in "
            "m/s^2 (default: 5)."
        ),
    )
    parser.add_argument(
        "--com-lookahead",
        type=float,
        default=0.25,
        help="Seconds to anticipate the next support foot for COM transfer.",
    )
    parser.add_argument(
        "--com-target-time-constant",
        type=float,
        default=0.15,
        help="COM target smoothing time constant in seconds.",
    )
    parser.add_argument(
        "--contact-switch-confirm-updates",
        type=int,
        default=3,
        help="Consecutive 50 Hz PKL samples required before lift-off.",
    )
    parser.add_argument(
        "--qp-validation-tolerance",
        type=float,
        default=5.0e-2,
        help=(
            "Tolerance for floating-base and contact-acceleration QP "
            "residuals (default: 5e-2 for float32 GPU solves)."
        ),
    )
    parser.add_argument(
        "--qp-acceleration-limit-scale",
        type=float,
        default=5.0,
        help="Scale Fabric acceleration limits for contact-closure QP.",
    )
    parser.add_argument(
        "--qp-base-acceleration-weight",
        type=float,
        default=1.0,
        help=(
            "QP tracking weight for the six unactuated base accelerations; "
            "joint accelerations retain unit weight (default: 1)."
        ),
    )
    parser.add_argument(
        "--qp-track-predicted-base",
        action="store_true",
        help=(
            "Experimental: track servo-inclusive base acceleration in QP. "
            "Disabled by default: static-reference regression worsened."
        ),
    )
    parser.add_argument('--qp-base-linear-acceleration-weight', type=float, default=None,
                        help='Override only the three base translation weights; rotation keeps --qp-base-acceleration-weight.')
    parser.add_argument('--qp-base-linear-weight-start', type=float, default=0.,
                        help='Simulation seconds to start changing the base translation cost.')
    parser.add_argument('--qp-base-linear-weight-ramp', type=float, default=0.,
                        help='Quintic transition duration in seconds; zero applies the weight immediately.')
    parser.add_argument(
        "--maximum-com-tracking-error",
        type=float,
        default=0.25,
        help=(
            "Maximum final COM position error in metres for motion tracking "
            "validation (default: 0.25)."
        ),
    )
    parser.add_argument(
        "--minimum-com-height-ratio",
        type=float,
        default=0.65,
        help=(
            "Minimum final COM height as a fraction of its initial height "
            "for motion tracking validation (default: 0.65)."
        ),
    )
    parser.add_argument(
        "--maximum-joint-position-rmse",
        type=float,
        default=0.35,
        help=(
            "Maximum final actuated-joint position RMSE in radians for "
            "motion tracking validation (default: 0.35)."
        ),
    )
    parser.add_argument('--vertical-feedback-start', type=float, default=2.44)
    parser.add_argument('--joint-feedback-limit', type=float, default=30.)
    parser.add_argument('--com-acceleration-preservation-weight', type=float, default=0.,
                        help='Preserve composed COM acceleration in projection and QP using the live COM Jacobian; no task requests.')
    parser.add_argument('--qp-com-feedback-scale', type=float, default=0.,
                        help='Add the original COM feedforward+PD objective to amp-full QP, scaled by its task metric; 0 restores projected-only tracking.')
    parser.add_argument('--qp-com-feedback-start', type=float, default=0.,
                        help='Simulation time at which the original COM objective starts blending in.')
    parser.add_argument('--qp-com-feedback-ramp', type=float, default=0.,
                        help='Quintic blend duration for the original COM objective, in seconds.')
    parser.add_argument('--qp-com-feedback-axes', choices=('xy', 'xyz'), default='xyz',
                        help='Axes of the optional original COM objective; xy preserves the existing vertical task.')
    parser.add_argument('--qp-fabric-metric', action='store_true',
                        help='Use the complete Fabric metric and unprojected acceleration as the single-QP objective, preserving the original task tradeoffs.')
    parser.add_argument('--projection-joint-correction-weight', type=float, default=0.,
                        help='Extra joint correction cost in contact projection; zero uses the physical mass metric.')
    parser.add_argument('--fabric-angular-feedback-limit',
                        type=lambda value: None if value == 'default' else float(value), default=None,
                        help='Body/pelvis feedback cap before projection (rad/s^2); default restores the configured per-axis caps.')
    parser.add_argument('--clock-transition-start', type=float, default=1.34)
    parser.add_argument('--clock-transition-ramp', type=float, default=.2)
    parser.add_argument('--clock-final-speed', type=float, default=.7)
    parser.add_argument('--clock-second-start',type=float,default=100.)
    parser.add_argument('--clock-second-ramp',type=float,default=.3)
    parser.add_argument('--clock-second-speed',type=float,default=None)
    parser.add_argument('--clock-third-start',type=float,default=100.)
    parser.add_argument('--clock-third-ramp',type=float,default=.2)
    parser.add_argument('--clock-third-speed',type=float,default=None)
    parser.add_argument('--late-foot-linear-feedback-start',type=float,default=float('inf'))
    parser.add_argument('--late-foot-linear-feedback-limit',type=float,default=None,
                        help='Optional norm cap for swing-foot position feedback; analytic feedforward and physical limits are retained.')
    parser.add_argument('--late-foot-linear-feedback-side',choices=('left','right','both'),default='both')
    parser.add_argument('--prepared-source-clock',type=Path,default=None,
                        help='Optional offline, hash-bound C2 source clock; original source fidelity remains enforced.')
    parser.add_argument('--clock-fourth-start',type=float,default=100.)
    parser.add_argument('--clock-fourth-ramp',type=float,default=.2)
    parser.add_argument('--clock-fifth-start',type=float,default=100.)
    parser.add_argument('--clock-fifth-ramp',type=float,default=.12)
    parser.add_argument('--clock-fifth-speed',type=float,default=None,
                        help='Optional subsequent smooth speed transition, in control seconds before the landing clock.')
    parser.add_argument('--clock-fourth-speed',type=float,default=None,
                        help='Optional positive fourth source speed after the third transition; preserves original source-frame fidelity.')
    parser.add_argument('--flight-feedback-limit', type=float, default=300.)
    parser.add_argument('--foot-feedback-starts', type=float, nargs=2, default=[1.3,.88])
    parser.add_argument('--foot-feedback-limits', type=float, nargs=2, default=[100.,60.])
    parser.add_argument('--foot-feedback-joints', choices=['ankles','leg'], nargs=2, default=['ankles','ankles'])
    parser.add_argument('--stop-at-fidelity-loss', action='store_true')
    parser.add_argument('--shoulder-roll-weight-scale', type=float, default=1.,
                        help='Experimental posture metric multiplier; disabled at the default 1.')
    parser.add_argument('--shoulder-roll-weight-start', type=float, default=float('inf'),
                        help='Simulation/control time in seconds, independent of the retimed source clock.')
    parser.add_argument('--shoulder-roll-weight-ramp', type=float, default=.1)
    parser.add_argument('--shoulder-roll-weight-side', choices=('left','right','both'), default='both')
    parser.add_argument('--shoulder-roll-guard', action='store_true',
                        help='Compare shoulder-weight candidates against a same-state baseline before execution.')
    parser.add_argument('--shoulder-roll-guard-relative', type=float, default=.02)
    parser.add_argument('--shoulder-roll-guard-slack', type=float, default=1.)
    parser.add_argument('--diagnostic-continue-after-fidelity-loss',dest='stop_at_fidelity_loss',action='store_false',default=argparse.SUPPRESS,help='Continue physics after the first source-fidelity violation; the original-source audit still fails.')
    parser.add_argument('--full-body-preview-start', type=float, default=100.)
    parser.add_argument('--full-body-preview-horizon', type=int, default=12)
    parser.add_argument('--full-body-preview-period', type=float, default=.02)
    parser.add_argument('--full-body-preview-track-wrench', action='store_true')
    parser.add_argument('--full-body-preview-warm-start', action='store_true')
    parser.add_argument('--full-body-preview-reuse-verified', action='store_true')
    parser.add_argument('--full-body-preview-guarded', action='store_true')
    parser.add_argument('--full-body-preview-collision-barrier', action='store_true')
    parser.add_argument('--full-body-preview-fine-control', action='store_true')
    parser.add_argument('--full-body-preview-state-feedback', action='store_true')
    parser.add_argument('--full-body-preview-release-points', action='store_true')
    parser.add_argument('--full-body-preview-native-contact-timing', action='store_true')
    parser.add_argument('--full-body-preview-exact-rotation', action='store_true')
    parser.add_argument('--full-body-preview-recover-joint-reserve', action='store_true')
    parser.add_argument('--full-body-preview-damp-first-control', action='store_true')
    parser.add_argument('--full-body-preview-separating-points', action='store_true')
    parser.add_argument('--full-body-preview-surface-kinematics', action='store_true')
    parser.add_argument('--com-feedback-start', type=float, default=100.)
    parser.add_argument('--com-feedback-limit', type=float, default=12.)
    parser.add_argument('--com-feedback-gain-scale', type=float, default=1.)
    parser.add_argument('--com-feedback-axes', choices=['xy','xyz'], default='xy')
    parser.add_argument('--com-feedback-metric-scale', type=float, default=1.)
    parser.add_argument('--com-feedback-metric-start', type=float, default=1.34)
    parser.add_argument('--global-posture-feedback-start', type=float, default=100.)
    parser.add_argument('--global-posture-feedback-joints', choices=['all','arms'], default='all')
    parser.add_argument('--global-posture-feedback-metric-scale', type=float, default=1.)
    parser.add_argument('--model-effort-start',type=float,default=float('inf'))
    parser.add_argument('--model-effort-ramp',type=float,default=.1)
    parser.add_argument('--support-load-floor-start',type=float,default=float('inf'))
    parser.add_argument('--support-load-floor',type=float,default=100.)
    parser.add_argument('--support-load-floor-transfer',action='store_true')
    parser.add_argument('--late-support-load-floor-start',type=float,default=float('inf'))
    parser.add_argument('--late-support-load-floor',type=float,default=None,
                        help='Optional late minimum force on an existing loaded support contact, in N.')
    parser.add_argument('--late-support-load-floor-ramp',type=float,default=.06)
    parser.add_argument('--minimum-support-acceleration-start',type=float,default=100.)
    parser.add_argument('--minimum-support-acceleration',type=float,default=-3.)
    parser.add_argument('--minimum-support-acceleration-metric-scale',type=float,default=2.)
    parser.add_argument('--maximum-support-acceleration',type=float,default=None,
                        help='Optional upper cap on the soft COM vertical acceleration target during loaded support.')
    parser.add_argument('--maximum-support-acceleration-start',type=float,default=float('inf'))
    parser.add_argument('--maximum-support-acceleration-ramp',type=float,default=.06)
    parser.add_argument('--com-upward-speed-limit',type=float,default=None,
                        help='Optional soft COM upward-speed feedback ceiling, not a hard physical constraint.')
    parser.add_argument('--com-upward-speed-start',type=float,default=float('inf'),help='Source seconds.')
    parser.add_argument('--com-upward-speed-ramp',type=float,default=.06)
    parser.add_argument('--com-upward-speed-end',type=float,default=float('inf'),
                        help='Source seconds to begin fading out the soft upward-speed ceiling.')
    parser.add_argument('--com-upward-speed-end-ramp',type=float,default=.06)
    parser.add_argument('--hip-yaw-barrier-start', type=float, default=float('inf'),
                        help='Source seconds to add the observed cross-hip mesh collision constraint.')
    parser.add_argument('--hip-yaw-barrier-bandwidth', type=float, default=8.)
    parser.add_argument('--hip-yaw-barrier-margin', type=float, default=.02)
    parser.add_argument('--com-upward-speed-bandwidth',type=float,default=10.)
    parser.add_argument('--com-upward-speed-minimum-acceleration',type=float,default=-6.)
    parser.add_argument('--weighted-tasks-start', type=float, default=float('inf'),
                        help='Source seconds to blend secondary costs into one constrained task QP')
    parser.add_argument('--weighted-tasks-ramp', type=float, default=.1)
    parser.add_argument('--support-flatten-start', type=float, default=float('inf'),
                        help='Source seconds to blend loaded planned stance feet toward a level sole')
    parser.add_argument('--root-position-task-start', type=float, default=float('inf'),
                        help='Source seconds to enable direct root-position acceleration tracking')
    parser.add_argument('--root-position-task-weight', type=float, default=1000.)
    parser.add_argument('--root-position-task-axes',choices=('y','xy','xyz'),default='xyz',
                        help='World axes for the optional soft root position task.')
    parser.add_argument('--root-position-task-ramp', type=float, default=.1)
    parser.add_argument('--root-position-task-late-weight',type=float,default=None)
    parser.add_argument('--root-position-task-late-start',type=float,default=float('inf'),help='Source seconds for a late cost transition in the same root task.')
    parser.add_argument('--root-position-task-late-ramp',type=float,default=.1)
    parser.add_argument('--root-position-task-late-axes',choices=('y','xy','xyz'),default='xyz')
    parser.add_argument('--root-position-task-reference', choices=('source', 'control'), default='source',
                        help='Use the original audit source or the unified control q/v/a for the soft root task.')
    parser.add_argument('--stance-posture-metric-start', type=float, default=float('inf'),
                        help='Source time to begin reducing only active stance-leg posture weights')
    parser.add_argument('--stance-posture-metric-scale', type=float, default=.5)
    parser.add_argument('--stance-posture-metric-ramp', type=float, default=.1)
    parser.add_argument('--stance-posture-secondary-start',type=float,default=100.)
    parser.add_argument('--leg-posture-secondary-start',type=float,default=float('inf'))
    parser.add_argument('--late-reference-smooth-stride',type=int,default=0)
    parser.add_argument('--late-reference-smooth-start',type=float,default=float('inf'))
    parser.add_argument('--late-reference-smooth-ramp',type=float,default=.15)
    parser.add_argument('--configuration-tracking-start',type=float,default=float('inf'))
    parser.add_argument('--configuration-keep-feet',action='store_true')
    parser.add_argument('--configuration-angular-weight',type=float,default=1.)
    parser.add_argument('--configuration-tracking-ramp',type=float,default=.1)
    parser.add_argument('--native-contact-plan-start',type=float,default=100.)
    parser.add_argument('--torque-continuity-start', type=float, default=float('inf'),
                        help='Source seconds to enable a soft previous-torque cost in every QP stage.')
    parser.add_argument('--torque-continuity-weight', type=float, default=10.)
    parser.add_argument('--torque-continuity-ramp', type=float, default=.1)
    parser.add_argument('--torque-continuity-end', type=float, default=float('inf'),
                        help='Source seconds to begin reducing the optional continuity cost.')
    parser.add_argument('--torque-continuity-end-ramp', type=float, default=.06)
    parser.add_argument('--torque-continuity-end-weight', type=float, default=0.)
    parser.add_argument('--torque-continuity-stance-only', action='store_true',
                        help='Apply optional effort continuity only to the six motors of each QP support leg.')
    parser.add_argument('--support-flatten-gain-scale',type=float,default=1.)
    parser.add_argument('--support-flatten-feedback-limit',type=float,default=25.)
    parser.add_argument('--stance-heading-start', type=float, default=float('inf'),
                        help='Source seconds to enable measured stance heading reference adaptation.')
    parser.add_argument('--stance-heading-limit', type=float, default=.4)
    parser.add_argument('--stance-heading-ramp', type=float, default=.15)
    parser.add_argument('--measured-stance-reference-start',type=float,default=100.)
    parser.add_argument('--measured-stance-reference-limit',type=float,default=.03)
    parser.add_argument('--measured-stance-reference-ramp',type=float,default=.10)
    parser.add_argument('--synchronous-effort-budget-start',type=float,default=100.)
    parser.add_argument('--synchronous-effort-budget-fraction',type=float,default=1.)
    parser.add_argument('--joint-limit-priority-start',type=float,default=100.)
    parser.add_argument('--joint-limit-priority-upper-only',action='store_true')
    parser.add_argument('--joint-limit-priority-scale', type=float, default=1.,
                        help='Scale the selected soft braking rows only after joint-limit-priority-start; earlier tracking keeps its original weight.')
    parser.add_argument('--late-momentum-priority-start',type=float,default=100.)
    parser.add_argument('--late-momentum-roll-only', action='store_true')
    parser.add_argument('--late-momentum-zero-target',action='store_true',help='Regulate angular momentum instead of following source momentum; pose references and audit stay unchanged.')
    parser.add_argument('--late-momentum-feedback-damping', type=float, default=None)
    parser.add_argument('--late-momentum-total-rate-limit', type=float, default=None)
    parser.add_argument('--late-momentum-metric-scale',type=float,default=1.)
    parser.add_argument('--retain-planned-support-start',type=float,default=100.)
    parser.add_argument('--retain-planned-support-force',type=float,default=70.)
    parser.add_argument('--retain-planned-support-weight',type=float,default=.2)
    parser.add_argument('--qp-snapshot-times',type=float,nargs='*',default=[])
    parser.add_argument('--late-balance-hierarchy-start', type=float, default=100.)
    parser.add_argument('--late-balance-hierarchy', choices=['y','xy','xyz','xy-body','xyz-body','body'], default='xy-body')
    parser.add_argument('--late-pelvis-priority-start', type=float, default=100.)
    parser.add_argument('--pelvis-orientation-replace-start',type=float,default=float('inf'))
    parser.add_argument('--pelvis-orientation-replace-ramp',type=float,default=.1)
    parser.add_argument('--pelvis-orientation-replace-metric-scale',type=float,default=1.)
    parser.add_argument('--pelvis-orientation-replace-feedback-limit',type=float,default=None,
                        help='Optional angular feedback norm cap during torso-to-pelvis allocation; preserves source feedforward.')
    parser.add_argument('--late-pelvis-metric-scale',type=float,default=1.,help='Scale only the promoted pelvis task after its start, preserving the earlier controller.')
    parser.add_argument('--late-pelvis-axes',nargs='+',choices=['x','y','z'],default=None,
                        help='World angular axes for the optional promoted pelvis task; default keeps all axes.')
    parser.add_argument('--late-pelvis-yaw-only',action='store_true')
    parser.add_argument('--contact-friction-rays',type=int,choices=[8,16],default=8)
    parser.add_argument('--contact-friction-refine-start',type=float,default=float('inf'))
    parser.add_argument('--support-orientation-priority-start', type=float, default=100.)
    parser.add_argument('--support-orientation-metric-scale', type=float, default=1.)
    parser.add_argument('--support-orientation-roll-pitch-only',action='store_true',help='Promote only the free support-foot tilt axes, with a 60 ms smooth onset; contact constraints and source q/v/a are unchanged.')
    parser.add_argument('--upper-posture-metric-start', type=float, default=100.)
    parser.add_argument('--upper-posture-metric-scale', type=float, default=1.)
    parser.add_argument('--late-upper-metric-start', type=float, default=float('inf'))
    parser.add_argument('--late-upper-metric-scale', type=float, default=None,
                        help='Optional replacement of the existing upper-posture multiplier, in source seconds.')
    parser.add_argument('--late-upper-metric-ramp', type=float, default=.06)
    parser.add_argument('--leg-posture-metric-start', type=float, default=float('inf'),
                        help='Source seconds to smoothly change leg posture cost in the same QP.')
    parser.add_argument('--leg-posture-metric-scale', type=float, default=1.)
    parser.add_argument('--transition-swing-priority-start', type=float, default=100.)
    parser.add_argument('--posture-total-accel-start', type=float, default=100.)
    parser.add_argument('--posture-total-accel-limit', type=float, default=150.)
    parser.add_argument('--body-feedback-start', type=float, default=100.)
    parser.add_argument('--body-feedback-limit', type=float, default=75.)
    parser.add_argument('--late-body-angular-metric-start', type=float, default=100.)
    parser.add_argument('--late-body-angular-metric-scale', type=float, default=1.)
    parser.add_argument('--late-foot-angular-metric-start', type=float, default=100.)
    parser.add_argument('--late-foot-angular-metric-scale', type=float, default=1.)
    parser.add_argument('--late-foot-angular-metric-end',type=float,default=float('inf'))
    parser.add_argument('--late-foot-angular-metric-side',choices=['both','left','right'],default='both')
    parser.add_argument('--late-foot-angular-feedback-limit', type=float, default=None,
                        help='Optional angular feedback norm cap during the existing foot angular weight window; reference acceleration is preserved.')
    parser.add_argument('--late-foot-metric-start', type=float, default=100.)
    parser.add_argument('--late-foot-metric-scale', type=float, default=1.)
    parser.add_argument('--late-foot-metric-end', type=float, default=float('inf'),
                        help='Optional source time to smoothly finish the elevated foot task weight.')
    parser.add_argument('--late-foot-metric-end-ramp', type=float, default=.04)
    parser.add_argument('--late-foot-metric-end-scale', type=float, default=1.)
    parser.add_argument('--centroidal-preview-start', type=float, default=100.)
    parser.add_argument('--centroidal-preview-walking', action='store_true')
    parser.add_argument('--centroidal-preview-persist-stance', action='store_true')
    parser.add_argument('--centroidal-preview-momentum', action='store_true')
    parser.add_argument('--global-posture-feedback-limit', type=float, default=150.)
    parser.add_argument('--recover-orientation-tube', action='store_true')
    parser.add_argument('--orientation-tube-end',type=float,default=float('inf'),help='Source time to end only the auxiliary hard orientation tube; original-source orientation audit stays active.')
    parser.add_argument('--position-tube-end',type=float,default=float('inf'),help='Source time to end only the auxiliary hard root-position tube; the original-source position audit stays active.')
    parser.add_argument('--relax-joint-tube', action='store_true')
    parser.add_argument('--tube-recovery-metric', type=float, default=1000.)
    parser.add_argument('--horizontal-feedback-start', type=float, default=100.)
    parser.add_argument('--horizontal-feedback-limit', type=float, default=40.)
    parser.add_argument('--horizontal-feedback-metric-scale', type=float, default=2.)
    parser.add_argument('--later-clearance-target', type=float, default=.085)
    parser.add_argument('--later-clearance-fall-fraction', type=float, default=.15)
    parser.add_argument('--later-clearance-first-step', type=int, default=8)
    parser.add_argument('--smooth-stride', type=int, default=0)
    parser.add_argument('--smooth-start', type=float, default=2.82)
    parser.add_argument('--smooth-ramp', type=float, default=.08)
    parser.add_argument('--tube-smooth-stride', type=int, default=0)
    parser.add_argument('--late-tube-smooth-stride',type=int,default=0)
    parser.add_argument('--late-tube-smooth-start',type=float,default=float('inf'))
    parser.add_argument('--late-tube-smooth-ramp',type=float,default=.1)
    parser.add_argument('--reference-correction-rmse-limit', type=float, default=0.)
    parser.add_argument('--orientation-tube-radius', type=float, default=.34,
                        help='Auxiliary root-angle error radius in radians; raw PKL acceptance is unchanged')
    parser.add_argument('--orientation-tube-start', type=float, default=float('inf'),
                        help='Source seconds to use the auxiliary root-angle radius')
    parser.add_argument('--position-tube-radius', type=float, default=.195)
    parser.add_argument('--position-tube-start', type=float, default=3.28)
    parser.add_argument('--first-correction-rmse-limit', type=float, default=None)
    parser.add_argument('--late-unload-start', type=float, default=100.)
    parser.add_argument('--late-unload-ramp', type=float, default=.01)
    parser.add_argument('--late-unload-weight', type=float, default=.05)
    parser.add_argument('--unload-deadline-start', type=float, default=float('inf'),
                        help='Source seconds to fit measured unloading inside the next swing boundary; contact and audit gates stay unchanged')
    parser.add_argument('--unload-deadline-weight', type=float, default=None,
                        help='Optional soft outgoing-wrench weight after unload-deadline-start')
    parser.add_argument('--full-swing-pose-start', type=float, default=100.)
    parser.add_argument('--reference-pose-candidate', type=Path, default=None)
    parser.add_argument('--prepared-contact-plan', type=Path, default=None)
    parser.add_argument('--joint-correction-overlay', type=Path, default=None)
    parser.add_argument('--secondary-joint-correction-overlay', type=Path, default=None)
    parser.add_argument('--swing-balance-priority-start', type=float, default=100.)
    parser.add_argument('--swing-only-priority', action='store_true')
    parser.add_argument('--landing-gated-unload-start', type=float, default=100.)
    parser.add_argument('--anticipate-unload-hysteresis', action='store_true')
    parser.add_argument('--anticipate-unload-overdue', action='store_true')
    parser.add_argument('--anticipate-unload-start', type=float, default=100.)
    parser.add_argument('--anticipate-unload-horizon', type=float, default=.08)
    parser.add_argument('--gate-contact-transitions', action='store_true')
    parser.add_argument('--anticipate-unload-error-limit', type=float, default=.03,
                        help='Measured landing error used for early load transfer; source landing audit is unchanged.')
    parser.add_argument('--late-contact-stabilization-start', type=float, default=100.)
    parser.add_argument('--late-contact-stabilization-kp', type=float, default=20.)
    parser.add_argument('--late-contact-stabilization-kd', type=float, default=8.)
    parser.add_argument('--tube-reference-method', choices=['pchip','bspline'], default='pchip')
    parser.add_argument('--fabric-priority-mode', choices=['none', 'paper', 'paper-tracking', 'contact-tracking'], default='none',
                        help='Paper Eq.8 stability nullspace in Fabric; retains one inverse-dynamics QP.')
    parser.add_argument('--fabric-tracking-feedback', action='store_true',
                        help='Compose existing physical tracking feedback inside Fabric before the single QP.')
    parser.add_argument('--fabric-secondary-scale', type=float, default=1.,
                        help='Optional soft weighting of historical secondary tasks during local feedback composition.')
    parser.add_argument('--fabric-root-regularization', type=float, default=None)
    parser.add_argument('--qp-task-only-objective', action='store_true',
                        help='Use task objective plus numerical regularization; retain scheduled unloading force target.')
    parser.add_argument('--qp-single-proximal', type=float, default=0.,
                        help='Preselect a proximal factor for the sole native QP attempt; keep original KKT checks.')
    parser.add_argument('--qp-single-proximal-start', type=float, default=0.)
    parser.add_argument('--qp-cold-start-time', type=float, default=float('inf'),
                        help='Control time to clear QP warm starts before each sole solve; no retries.')
    parser.add_argument('--qp-equilibration-start', type=float, default=float('inf'),
                        help='Control time to enable equivalent QP variable scaling before the sole solve.')
    parser.add_argument('--qp-equilibration-cap', type=float, default=1.,
                        help='Maximum variable scaling; original physical and KKT checks are retained.')
    parser.add_argument('--qp-late-equilibration-start', type=float, default=float('inf'),
                        help='Optional second numerical scaling transition in control seconds; chosen before the sole QP attempt.')
    parser.add_argument('--qp-late-equilibration-cap', type=float, default=1.)
    parser.add_argument('--qp-final-equilibration-start', type=float, default=float('inf'),
                        help='Control seconds to select a subsequent variable scaling before the sole QP solve.')
    parser.add_argument('--qp-final-equilibration-cap', type=float, default=1.)
    parser.add_argument('--qp-late-iteration-start', type=float, default=float('inf'),
                        help='Control seconds at which to change the single native QP iteration budget; no retry.')
    parser.add_argument('--qp-late-iterations', type=int, default=5000,
                        help='Single-attempt iteration budget after qp-late-iteration-start; physical/KKT checks stay unchanged.')
    parser.add_argument('--tracking-tube-reference', choices=['source', 'control'], default='source',
                        help='Optional tube reference alignment; original PKL auditing is unchanged.')
    parser.add_argument('--tracking-tube-soft-weight', type=float, default=0.,
                        help='Positive: penalize normalized tube slack in the sole QP; zero retains hard tubes.')
    parser.add_argument('--tracking-tube-soft-start', type=float, default=0.,
                        help='Simulation time to enable auxiliary tube slack; preserve hard tubes before this time.')
    parser.add_argument('--tracking-tube-soft-joint-only', action='store_true')
    parser.add_argument('--late-tube-soft-weight-start',type=float,default=float('inf'),
                        help='Source seconds to start a 60 ms blend of the existing tube slack cost.')
    parser.add_argument('--late-tube-soft-weight',type=float,default=None,
                        help='Optional late slack cost; physical constraints and source acceptance stay unchanged.')
    parser.add_argument('--tracking-tube-soft-joint-orientation', action='store_true',
                        help='Soften joint and orientation auxiliary tube rows while retaining the hard root-position row; source auditing is unchanged.')
    parser.add_argument('--soft-joint-braking-start', type=float, default=float('inf'),
                        help='Source-clock seconds to enable one-sided soft joint-stop braking in the same QP.')
    parser.add_argument('--soft-joint-braking-weight', type=float, default=100.)
    parser.add_argument('--soft-joint-braking-joints', nargs='+', choices=G1_JOINT_NAMES,
                        default=['waist_roll_joint'])
    parser.add_argument('--tracking-tube-orientation-bandwidth', type=float, default=None)
    parser.add_argument('--tracking-tube-joint-bandwidth', type=float, default=None)
    parser.add_argument('--late-joint-tube-start', type=float, default=100.)
    parser.add_argument('--late-joint-tube-bandwidth', type=float, default=4.)
    parser.add_argument('--tracking-margin-start', type=float, default=float('inf'))
    parser.add_argument('--tracking-margin-reaction-time', type=float, default=.08)
    parser.add_argument('--tracking-margin-maximum', type=float, default=16.)
    parser.add_argument('--tracking-tube-start', type=float, default=1.34)
    parser.add_argument('--tracking-tube-end', type=float, default=100.)
    parser.add_argument('--joint-tube-end', type=float, default=float('inf'),
                        help='Source time to disable only the auxiliary joint tracking barrier; original-source acceptance stays active.')
    parser.add_argument('--contact-pause-start', type=float, default=4.2)
    parser.add_argument('--contact-pause-end', type=float, default=None)
    parser.add_argument('--contact-pause-landings', type=int, default=11)
    parser.add_argument('--contact-pause-resume', type=float, default=.3)
    parser.add_argument('--contact-pause-early-resume', action='store_true')
    parser.add_argument('--extra-contact-pause', type=float, nargs=3, action='append', default=[],
                        metavar=('START', 'END', 'LANDINGS'),
                        help='Additional causal source-time landing pause; repeat for subsequent landings.')
    parser.add_argument('--extra-contact-pause-resume', type=float, default=.1)
    parser.add_argument('--late-tracking-tube-start', type=float, default=100.)
    parser.add_argument('--late-tracking-tube-bandwidth', type=float, default=10.)
    parser.add_argument('--late-position-tube-start',type=float,default=float('inf'),
                        help='Source time to change only the auxiliary root-position barrier bandwidth.')
    parser.add_argument('--late-position-tube-bandwidth',type=float,default=10.)
    parser.add_argument('--root-xy-tube-start',type=float,default=float('inf'),
                        help='Optional source-clock start of an additional horizontal root tracking barrier.')
    parser.add_argument('--root-xy-tube-radius',type=float,default=.18)
    parser.add_argument('--root-xy-tube-bandwidth',type=float,default=10.)
    parser.add_argument('--support-knee-min-angle', type=float, default=None,
                        help='Optional support-knee flexion barrier; does not change physical limits or source fidelity checks.')
    parser.add_argument('--support-knee-guard-start', type=float, default=float('inf'),
                        help='Source time to activate the additional support-knee acceleration constraint.')
    parser.add_argument('--support-knee-guard-bandwidth', type=float, default=20.)
    parser.add_argument('--physical-braking-start', type=float, default=100.)
    parser.add_argument('--convex-wrist-collision-start', type=float, default=0.)
    parser.add_argument('--collision-barrier-bandwidth-start', type=float, default=float('inf'))
    parser.add_argument('--collision-barrier-bandwidth', type=float, default=8.)
    parser.add_argument('--physical-braking-upper-only', action='store_true')
    parser.add_argument('--physical-braking-waist-only', action='store_true',
                        help='Apply optional measured braking bounds only to the three waist joints.')
    parser.add_argument('--physical-braking-joints', nargs='+', choices=G1_JOINT_NAMES,
                        help='Apply optional measured braking bounds to these named joints only; mutually exclusive with other braking selectors.')
    parser.add_argument('--observed-joint-stop-start', type=float, default=float('inf'),
                        help='Source seconds to model measured stationary mechanical joint stops inside the same inverse-dynamics QP; applies no extra effort.')
    parser.add_argument('--observed-joint-stop-joints', nargs='+', choices=G1_JOINT_NAMES,
                        default=['waist_roll_joint'])
    parser.add_argument('--observed-joint-stop-weight', type=float, default=1e-4,
                        help='Soft passive-reaction cost to encourage unloading an observed joint stop.')
    parser.add_argument('--tracking-tube-bandwidth', type=float, default=10.)
    parser.add_argument('--late-leg-feedback-start', type=float, default=1.88)
    parser.add_argument('--late-leg-feedback-limit', type=float, default=150.)
    parser.add_argument('--late-leg-feedback-stance', action='store_true')
    parser.add_argument('--late-leg-feedback-side', choices=['right','both'], default='right')
    parser.add_argument('--waist-feedback-start', type=float, default=1.7)
    parser.add_argument('--waist-feedback-limit', type=float, default=100.)
    parser.add_argument('--waist-metric-scale', type=float, default=4.)
    parser.add_argument('--late-waist-metric-scale', type=float, default=None,
                        help='Optional replacement waist metric multiplier after a source-time transition.')
    parser.add_argument('--late-waist-metric-start', type=float, default=float('inf'))
    parser.add_argument('--late-waist-metric-ramp', type=float, default=.06)
    parser.add_argument('--flight-feedback-joints', choices=['ankles','leg'], default='ankles')
    parser.add_argument('--flight-feedback-start', type=float, default=0.)
    parser.add_argument('--flight-feedback-ramp', type=float, default=.1)
    return configure_full_tracking_profile(parser)


def main() -> None:
    global FABRIC_TIMESTEP, PD_TIMESTEP
    args = parse_args()
    acceleration_only = args.tracking_profile == 'amp-full'
    recompose_tasks = args.fabric_tracking_feedback or args.fabric_priority_mode == 'contact-tracking'
    task_feedback_enabled = not acceleration_only or args.fabric_tracking_feedback
    if math.isnan(args.unload_deadline_start) or args.unload_deadline_start < 0:
        raise ValueError('Unload deadline start must be nonnegative source seconds')
    if args.unload_deadline_weight is not None and (
            not math.isfinite(args.unload_deadline_weight) or args.unload_deadline_weight <= 0):
        raise ValueError('Unload deadline force weight must be finite and positive')
    if not math.isfinite(args.anticipate_unload_error_limit) or not 0 < args.anticipate_unload_error_limit <= .04:
        raise ValueError('Anticipatory transfer error must be positive and at most the 4 cm landing tolerance')
    if not math.isfinite(args.fabric_secondary_scale) or not 0 <= args.fabric_secondary_scale <= 1:
        raise ValueError('Fabric secondary scale must lie in [0,1]')
    if args.fabric_secondary_scale != 1 and not args.fabric_tracking_feedback:
        raise ValueError('Fabric secondary scaling requires local tracking feedback')
    if args.qp_task_only_objective and not args.fabric_tracking_feedback:
        raise ValueError('Task-only objective requires local Fabric tracking feedback')
    if any(not math.isfinite(x) or x < 0 for x in (args.qp_single_proximal, args.qp_single_proximal_start)):
        raise ValueError('Single-QP proximal factor and start must be finite and nonnegative')
    if args.qp_single_proximal and not acceleration_only:
        raise ValueError('Single-QP proximal selection requires amp-full')
    if math.isnan(args.qp_cold_start_time) or args.qp_cold_start_time < 0:
        raise ValueError('QP cold-start time must be nonnegative')
    if math.isfinite(args.qp_cold_start_time) and not acceleration_only:
        raise ValueError('Scheduled QP cold starts require amp-full')
    if (math.isnan(args.qp_equilibration_start) or args.qp_equilibration_start < 0
            or not math.isfinite(args.qp_equilibration_cap) or args.qp_equilibration_cap < 1):
        raise ValueError('QP equilibration requires a nonnegative start and finite cap >= 1')
    if math.isfinite(args.qp_equilibration_start) and not acceleration_only:
        raise ValueError('Scheduled QP equilibration requires amp-full')
    if (math.isnan(args.qp_late_equilibration_start) or args.qp_late_equilibration_start < 0
            or not math.isfinite(args.qp_late_equilibration_cap) or args.qp_late_equilibration_cap < 1):
        raise ValueError('Late QP equilibration requires a nonnegative start and finite cap >= 1')
    if math.isfinite(args.qp_late_equilibration_start) and not acceleration_only:
        raise ValueError('Late QP equilibration requires amp-full')
    if (math.isnan(args.qp_final_equilibration_start) or args.qp_final_equilibration_start < 0
            or not math.isfinite(args.qp_final_equilibration_cap) or args.qp_final_equilibration_cap < 1
            or (math.isfinite(args.qp_final_equilibration_start) and not acceleration_only)):
        raise ValueError('Final QP scaling requires amp-full, a nonnegative start and finite cap >= 1')
    if (math.isnan(args.qp_late_iteration_start) or args.qp_late_iteration_start < 0
            or args.qp_late_iterations <= 0):
        raise ValueError('Late QP iteration budget requires a nonnegative start and positive count')
    if math.isfinite(args.qp_late_iteration_start) and not acceleration_only:
        raise ValueError('Late single-QP iteration budget requires amp-full')
    if args.extra_contact_pause:
        if not acceleration_only:
            raise ValueError('Additional landing clocks require amp-full')
        from g1_landing_clock_chain import LandingClockChain
        LandingClockChain(args.extra_contact_pause, args.extra_contact_pause_resume)
        if args.contact_pause_end is not None and args.extra_contact_pause[0][0] < args.contact_pause_end:
            raise ValueError('Additional landing clocks must follow the initial pause in source time')
    if args.fabric_root_regularization is not None and (
            not math.isfinite(args.fabric_root_regularization) or args.fabric_root_regularization <= 0):
        raise ValueError('Fabric root regularization must be finite and positive')
    if (math.isnan(args.late_waist_metric_start) or args.late_waist_metric_start < 0
            or not math.isfinite(args.late_waist_metric_ramp) or args.late_waist_metric_ramp <= 0
            or (args.late_waist_metric_scale is not None and
                (not math.isfinite(args.late_waist_metric_scale) or args.late_waist_metric_scale <= 0))):
        raise ValueError('Late waist metric needs a positive scale/ramp and nonnegative source start')
    if args.fabric_tracking_feedback and args.fabric_priority_mode == 'paper':
        raise ValueError('Physical tracking feedback requires none, paper-tracking or contact-tracking')
    if args.fabric_priority_mode != 'none' or recompose_tasks:
        if not acceleration_only:
            raise ValueError('Paper Fabric priority currently requires amp-full single-QP control')
        # Preserve the prioritized root objective in the physical QP instead
        # of replacing its metric by fixed generalized-acceleration weights.
        args.qp_fabric_metric = True
    if not math.isfinite(args.tracking_tube_soft_weight) or args.tracking_tube_soft_weight < 0:
        raise ValueError('tracking_tube_soft_weight must be finite and nonnegative')
    if args.late_tube_soft_weight is not None and (
            not args.tracking_tube_soft_weight or not math.isfinite(args.late_tube_soft_weight)
            or args.late_tube_soft_weight<=0 or math.isnan(args.late_tube_soft_weight_start)
            or args.late_tube_soft_weight_start<0):
        raise ValueError('Late tube cost needs an enabled soft tube, positive finite weight and nonnegative start')
    if not math.isfinite(args.tracking_tube_soft_start) or args.tracking_tube_soft_start < 0:
        raise ValueError('Tube soft start must be finite and nonnegative')
    if args.tracking_tube_soft_joint_only and not args.tracking_tube_soft_weight:
        raise ValueError('Joint-only tube slack requires positive soft weight')
    if args.tracking_tube_soft_joint_orientation and (
            not args.tracking_tube_soft_weight or args.tracking_tube_soft_joint_only):
        raise ValueError('Joint/orientation tube slack requires positive weight and cannot combine with joint-only slack')
    if (math.isnan(args.root_xy_tube_start) or args.root_xy_tube_start<0
            or not math.isfinite(args.root_xy_tube_radius) or not 0<args.root_xy_tube_radius<=.2
            or not math.isfinite(args.root_xy_tube_bandwidth) or args.root_xy_tube_bandwidth<=0):
        raise ValueError('Invalid root XY tracking barrier configuration')
    if args.late_foot_linear_feedback_limit is not None and (
            not math.isfinite(args.late_foot_linear_feedback_limit) or args.late_foot_linear_feedback_limit<=0
            or math.isnan(args.late_foot_linear_feedback_start) or args.late_foot_linear_feedback_start<0):
        raise ValueError('Foot linear feedback requires a positive finite limit and nonnegative start')
    if args.late_foot_angular_feedback_limit is not None and (
            not math.isfinite(args.late_foot_angular_feedback_limit) or args.late_foot_angular_feedback_limit <= 0):
        raise ValueError('Foot angular feedback limit must be finite and positive')
    if (not math.isfinite(args.pelvis_orientation_replace_metric_scale)
            or args.pelvis_orientation_replace_metric_scale <= 0):
        raise ValueError('Pelvis replacement metric scale must be finite and positive')
    if args.pelvis_orientation_replace_feedback_limit is not None and (
            not math.isfinite(args.pelvis_orientation_replace_feedback_limit)
            or args.pelvis_orientation_replace_feedback_limit <= 0):
        raise ValueError('Pelvis replacement feedback limit must be finite and positive')
    if (math.isnan(args.soft_joint_braking_start) or args.soft_joint_braking_start < 0
            or not math.isfinite(args.soft_joint_braking_weight) or args.soft_joint_braking_weight <= 0
            or len(set(args.soft_joint_braking_joints)) != len(args.soft_joint_braking_joints)):
        raise ValueError('Invalid soft joint braking parameters')
    if math.isfinite(args.soft_joint_braking_start) and (
            not acceleration_only or not args.tracking_tube_soft_weight
            or not args.tracking_tube_start <= args.soft_joint_braking_start < args.tracking_tube_end):
        raise ValueError('Soft joint braking requires amp-full and an active soft tracking tube')
    if (args.tracking_tube_soft_weight or args.tracking_tube_reference != 'source') and not acceleration_only:
        raise ValueError('Tube experiments require the amp-full direct single-QP profile')
    for name in ('projection_joint_correction_weight', 'com_acceleration_preservation_weight', 'qp_com_feedback_scale'):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f'{name} must be finite and nonnegative')
    if args.maximum_support_acceleration is not None:
        if (not math.isfinite(args.maximum_support_acceleration) or args.maximum_support_acceleration < 0
                or not math.isfinite(args.maximum_support_acceleration_start) or args.maximum_support_acceleration_start < 0
                or not math.isfinite(args.maximum_support_acceleration_ramp) or args.maximum_support_acceleration_ramp <= 0
                or not acceleration_only):
            raise ValueError('COM upward cap requires amp-full, a nonnegative finite cap/start, and a positive finite ramp')
    if args.com_upward_speed_limit is not None:
        if (not acceleration_only or not math.isfinite(args.com_upward_speed_limit) or args.com_upward_speed_limit < 0
                or not math.isfinite(args.com_upward_speed_start) or args.com_upward_speed_start < 0
                or not math.isfinite(args.com_upward_speed_ramp) or args.com_upward_speed_ramp <= 0
                or math.isnan(args.com_upward_speed_end) or args.com_upward_speed_end < args.com_upward_speed_start
                or not math.isfinite(args.com_upward_speed_end_ramp) or args.com_upward_speed_end_ramp <= 0
                or not math.isfinite(args.com_upward_speed_bandwidth) or args.com_upward_speed_bandwidth <= 0
                or not math.isfinite(args.com_upward_speed_minimum_acceleration)
                or not -9.81 < args.com_upward_speed_minimum_acceleration <= 0):
            raise ValueError('Invalid COM upward-speed policy; requires amp-full and finite valid parameters')
    if (math.isnan(args.hip_yaw_barrier_start) or args.hip_yaw_barrier_start < 0
            or not math.isfinite(args.hip_yaw_barrier_bandwidth) or args.hip_yaw_barrier_bandwidth <= 0
            or not math.isfinite(args.hip_yaw_barrier_margin) or args.hip_yaw_barrier_margin < 0
            or (math.isfinite(args.hip_yaw_barrier_start) and not acceleration_only)):
        raise ValueError('Hip barrier requires amp-full, a nonnegative start/margin and positive bandwidth')
    if (math.isnan(args.late_support_load_floor_start) or args.late_support_load_floor_start < 0
            or not math.isfinite(args.late_support_load_floor_ramp) or args.late_support_load_floor_ramp <= 0
            or (args.late_support_load_floor is not None and
                (not math.isfinite(args.late_support_load_floor) or args.late_support_load_floor < 0))):
        raise ValueError('Late support floor requires nonnegative start/load and a positive finite ramp')
    if (args.qp_com_feedback_scale or args.qp_fabric_metric) and not acceleration_only:
        raise ValueError('Fabric/COM feedback objectives require the amp-full single-QP profile')
    com_feedback_scale(args.qp_com_feedback_scale, 0., start=args.qp_com_feedback_start,
                       ramp=args.qp_com_feedback_ramp)
    if args.qp_fabric_metric and (args.com_acceleration_preservation_weight or args.qp_base_linear_acceleration_weight is not None):
        raise ValueError('The complete Fabric metric replaces projected/base acceleration weights')
    if math.isnan(args.joint_tube_end) or args.joint_tube_end < 0:
        raise ValueError('joint_tube_end must be nonnegative')
    from g1_shoulder_roll import shoulder_roll_multiplier, apply_shoulder_roll_weights
    shoulder_roll_multiplier(0., args.shoulder_roll_weight_start,
                             args.shoulder_roll_weight_ramp, args.shoulder_roll_weight_scale)
    if args.shoulder_roll_weight_scale > 1 and (
            not math.isfinite(args.shoulder_roll_weight_start)
            or args.tracking_profile != 'amp-full' or not args.direct_wbc):
        raise ValueError('Shoulder weight experiment requires amp-full direct WBC and a finite start')
    shoulder_guard = None
    if args.shoulder_roll_guard:
        from g1_guarded_shoulder import GuardedShoulderWeights
        if args.shoulder_roll_weight_scale <= 1 or math.isfinite(args.weighted_tasks_start):
            raise ValueError('Guard requires a shoulder scale > 1 and the original priority stages')
        shoulder_guard = GuardedShoulderWeights(args.shoulder_roll_guard_relative, args.shoulder_roll_guard_slack)
    if not math.isfinite(args.joint_limit_priority_scale) or args.joint_limit_priority_scale <= 0:
        raise ValueError('Joint-limit priority scale must be finite and positive')
    if (math.isnan(args.leg_posture_metric_start) or args.leg_posture_metric_start < 0
            or not math.isfinite(args.leg_posture_metric_scale) or args.leg_posture_metric_scale <= 0):
        raise ValueError('Leg posture cost needs a nonnegative start and positive finite scale')
    if (math.isnan(args.late_upper_metric_start) or args.late_upper_metric_start < 0
            or not math.isfinite(args.late_upper_metric_ramp) or args.late_upper_metric_ramp <= 0
            or (args.late_upper_metric_scale is not None and
                (not math.isfinite(args.late_upper_metric_scale) or args.late_upper_metric_scale <= 0))):
        raise ValueError('Late upper-posture weight needs a nonnegative start and positive finite scale/ramp')
    if args.support_knee_min_angle is not None:
        if (not math.isfinite(args.support_knee_min_angle) or not 0 < args.support_knee_min_angle < 1
                or not math.isfinite(args.support_knee_guard_bandwidth) or args.support_knee_guard_bandwidth <= 0
                or math.isnan(args.support_knee_guard_start) or args.support_knee_guard_start < 0):
            raise ValueError('Invalid optional support-knee guard parameters')
    if args.physical_braking_waist_only and args.physical_braking_upper_only:
        raise ValueError('Choose waist-only or upper-body optional braking bounds')
    if args.physical_braking_joints is not None:
        if args.physical_braking_waist_only or args.physical_braking_upper_only or args.acceleration_and_braking_limits:
            raise ValueError('Named braking joints cannot be combined with another braking selector')
        from g1_joint_braking import named_braking_coordinates
        named_braking_coordinates(G1_JOINT_NAMES, args.physical_braking_joints)
    if math.isnan(args.observed_joint_stop_start) or args.observed_joint_stop_start < 0:
        raise ValueError('Observed joint-stop start must be nonnegative')
    if not math.isfinite(args.observed_joint_stop_weight) or args.observed_joint_stop_weight <= 0:
        raise ValueError('Observed joint-stop weight must be finite and positive')
    if math.isfinite(args.observed_joint_stop_start):
        if args.tracking_profile != 'amp-full' or not args.direct_wbc or not args.tracking_servo:
            raise ValueError('Observed joint stops require amp-full direct-effort tracking')
        if len(set(args.observed_joint_stop_joints)) != len(args.observed_joint_stop_joints):
            raise ValueError('Observed joint-stop names must be distinct')
    if (math.isnan(args.torque_continuity_start) or args.torque_continuity_start < 0
            or not math.isfinite(args.torque_continuity_weight) or args.torque_continuity_weight < 0
            or not math.isfinite(args.torque_continuity_ramp) or args.torque_continuity_ramp <= 0):
        raise ValueError('Torque continuity needs a nonnegative start/weight and a positive ramp')
    if (math.isnan(args.torque_continuity_end) or args.torque_continuity_end < args.torque_continuity_start
            or not math.isfinite(args.torque_continuity_end_ramp) or args.torque_continuity_end_ramp <= 0
            or not math.isfinite(args.torque_continuity_end_weight) or args.torque_continuity_end_weight < 0):
        raise ValueError('Torque continuity end needs an ordered start, positive ramp and nonnegative weight')
    if math.isfinite(args.torque_continuity_start) and (args.tracking_profile != 'amp-full'
            or not args.direct_wbc or not args.tracking_servo):
        raise ValueError('Torque continuity requires amp-full direct effort tracking')
    if args.torque_continuity_stance_only and not math.isfinite(args.torque_continuity_start):
        raise ValueError('Stance-only effort continuity requires a finite activation time')
    if (math.isnan(args.stance_heading_start) or args.stance_heading_start < 0
            or not math.isfinite(args.stance_heading_limit) or not 0 < args.stance_heading_limit <= .5
            or not math.isfinite(args.stance_heading_ramp) or args.stance_heading_ramp <= 0):
        raise ValueError('Stance heading requires a nonnegative start, limit in (0,.5], and positive ramp')
    if math.isfinite(args.stance_heading_start) and (
            args.tracking_profile != 'amp-full' or args.measured_stance_reference_start < float('inf')):
        raise ValueError('Stance heading requires amp-full and excludes simultaneous stance position adaptation')
    if math.isnan(args.late_physics_substeps_start) or args.late_physics_substeps_start < 0:
        raise ValueError('Late physics substeps require a nonnegative simulation start')
    if (args.late_physics_substeps > 1 and math.isfinite(args.late_physics_substeps_start)
            and (args.tracking_profile != 'amp-full' or not args.direct_wbc
                 or args.control_frequency != (args.physics_frequency or 500))):
        raise ValueError('Late substeps require synchronous amp-full direct WBC')
    if (not math.isfinite(args.orientation_tube_radius)
            or not 0 < args.orientation_tube_radius <= .34
            or math.isnan(args.orientation_tube_start) or args.orientation_tube_start < 0):
        raise ValueError('Orientation tube requires a positive radius at most .34 rad and nonnegative source start')
    if args.motion_start_in_motion:
        if (args.motion_pkl is None or args.reference_manager or args.mocap_pipeline
                or args.flat_foot_spawn or args.fixed_joint_test or args.weight_transfer_test
                or args.single_support_test or args.require_support_load
                or (not args.g1_md and args.task_profile != 'dynamics-aware')
                or args.reference_mode != 'unified'
                or args.motion_interpolation not in ('pchip-analytic', 'bspline-analytic')):
            raise ValueError('--motion-start-in-motion requires an analytic unified PKL reference and excludes stationary/managed starts')
        args.motion_warmup_duration = args.motion_ramp_duration = 0.
    if (not math.isfinite(args.mocap_unload_weight) or args.mocap_unload_weight < 0
            or (args.mocap_unload_weight and not args.mocap_pipeline)):
        raise ValueError('--mocap-unload-weight requires the mocap pipeline and a finite nonnegative weight')
    if args.track_full_motion and (not args.motion_start_in_motion or args.tracking_steps
                                   or args.motion_frame != 0 or args.until_motion_end):
        raise ValueError('--track-full-motion requires dynamic start at frame zero and excludes bounded steps or stationary completion')
    if (not math.isfinite(args.swing_midpoint_clearance) or args.swing_midpoint_clearance < 0
            or (args.swing_midpoint_clearance and not (args.motion_start_in_motion and (args.tracking_steps or args.track_full_motion)))):
        raise ValueError('--swing-midpoint-clearance requires dynamic bounded steps or full-motion tracking')
    if (not math.isfinite(args.dynamic_unload_weight) or args.dynamic_unload_weight < 0
            or (args.dynamic_unload_weight and not args.motion_start_in_motion)):
        raise ValueError('--dynamic-unload-weight requires dynamic start and a finite nonnegative weight')
    if args.tracking_steps < 0:
        raise ValueError('--tracking-steps must be nonnegative')
    if args.tracking_steps:
        if not (args.mocap_pipeline or args.motion_start_in_motion) or args.motion_frame != 0:
            raise ValueError('--tracking-steps requires mocap pipeline or dynamic start from frame zero')
        if not args.motion_start_in_motion:
            args.until_motion_end = True
    if args.mocap_first_step_project_torque and args.mocap_first_step_report is None:
        raise ValueError('--mocap-first-step-project-torque requires a bounded first-step report')
    if args.mocap_first_step_report is not None:
        if not args.mocap_pipeline or args.acceleration_and_braking_limits or args.source_step_count != 1:
            raise ValueError("First-step probe requires mocap pipeline, one source step, and acceleration/braking limits off")
        args.mocap_unload_test = True
    if args.mocap_unload_test:
        if not args.mocap_pipeline:
            raise ValueError("--mocap-unload-test requires --mocap-pipeline")
        args.until_motion_end = False
    if args.mocap_pipeline:
        if args.motion_pkl is None:
            raise ValueError('--mocap-pipeline requires --motion-pkl')
        if args.source_arm_roll_offset or args.left_shoulder_roll_offset or args.coordinated_source_start or args.unified_transfer_test or args.unified_source_step_test:
            raise ValueError('Mocap pipeline excludes experimental reference replacements')
        if args.motion_warmup_duration <= 0 or args.motion_ramp_duration <= 0:
            raise ValueError('Mocap pipeline requires positive warmup and clock ramp durations')
        if (args.source_support_switch or args.source_contact_events
                or args.source_step_profile is not None or args.source_step_stop_duration):
            raise ValueError('Mocap pipeline excludes reduced-model source-step/stop planners')
        args.reference_mode = 'unified'
        args.task_profile = 'dynamics-aware'
        args.reference_manager = True
        args.reference_footsteps = 'source'
        args.motion_interpolation = 'pchip-analytic' if args.tracking_steps else 'bspline-analytic'
        args.direct_wbc = True
        if args.tracking_steps:
            if args.source_recovery_after_step or args.source_recovery_step_count:
                raise ValueError('Strict steps cannot contain recovery steps')
            # Track the bounded PKL directly. SourceStepRollout introduces
            # reduced COM previews and a per-step stopping planner.
            # Contact sequencing remains owned by the measured-contact manager.
    if args.g1_md:
        from fabrics_sim.solver.fabric_mit import configure_fabric_mit
        configure_fabric_mit(args)
        PD_TIMESTEP = 0.0005  # Isaac PhysX: two substeps per 1 kHz control tick.
    if args.unified_source_step_test:
        args.unified_transfer_test = True
    from fabrics_sim.reference.whole_body import WholeBodyReference, validate_reference_options
    unified_reference = validate_reference_options(args)
    if args.unified_transfer_test and (not unified_reference or not args.reference_manager or args.until_motion_end):
        raise ValueError("--unified-transfer-test requires unified references/manager and cannot use --until-motion-end")
    if args.unified_source_step_test and not (args.direct_wbc and args.motor_armature
            and args.control_frequency == 500 and args.observed_contact_patches
            and args.early_touchdown and args.reference_footsteps == "source"):
        raise ValueError("--unified-source-step-test requires direct WBC, motor armature, 500 Hz, "
                         "observed contact patches, early touchdown and source footsteps")
    if not np.isfinite(args.later_step_stop_duration) or args.later_step_stop_duration <= 0:
        raise ValueError("Later source steps require a positive finite stop duration")
    source_step_profile = {}
    if args.source_step_profile is not None:
        source_step_profile = json.loads(args.source_step_profile.read_text())
        if (not isinstance(source_step_profile,dict) or set(source_step_profile)-{'first','following'}
                or any(not isinstance(value,dict) for value in source_step_profile.values())):
            raise ValueError('Source step profile requires first/following option objects')
    if args.source_contact_events and (not args.source_support_switch or args.source_step_count<1):
        raise ValueError("Source contact events require --source-support-switch")
    if not np.isfinite(args.source_tail_duration) or args.source_tail_duration <= 0:
        raise ValueError("Source tail duration must be finite and positive")
    from fabrics_sim.reference.source_sequence import source_recovery_active
    source_recovery_active(0, args.source_recovery_after_step, args.source_recovery_step_count)
    if (not np.isfinite(args.recovery_left_transfer_roll_offset)
            or args.source_recovery_after_step < 0 or args.source_recovery_after_step >= args.source_step_count
            or not np.isfinite(args.recovery_maximum_step_distance) or args.recovery_maximum_step_distance <= 0
            or not np.isfinite(args.recovery_landing_com_fraction) or not 0 <= args.recovery_landing_com_fraction <= 1):
        raise ValueError("Recovery requires a preceding source step, a positive distance cap and a COM fraction in [0,1]")
    if args.source_step_count < 1:
        raise ValueError("Source step count must be positive")
    if (not np.isfinite(args.later_step_landing_fraction) or not 0 < args.later_step_landing_fraction <= 1
            or not np.isfinite(args.later_transfer_height_offset)
            or not np.isfinite(args.later_transfer_lift_height) or args.later_transfer_lift_height <= 0):
        raise ValueError("Later steps require finite height changes, positive lift and landing fraction in (0,1]")
    if args.source_step_count > 1 and not args.source_support_switch:
        raise ValueError("Multiple source steps require --source-support-switch")
    if (not np.isfinite(args.following_step_landing_fraction) or not 0 < args.following_step_landing_fraction <= 1
            or not np.isfinite(args.following_step_landing_height) or args.following_step_landing_height <= 0):
        raise ValueError("Following step requires a valid landing fraction and height")
    if args.source_support_switch and (not args.coordinated_source_start or args.source_step_stop_duration<=0):
        raise ValueError("--source-support-switch requires coordinated start and a landing stop")
    if (not np.isfinite(args.source_step_stop_duration) or args.source_step_stop_duration < 0
            or (args.source_step_stop_duration and not args.coordinated_source_start)):
        raise ValueError("Source-step stop requires nonnegative duration and coordinated start")
    if (not np.isfinite(args.source_step_landing_search_depth) or not 0 <= args.source_step_landing_search_depth <= .01
            or (args.source_step_landing_search_depth != 0 and not args.coordinated_source_start)):
        raise ValueError("Source-step landing search requires coordinated start and depth in [0,.01]")
    if (not np.isfinite(args.source_step_landing_fraction) or not 0 < args.source_step_landing_fraction <= 1
            or (args.source_step_landing_fraction != 1 and not args.coordinated_source_start)):
        raise ValueError("--source-step-landing-fraction requires (0,1] and coordinated source start for adaptation")
    if args.coordinated_source_start:
        if (args.unified_transfer_test or not unified_reference or not args.reference_manager
                or not args.direct_wbc or not args.motor_armature or args.control_frequency != 500
                or not args.observed_contact_patches or not args.early_touchdown
                or args.reference_footsteps != "source"):
            raise ValueError('--coordinated-source-start requires unified source references, manager, '
                             '500 Hz direct WBC, armature and observed/early contacts; excludes transfer diagnostics')
    FABRIC_TIMESTEP, PD_TIMESTEP = control_timesteps(
        args.control_frequency, args.physics_frequency, g1_md=args.g1_md)
    if args.g1_md:
        print(f"[g1.md] Fabric composition -> constrained dynamics -> integrated MIT torque; "
              f"{1/FABRIC_TIMESTEP:g} Hz control / {1/PD_TIMESTEP:g} Hz physics "
              "(simulation time, not a real-time guarantee)")
    if args.tracking_profile == 'amp-full':
        # Preserve confirmation time when refining the synchronous baseline.
        args.contact_switch_confirm_updates = max(1, round(
            args.contact_switch_confirm_updates * .002 / FABRIC_TIMESTEP))
    if args.until_motion_end and not args.reference_manager:
        raise ValueError('--until-motion-end requires --reference-manager')
    if args.reference_footsteps != "compact" and not args.reference_manager:
        raise ValueError('--reference-footsteps requires --reference-manager')
    if (not math.isfinite(args.toe_transfer_load_fraction) or not 0.5 <= args.toe_transfer_load_fraction <= 1.0 or args.toe_transfer_load_fraction != 0.8):
        raise ValueError('--toe-transfer-load-fraction requires --toe-off and a fraction between 0.5 and 1')
    if (not math.isfinite(args.source_pelvis_weight) or args.source_pelvis_weight < 0
            or (args.source_pelvis_weight > 0 and not
                ((args.reference_manager and args.reference_footsteps == "source") or args.motion_start_in_motion))):
        raise ValueError('--source-pelvis-weight requires a nonnegative finite value and managed source footsteps or a dynamic source start')
    if args.physics_iterations is not None and (args.physics_iterations[0]<1 or args.physics_iterations[1]<0):
        raise ValueError('Physics position iterations must be positive; velocity iterations must be nonnegative')
    if args.source_bridge_duration is not None and (not math.isfinite(args.source_bridge_duration)
            or args.source_bridge_duration<=0 or not args.source_swing_reference):
        raise ValueError('--source-bridge-duration requires a positive finite duration and source swing reference')
    if args.landing_speed_limit is not None and (not (args.reference_manager or args.motion_start_in_motion)
            or not math.isfinite(args.landing_speed_limit) or args.landing_speed_limit<=0):
        raise ValueError('--landing-speed-limit requires a reference manager or dynamic start and positive finite speed')
    if (math.isnan(args.weighted_tasks_start) or args.weighted_tasks_start < 0
            or not math.isfinite(args.weighted_tasks_ramp) or args.weighted_tasks_ramp <= 0):
        raise ValueError('Weighted task transition requires nonnegative start and positive ramp')
    if math.isfinite(args.weighted_tasks_start) and (
            args.tracking_profile != 'amp-full' or args.g1_com_priority):
        raise ValueError('Weighted task experiment requires amp-full without a separate COM priority')
    if math.isnan(args.support_flatten_start) or args.support_flatten_start < 0:
        raise ValueError('Support flatten start must be nonnegative or infinity')
    if (not math.isfinite(args.support_flatten_gain_scale) or args.support_flatten_gain_scale<=0
            or not math.isfinite(args.support_flatten_feedback_limit) or args.support_flatten_feedback_limit<=0):
        raise ValueError('Support flatten gain and feedback limit must be finite and positive')
    if math.isfinite(args.support_flatten_start) and args.tracking_profile != 'amp-full':
        raise ValueError('Support flatten experiment requires the amp-full profile')
    if (math.isnan(args.root_position_task_start) or args.root_position_task_start < 0
            or any(not math.isfinite(x) or x <= 0 for x in
                   (args.root_position_task_weight, args.root_position_task_ramp))):
        raise ValueError('Root tracking requires nonnegative start and positive finite weight/ramp')
    if math.isfinite(args.root_position_task_start) and not (
            args.tracking_profile == 'amp-full' and args.direct_wbc):
        raise ValueError('Root position task currently requires the amp-full direct-WBC profile')
    if args.root_position_task_late_weight is not None and (
            not math.isfinite(args.root_position_task_late_weight) or args.root_position_task_late_weight <= 0
            or not math.isfinite(args.root_position_task_late_start)
            or args.root_position_task_late_start < args.root_position_task_start
            or not math.isfinite(args.root_position_task_late_ramp) or args.root_position_task_late_ramp <= 0):
        raise ValueError('Late root cost requires an enabled root task, finite positive weight/ramp and a later source start')
    if (math.isnan(args.stance_posture_metric_start) or args.stance_posture_metric_start < 0
            or not math.isfinite(args.stance_posture_metric_scale)
            or not 0 < args.stance_posture_metric_scale <= 1
            or not math.isfinite(args.stance_posture_metric_ramp) or args.stance_posture_metric_ramp <= 0):
        raise ValueError('Stance posture weighting requires nonnegative start, scale in (0,1], and positive ramp')
    if math.isfinite(args.stance_posture_metric_start) and not (
            args.motion_start_in_motion and args.task_profile == 'dynamics-aware'):
        raise ValueError('Stance posture weighting requires dynamic source tracking and physical tasks')
    if math.isfinite(args.rolling_contact_kinematics_start) and (
            args.rolling_contact_kinematics_start < 0 or not args.observed_contact_patches
            or not args.motion_start_in_motion or not args.track_full_motion):
        raise ValueError('Rolling contact correction requires full dynamic tracking with native contact patches')
    if math.isnan(args.rolling_contact_kinematics_start) or args.rolling_contact_kinematics_start == -float('inf'):
        raise ValueError('Rolling contact correction start must be nonnegative or inf')
    if not math.isfinite(args.landing_speed_limit_start) or args.landing_speed_limit_start < 0:
        raise ValueError('--landing-speed-limit-start must be finite and nonnegative')
    if args.late_momentum_zero_target and args.late_momentum_feedback_damping is None:
        raise ValueError('Zero momentum regulation requires a positive feedback damping')
    if args.balance_body_priority and (not args.separate_balance_priority
            or args.source_pelvis_weight<=0 or args.task_profile!='dynamics-aware'):
        raise ValueError('--balance-body-priority requires separate balance, physical tasks and a positive source pelvis weight')
    if args.landing_orientation_priority and not (args.observed_contact_patches and args.transfer_balance_priority):
        raise ValueError('--landing-orientation-priority requires observed contact patches and transfer balance priority')
    if args.horizontal_balance_priority and (not args.separate_balance_priority):
        raise ValueError('--horizontal-balance-priority requires separate balance priority and excludes swing height priority')
    if args.tracking_servo and not args.direct_wbc:
        raise ValueError('--tracking-servo requires --direct-wbc')
    if args.contact_aware_servo and not args.tracking_servo:
        raise ValueError('--contact-aware-servo requires --tracking-servo')
    if args.nominal_implicit_servo and not args.contact_aware_servo:
        raise ValueError('--nominal-implicit-servo requires --contact-aware-servo')
    if args.balance_priority and args.task_profile != "dynamics-aware":
        raise ValueError('--balance-priority requires --task-profile dynamics-aware')
    if args.wrist_hip_avoidance and (args.task_profile != 'dynamics-aware' or args.disable_collision_task):
        raise ValueError('--wrist-hip-avoidance requires enabled dynamics-aware collision tasks')
    if (not np.isfinite(args.source_arm_roll_offset) or args.source_arm_roll_offset<0
            or (args.source_arm_roll_offset and not (unified_reference and args.motion_pkl is not None))):
        raise ValueError('Arm adaptation requires a nonnegative offset and unified PKL tracking')
    if args.convex_wrist_collision and (not args.wrist_hip_avoidance or str(args.device) != 'cpu'):
        raise ValueError('Convex wrist collision requires wrist avoidance and CPU dynamics')
    if args.wrist_hip_barrier and not args.wrist_hip_avoidance:
        raise ValueError('--wrist-hip-barrier requires --wrist-hip-avoidance')
    if args.wrist_body_avoidance and not args.wrist_hip_barrier:
        raise ValueError('--wrist-body-avoidance requires --wrist-hip-barrier')
    if args.transfer_balance_priority and (not (args.unified_transfer_test or args.coordinated_source_start) or args.balance_priority):
        raise ValueError('--transfer-balance-priority requires dynamic references or a unified transfer, and excludes --balance-priority')
    if (args.source_swing_position_priority or args.source_swing_pose_priority) and not (
            unified_reference and (args.coordinated_source_start or args.mocap_first_step_report is not None)):
        raise ValueError("Source swing priority requires a coordinated unified source reference or bounded first-step probe")
    if args.source_swing_position_priority and args.source_swing_pose_priority:
        raise ValueError("Select source swing position or pose priority, not both")
    if (args.swing_clearance_priority or args.swing_position_priority or args.swing_pose_priority) and (not args.balance_priority):
        raise ValueError('Swing-foot priority requires a COM priority option')
    if sum((args.swing_clearance_priority,args.swing_position_priority,args.swing_pose_priority))>1:
        raise ValueError('Select one swing-foot priority level')
    if (args.upper_body_priority or args.whole_body_posture_priority) and (not (args.balance_priority or args.transfer_balance_priority)):
        raise ValueError('Upper-body priority requires an active COM priority configuration')
    if args.upper_body_priority and args.whole_body_posture_priority:
        raise ValueError('Select upper-body-only or whole-body posture priority')
    if args.centroidal_balance_priority and (not (args.balance_priority or args.transfer_balance_priority)):
        raise ValueError('Centroidal balance requires an active COM priority')
    if args.body_orientation_priority and (args.upper_body_priority or args.source_pelvis_weight <= 0 or (not (args.balance_priority or args.transfer_balance_priority))):
        raise ValueError('Body orientation priority needs a COM priority and positive pelvis weight; excludes upper-body priority')
    if not math.isfinite(args.swing_touchdown_lead) or not 0 <= args.swing_touchdown_lead <= .3:
        raise ValueError('--swing-touchdown-lead must be between 0 and 0.3 seconds')
    if not math.isfinite(args.minimum_step_duration) or args.minimum_step_duration < 0:
        raise ValueError('--minimum-step-duration must be finite and nonnegative')
    if args.observed_contact_patches:
        if not (unified_reference and (args.motion_start_in_motion or (args.reference_manager and args.early_touchdown))):
            raise ValueError('--observed-contact-patches requires unified references with managed early touchdown or a dynamic source start')
        args.diagnose_contacts = True
    if (not math.isfinite(args.joint_limit_avoidance_weight) or args.joint_limit_avoidance_weight < 0
            or (args.joint_limit_avoidance_weight > 0 and args.task_profile != 'dynamics-aware')):
        raise ValueError('--joint-limit-avoidance-weight requires a finite nonnegative value and physical tasks')
    if args.qp_reuse_workspace and args.qp_solver != 'osqp':
        raise ValueError('--qp-reuse-workspace requires --qp-solver osqp')
    if args.qp_active_set_predictor and args.qp_solver != 'osqp':
        raise ValueError('--qp-active-set-predictor requires --qp-solver osqp')
    if args.source_momentum_reference and (not args.reference_manager or args.reference_footsteps != 'source'):
        raise ValueError('--source-momentum-reference requires managed source footsteps')
    if args.align_takeoff_frame and (not args.reference_manager or args.reference_footsteps != 'source'):
        raise ValueError('--align-takeoff-frame requires managed source footsteps')
    if not np.isfinite(args.takeoff_lead) or not 0. <= args.takeoff_lead <= .15 or (args.takeoff_lead and not args.align_takeoff_frame):
        raise ValueError('--takeoff-lead needs aligned takeoff and a value between 0 and .15')
    if args.source_swing_reference and (not args.reference_manager or args.reference_footsteps != 'source'
                                       or args.motion_interpolation != 'pchip-analytic'):
        raise ValueError('--source-swing-reference requires managed source footsteps and pchip-analytic')
    if args.joint_limit_time_constant is not None and (
            not math.isfinite(args.joint_limit_time_constant) or args.joint_limit_time_constant <= 0
            or args.motion_pkl is None):
        raise ValueError('--joint-limit-time-constant requires a PKL and a finite positive value')
    if args.flat_foot_spawn and (args.motion_pkl is None or args.task_profile != "dynamics-aware"
                                or args.motion_warmup_duration <= 0):
        raise ValueError('--flat-foot-spawn requires a PKL, dynamics-aware tasks and positive warmup')
    if args.joint_limit_barrier_bandwidth is not None and (
            args.motion_pkl is None or not math.isfinite(args.joint_limit_barrier_bandwidth)
            or args.joint_limit_barrier_bandwidth <= 0):
        raise ValueError('--joint-limit-barrier-bandwidth requires a PKL and finite positive bandwidth')
    if args.early_touchdown and not args.reference_manager:
        raise ValueError('--early-touchdown requires --reference-manager')
    if args.direct_wbc and (args.task_profile != "dynamics-aware" or args.fixed_joint_test):
        raise ValueError('--direct-wbc requires dynamics-aware tasks and excludes fixed-joint servo diagnostics')
    if args.contact_stabilization_kp is None:
        args.contact_stabilization_kp = 20. if args.reference_manager else 0.
    if args.contact_stabilization_kd is None:
        args.contact_stabilization_kd = 8. if args.reference_manager else 0.
    if args.stabilize_foot_rotation is None:
        args.stabilize_foot_rotation = args.reference_manager
    if args.contact_cycle_test:
        if args.single_support_test and args.single_support_test != args.contact_cycle_test:
            raise ValueError("Contact cycle and single-support test must select the same support foot.")
        args.single_support_test = args.contact_cycle_test
        args.align_balance_pelvis = True
        args.require_support_load = True
    if args.g1_com_priority and not args.g1_md:
        raise ValueError('--g1-com-priority requires --g1-md')
    if args.diagnose_com and args.motion_pkl is None:
        raise ValueError("--diagnose-com requires --motion-pkl for sampled history output.")
    if args.diagnose_contacts and args.motion_pkl is None:
        raise ValueError("--diagnose-contacts requires --motion-pkl for sampled history output.")
    if args.diagnose_qp_time and (not args.diagnose_com or args.fixed_joint_test or
            any(not math.isfinite(t) or t < 0 or t >= args.duration for t in args.diagnose_qp_time)):
        raise ValueError("QP snapshot times must be finite within duration, require --diagnose-com and exclude --fixed-joint-test.")
    if not 0. <= args.jdot_filter_alpha < 1.:
        raise ValueError("--jdot-filter-alpha must be in [0,1).")
    if args.weight_transfer_test and (args.motion_pkl is None or args.fixed_joint_test):
        raise ValueError("--weight-transfer-test requires motion PKL and cannot combine with --fixed-joint-test.")
    if args.single_support_test and (args.motion_pkl is None or args.fixed_joint_test or args.weight_transfer_test
                                    or not args.align_balance_pelvis):
        raise ValueError("Single-support test requires PKL and --align-balance-pelvis; excludes other isolated tests.")
    if not math.isfinite(args.left_shoulder_roll_offset):
        raise ValueError("--left-shoulder-roll-offset must be finite.")
    if not math.isfinite(args.servo_bandwidth) or args.servo_bandwidth <= 0:
        raise ValueError("--servo-bandwidth must be finite and positive.")
    if args.disable_self_collisions:
        print("DIAGNOSTIC: self-collisions disabled; this run cannot certify motion tracking.")
    if args.contact_mode is not None:
        print(
            "warning: --contact-mode is deprecated and ignored; "
            "contact mode comes from PKL prediction when a motion is loaded."
        )
    if args.regularization < 0.0:
        raise ValueError("--regularization must be nonnegative.")
    if args.duration <= 0.0:
        raise ValueError("--duration must be positive.")
    if args.motion_speed <= 0.0:
        raise ValueError("--motion-speed must be positive.")
    if not math.isfinite(args.motion_ramp_duration) or args.motion_ramp_duration < 0:
        raise ValueError("--motion-ramp-duration must be finite and nonnegative.")
    if args.reference_manager and (args.motion_pkl is None or args.task_profile != "dynamics-aware"
            or args.fixed_joint_test or args.weight_transfer_test or args.single_support_test
            or args.allow_flight or args.motion_ramp_duration <= 0):
        raise ValueError("Reference manager requires a PKL, dynamics-aware profile, positive ramp, and normal non-flight tracking.")
    if args.motion_warmup_duration < 0.0:
        raise ValueError("--motion-warmup-duration must be nonnegative.")
    if not math.isfinite(args.support_transfer_duration) or args.support_transfer_duration <= 0:
        raise ValueError("--support-transfer-duration must be finite and positive.")
    if not math.isfinite(args.cycle_crouch_depth) or not 0 <= args.cycle_crouch_depth <= .08:
        raise ValueError("--cycle-crouch-depth must be between 0 and .08 m.")
    if args.cycle_crouch_depth and not args.contact_cycle_test:
        raise ValueError("--cycle-crouch-depth requires --contact-cycle-test.")
    cycle_preparation = 3. if args.contact_cycle_test and args.cycle_crouch_depth else 0.
    if args.reference_contact_height_tolerance <= 0.0:
        raise ValueError(
            "--reference-contact-height-tolerance must be positive."
        )
    if args.contact_stabilization_kp < 0.0:
        raise ValueError("--contact-stabilization-kp must be nonnegative.")
    if args.contact_stabilization_kd < 0.0:
        raise ValueError("--contact-stabilization-kd must be nonnegative.")
    if args.contact_stabilization_max_acceleration <= 0.0:
        raise ValueError(
            "--contact-stabilization-max-acceleration must be positive."
        )
    if args.com_lookahead < 0.0:
        raise ValueError("--com-lookahead must be nonnegative.")
    if args.com_target_time_constant <= 0.0:
        raise ValueError("--com-target-time-constant must be positive.")
    if args.contact_switch_confirm_updates < 1:
        raise ValueError(
            "--contact-switch-confirm-updates must be at least one."
        )
    if args.qp_validation_tolerance <= 0.0:
        raise ValueError("--qp-validation-tolerance must be positive.")
    if args.qp_acceleration_limit_scale <= 0.0:
        raise ValueError("--qp-acceleration-limit-scale must be positive.")
    initial_linear_weight = linear_acceleration_weight(
        args.qp_base_acceleration_weight, args.qp_base_linear_acceleration_weight, 0.,
        start=args.qp_base_linear_weight_start, ramp=args.qp_base_linear_weight_ramp)
    if args.maximum_com_tracking_error <= 0.0:
        raise ValueError("--maximum-com-tracking-error must be positive.")
    if not 0.0 < args.minimum_com_height_ratio <= 1.0:
        raise ValueError(
            "--minimum-com-height-ratio must be in the interval (0, 1]."
        )
    if args.maximum_joint_position_rmse <= 0.0:
        raise ValueError("--maximum-joint-position-rmse must be positive.")
    if args.swing_metric_scale is not None and (not (args.reference_manager or args.motion_start_in_motion)
            or not math.isfinite(args.swing_metric_scale) or args.swing_metric_scale <= 0):
        raise ValueError("--swing-metric-scale requires a reference manager or dynamic start and a finite positive value.")
    if args.posture_metric_scale is not None and (not (args.reference_manager or args.motion_start_in_motion)
            or not math.isfinite(args.posture_metric_scale) or args.posture_metric_scale <= 0):
        raise ValueError("--posture-metric-scale requires a reference manager or dynamic start and a finite positive value.")
    if not math.isfinite(args.motion_feedback_scale) or args.motion_feedback_scale <= 0:
        raise ValueError('--motion-feedback-scale must be finite and positive')
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)

    wp.init()
    fabric = G1PoseFabric(
        batch_size=1,
        device=str(device),
        timestep=FABRIC_TIMESTEP,
        graph_capturable=False,
        # Match the actual USD rather than the older URDF (different torso
        # geometry and inertial properties).
        robot_name=G1_USD_PATH.stem,
        task_profile=args.task_profile,
    )
    if acceleration_only:
        fabric.dynamics_tasks.cache_task_requests = recompose_tasks
        fabric.dynamics_tasks.priority_mode = 'none' if recompose_tasks else args.fabric_priority_mode
    if args.fabric_root_regularization is not None:
        if fabric.dynamics_tasks is None:
            raise ValueError('Root regularization requires physical Fabric tasks')
        fabric.dynamics_tasks.config['regularization'] = args.fabric_root_regularization
    if args.com_acceleration_preservation_weight and fabric.dynamics_tasks is None:
        raise ValueError('COM acceleration preservation requires dynamics-aware kinematics')
    if args.motion_start_in_motion:
        if args.swing_metric_scale is not None:
            fabric.fabric_params['feet_pose_attractor']['metric'] = [
                x*args.swing_metric_scale for x in fabric.fabric_params['feet_pose_attractor']['metric']]
        if args.posture_metric_scale is not None:
            fabric.fabric_params['joint_posture_attractor']['metric'] *= args.posture_metric_scale
    if args.motion_feedback_scale != 1.:
        if fabric.dynamics_tasks is None:
            raise ValueError('--motion-feedback-scale requires dynamics-aware tasks')
        for params in (fabric.fabric_params['joint_posture_attractor'],
                       fabric.fabric_params['feet_pose_attractor'],
                       fabric.dynamics_tasks.config['body']):
            for key, scale in (('stiffness', args.motion_feedback_scale),
                               ('damping', math.sqrt(args.motion_feedback_scale))):
                value = np.asarray(params[key]) * scale
                params[key] = value.tolist() if value.ndim else float(value)
    fabric.fabric_params['joint_posture_attractor']['max_acceleration'] = args.joint_feedback_limit
    if args.fabric_angular_feedback_limit is not None:
        if (fabric.dynamics_tasks is None or not math.isfinite(args.fabric_angular_feedback_limit)
                or args.fabric_angular_feedback_limit <= 0):
            raise ValueError('Fabric angular feedback limit requires dynamics-aware tasks and a positive finite value')
        fabric.dynamics_tasks.config['body']['max_acceleration'] = [args.fabric_angular_feedback_limit]*3
    if not args.acceleration_and_braking_limits:
        fabric.fabric_params['joint_limits']['active'] = False
        args.limit_predicted_acceleration = False
    if args.collision_obstacles is not None and fabric.dynamics_tasks is None:
        raise ValueError("--collision-obstacles requires --task-profile dynamics-aware.")
    if fabric.dynamics_tasks is not None:
        fabric.dynamics_tasks.set_hand_tracking(args.hand_tracking)
        fabric.dynamics_tasks.collision_enabled = not args.disable_collision_task
        fabric.dynamics_tasks.stable_constraint_rows = args.qp_reuse_workspace
        if args.wrist_hip_avoidance:
            fabric.dynamics_tasks.enable_wrist_hip_avoidance(include_upper_body=args.wrist_body_avoidance, convex=args.convex_wrist_collision)
            delayed_convex_geometry=fabric.dynamics_tasks.convex_wrist_geometry
            if args.convex_wrist_collision_start>0:
                fabric.dynamics_tasks.convex_wrist_geometry=None
            fabric.dynamics_tasks.wrist_hip_barrier = args.wrist_hip_barrier
        fabric.dynamics_tasks.momentum_enabled = not args.disable_momentum_task
        if args.collision_obstacles is not None:
            spheres = torch.tensor(json.loads(args.collision_obstacles.read_text()), device=device, dtype=torch.float32)
            fabric.set_collision_obstacles(spheres.reshape(1, -1, 4))
    motion_trajectory = None
    motion_contact_sequence = None
    motion_planned_modes = None
    motion_fps = None
    motion_start_frame = 0
    bounded_events = None
    measured_steps = None
    if args.motion_pkl is None:
        q = torch.zeros((1, 36), dtype=torch.float32, device=device)
        q[:, 2] = 0.8
        q[:, 6] = 1.0
    else:
        motion_trajectory_cpu, motion_fps = load_motion_trajectory(
            args.motion_pkl
        )
        if args.left_shoulder_roll_offset:
            motion_trajectory_cpu = offset_left_shoulder_reference(
                motion_trajectory_cpu, args.left_shoulder_roll_offset)
            print(f"REFERENCE MODIFIED: left shoulder roll offset={args.left_shoulder_roll_offset:g} rad; PKL unchanged")
        contact_path = args.contacts_pkl or args.motion_pkl
        if args.contacts_pkl is not None:
            contact_q, contact_fps = load_motion_trajectory(contact_path)
            if contact_fps != motion_fps or contact_q.shape != motion_trajectory_cpu.shape or not torch.allclose(contact_q, motion_trajectory_cpu, atol=1e-6, rtol=0):
                raise ValueError('Contact companion poses/fps do not match motion')
        motion_contact_sequence_cpu = load_motion_contact_sequence(contact_path)
        if args.contacts_pkl is not None and motion_contact_sequence_cpu is None:
            raise ValueError('Explicit contact companion has no contact labels')
        if args.tracking_steps or args.track_full_motion:
            from fabrics_sim.reference.bounded_steps import select_steps, select_dynamic_steps, select_all_dynamic_steps
            from fabrics_sim.reference.manager import contact_sequence_modes
            if motion_contact_sequence_cpu is None:
                raise ValueError('Strict bounded steps require contact labels; supply --contacts-pkl')
            selector = select_dynamic_steps if args.motion_start_in_motion else select_steps
            modes = contact_sequence_modes(motion_contact_sequence_cpu.numpy(), len(motion_trajectory_cpu))
            bounded_events = (select_all_dynamic_steps(modes) if args.track_full_motion
                              else selector(modes, args.tracking_steps))
            if not bounded_events:
                raise ValueError('The contact plan contains no complete source landings')
            args.original_source_frame_count = len(motion_trajectory_cpu)
            if not args.motion_start_in_motion:
                end = bounded_events[-1]['end_frame']
                motion_trajectory_cpu = motion_trajectory_cpu[:end+1]
                motion_contact_sequence_cpu = motion_contact_sequence_cpu[:end+1]
            # Continuous walking keeps the source tail: truncating here would
            # force reference v/a to zero before touchdown is confirmed.
            print(f'Strict source step plan: {bounded_events}', flush=True)
        motion_start_frame = (
            motion_trajectory_cpu.shape[0] // 2
            if args.motion_frame == -1
            else args.motion_frame
        )
        if not 0 <= motion_start_frame < motion_trajectory_cpu.shape[0]:
            raise IndexError(
                f"Motion frame {motion_start_frame} is outside "
                f"[0, {motion_trajectory_cpu.shape[0] - 1}]"
            )
        motion_trajectory = motion_trajectory_cpu.to(device)
        args.source_frame_count=len(motion_trajectory)
        args.source_start_frame=motion_start_frame
        if motion_contact_sequence_cpu is not None:
            motion_contact_sequence = motion_contact_sequence_cpu.to(device)
        q = motion_trajectory[
            motion_start_frame:motion_start_frame + 1
        ].clone()
        print(
            f"loaded motion: {args.motion_pkl} "
            f"(frames={motion_trajectory.shape[0]}, fps={motion_fps:g}, "
            f"start={motion_start_frame})"
        )
    motion_ground_shift = 0.0
    if motion_trajectory is not None and args.ground_motion:
        motion_trajectory, motion_ground_shift = ground_motion_reference(fabric, motion_trajectory, motion_start_frame)
        q = motion_trajectory[motion_start_frame:motion_start_frame+1].clone()
        print(f"Spawn ground alignment: reference z shift={motion_ground_shift:.6f} m; PKL unchanged", flush=True)
    control_motion_trajectory = motion_trajectory
    if args.source_arm_roll_offset:
        import xml.etree.ElementTree as ET
        from fabrics_sim.reference.arm_clearance import arm_clearance_reference
        limits={node.attrib['name']:node.find('limit') for node in ET.parse(fabric.urdf_path).getroot().findall('joint')}
        control_motion_trajectory = arm_clearance_reference(motion_trajectory,G1_JOINT_NAMES,
            args.source_arm_roll_offset,
            q.new_tensor([float(limits[name].attrib['lower']) for name in G1_JOINT_NAMES]),
            q.new_tensor([float(limits[name].attrib['upper']) for name in G1_JOINT_NAMES]))
        q = control_motion_trajectory[motion_start_frame:motion_start_frame+1].clone()
        print(f'Explicit arm clearance adaptation: {args.source_arm_roll_offset:.4f} rad; original PKL retained for audit',flush=True)
    if args.flat_foot_spawn:
        from fabrics_sim.reference.initial_stance import flat_foot_spawn
        source_spawn=q.clone()
        q=flat_foot_spawn(fabric,q,G1_JOINT_NAMES)
        print(f"Flat-foot spawn: root height change={float(q[0,2]-source_spawn[0,2]):.6f} m, "
              f"max joint change={float((q[:,7:]-source_spawn[:,7:]).abs().max()):.6f} rad; reference unchanged",flush=True)
    qd = torch.zeros((1, 35), dtype=torch.float32, device=device)
    if (
        motion_trajectory is not None
        and args.motion_warmup_duration == 0.0
        and args.motion_ramp_duration == 0.0
    ):
        qd = args.motion_speed * motion_generalized_velocity(
            motion_trajectory,
            motion_start_frame,
            motion_fps,
        )

    if unified_reference:
        WholeBodyReference(q, qd, torch.zeros_like(qd)).apply(fabric)
        com_balance_target = fabric.center_of_mass_taskmap.position_target.clone()
        initial_center_of_mass = com_balance_target.clone()
        features = fabric.fabrics_features['center_of_mass']['center_of_mass_attractor']
        com_target_velocity = features['velocity'].clone()
        com_target_acceleration = features['acceleration'].clone()
    else:
        configure_motion_targets(fabric, q)
        initial_center_of_mass = fabric.compute_center_of_mass(q).detach().clone()
        com_balance_target = initial_center_of_mass.clone()
        com_target_velocity = torch.zeros_like(com_balance_target)
        com_target_acceleration = torch.zeros_like(com_balance_target)
        fabric.set_center_of_mass_target(
            com_balance_target,
            com_target_velocity,
            torch.zeros_like(com_balance_target),
        )
    ground_reference_q = (
        motion_trajectory[0:1] if motion_trajectory is not None else q
    )
    motion_sampler = None
    if motion_trajectory is not None and args.motion_interpolation in ('pchip', 'pchip-analytic', 'bspline-analytic'):
        from fabrics_sim.reference.interpolation import MotionInterpolator
        if args.mocap_pipeline:
            from fabrics_sim.reference.preprocessing import prepare_reference
            motion_sampler, preparation = prepare_reference(control_motion_trajectory, motion_fps,
                G1_JOINT_NAMES, fabric.urdf_path,
                args.reference_acceleration_cap if args.acceleration_and_braking_limits else None, args.motion_speed,
                method="pchip" if args.tracking_steps else "bspline")
            args.motion_speed = preparation['playback_speed']
            preparation.update(motion=str(args.motion_pkl), contact_source=str(args.contacts_pkl or args.motion_pkl),
                               contact_labels_present=motion_contact_sequence is not None,
                               reference_policy='direct_full_body_pkl', reduced_model_preview=False,
                               capture_point_gate=False)
            args.output_path.parent.mkdir(parents=True, exist_ok=True)
            args.output_path.with_suffix('.preparation.json').write_text(json.dumps(preparation, indent=2)+'\n')
            print('Online reference prepared: '+json.dumps(preparation), flush=True)
        else:
            motion_sampler = MotionInterpolator(control_motion_trajectory,
                method='bspline' if args.motion_interpolation == 'bspline-analytic' else 'pchip')

    def sample_motion(frame):
        return (interpolate_motion_trajectory(control_motion_trajectory, frame)
                if motion_sampler is None else motion_sampler(frame))

    reference_ground_heights = reference_feet_transforms(
        fabric, ground_reference_q
    )[..., 2]
    reference_manager = None
    motion_completion = None
    managed_clock = None
    landing_transition = None
    landing_key = None
    com_preview = None
    com_preview_last_replan = -float('inf')
    landing_height_cache = {}
    landing_velocity_cache = {}
    source_momentum_cache = {}
    source_momentum_samplers = {}

    def source_landing_pose(pose):
        from fabrics_sim.reference.footstep import source_contact_step, source_grounded_step
        return (source_contact_step(pose) if args.source_swing_reference else source_grounded_step(pose))


    def reference_landing_state(frame, stance_index=None):
        return None
        key=frame
        kwargs={}
        if key not in landing_height_cache:
            # Derive height from the same contact geometry as the landing plan.
            # This is reference-only IK; no live pose is written.
            from fabrics_sim.reference.initial_stance import flat_foot_reference
            grounded = flat_foot_reference(fabric, motion_trajectory[frame:frame+1], G1_JOINT_NAMES, **kwargs)
            landing_height_cache[key] = _to_numpy(fabric.compute_center_of_mass(grounded)[0]).copy()
        result=landing_height_cache[key].copy()
        if stance_index is not None:
            result[:2]+=.5*_to_numpy(contact_anchors[0,stance_index,:2]-reference_feet_all[frame,stance_index,:2])
        return result

    def reference_landing_height(frame, stance_index=None):
        state=reference_landing_state(frame,stance_index)
        return None if state is None else float(state[2])

    def reference_landing_com_xy(frame, stance_index):
        return None
        source=reference_landing_state(frame,stance_index)[:2]
        midpoint=.5*_to_numpy(contact_anchors[0,stance_index,:2]+reference_feet_all[frame,1-stance_index,:2])
        forward=_to_numpy(_quat_xyzw_to_matrix(contact_anchors[:,stance_index,3:])[0,:2,0])
        forward=forward/np.linalg.norm(forward)
        # Source fore-aft placement permits forward capture at the new foot.
        # Keep lateral transfer centered between the soles: retargeted source
        # COM sway alone is not a viable terminal capture condition.
        return midpoint+forward*np.dot(source-midpoint,forward)

    def preferred_step_duration(endpoint,takeoff,start,velocity,goal,swing,source_frame,source_rate):
        duration=max(args.minimum_step_duration,(endpoint-takeoff)/(motion_fps*args.motion_speed))
        return duration

    source_com_bridge = source_foot_bridge = None
    source_rotation_bridge = None
    toe_contact = None
    toe_foot = None
    toe_ready_seconds = 0.
    patch_manager = None
    if args.observed_contact_patches:
        from fabrics_sim.solver.contact_patches import ContactPatchManager, loaded_sole_contacts
        patch_manager=ContactPatchManager()
        corners = q.new_tensor([[-.05,-.025,-.035],[-.05,.025,-.035],[.12,-.025,-.035],[.12,.025,-.035]])
    if motion_trajectory is not None:
        motion_planned_modes = torch.cat(
            [
                planned_contact_mode_for_frame(
                    fabric,
                    motion_trajectory[frame:frame + 1],
                    reference_ground_heights,
                    args.reference_contact_height_tolerance,
                    motion_contact_sequence,
                    frame,
                    args.allow_flight,
                )
                for frame in range(motion_trajectory.shape[0])
            ],
            dim=0,
        )
        mode_counts = torch.bincount(
            motion_planned_modes, minlength=4
        ).detach().cpu().tolist()
        print(
            "sanitized contact modes "
            f"flight/left/right/double={mode_counts}"
        )
    if args.native_contact_plan_start<100.:
        if motion_contact_sequence is None or motion_planned_modes is None or not math.isfinite(args.native_contact_plan_start) or args.native_contact_plan_start<0:
            raise ValueError('Native contact plan requires finite source time and an explicit contact sequence')
        first_native_frame=min(len(motion_planned_modes),int(round(args.native_contact_plan_start*motion_fps)))
        supplied_modes=motion_contact_sequence[:,0].long()+2*motion_contact_sequence[:,1].long()
        previous_modes=motion_planned_modes.clone()
        motion_planned_modes[first_native_frame:]=supplied_modes[first_native_frame:]
        args.output_path.with_suffix('.native_contact_plan.json').write_text(json.dumps(dict(
            first_source_frame=first_native_frame,
            changed_source_frames=torch.nonzero(previous_modes!=motion_planned_modes).flatten().cpu().tolist(),
            measured_support_gates_unchanged=True),indent=2)+'\n')
    if args.reference_manager:
        from fabrics_sim.reference.manager import ReferenceManager, infer_contact_plan, infer_corner_contact_plan, contact_sequence_modes
        fabric.dynamics_tasks.managed_walking = True
        fabric.dynamics_tasks.source_pelvis_metric_scale = args.source_pelvis_weight
        if args.swing_metric_scale is not None:
            fabric.dynamics_tasks.config['managed_walking']['swing_metric_scale'] = args.swing_metric_scale
        if args.posture_metric_scale is not None:
            fabric.dynamics_tasks.config['managed_walking']['posture_metric_scale'] = args.posture_metric_scale
        with torch.no_grad():
            reference_feet_all = torch.cat([reference_feet_transforms(fabric, row[None])
                                           for row in motion_trajectory], dim=0)
            corners = q.new_tensor([[-.05,-.025,-.035],[-.05,.025,-.035],[.12,-.025,-.035],[.12,.025,-.035]])
            rotations = _quat_xyzw_to_matrix(reference_feet_all[...,3:7])
            sole_points = reference_feet_all[...,:3].unsqueeze(-2)+(rotations.unsqueeze(-3)@corners[...,None]).squeeze(-1)
            heights = sole_points[...,2].amin(-1).cpu().numpy()
            centers = reference_feet_all[...,:3].cpu().numpy()
        if args.source_swing_reference:
            from fabrics_sim.reference.interpolation import MotionInterpolator
            source_foot_samplers=[MotionInterpolator(reference_feet_all[:,i],pose_dimension=7) for i in range(2)]
            source_xyz_coefficients=[None if sampler.coefficients is None else
                                     sampler.coefficients[:,:,:3].detach().cpu().double().numpy()
                                     for sampler in source_foot_samplers]
        speeds = np.linalg.norm(np.gradient(centers,1./motion_fps,axis=0),axis=-1)
        if args.reference_footsteps == "source":
            candidate_modes,ambiguous,speeds=infer_corner_contact_plan(sole_points.cpu().numpy(),motion_fps)
            contact_plan_method='sole_corner'
        else:
            candidate_modes, ambiguous = infer_contact_plan(heights,speeds,motion_fps)
            contact_plan_method='ankle'
        contact_geometry_method = contact_plan_method
        if motion_contact_sequence is not None:
            candidate_modes = contact_sequence_modes(
                motion_contact_sequence.detach().cpu().numpy(), len(motion_trajectory))
            contact_plan_method = 'pkl_contact_sequence'
        if bounded_events is not None:
            from fabrics_sim.reference.bounded_steps import MeasuredSteps
            goals = np.stack([_to_numpy(reference_feet_all[e['end_frame'],1-e['support'],:3])
                              for e in bounded_events])
            measured_steps = MeasuredSteps(bounded_events, goals)
            args.output_path.parent.mkdir(parents=True, exist_ok=True)
            args.output_path.with_suffix('.steps_plan.json').write_text(json.dumps(dict(
                events=bounded_events, landing_positions_world=goals.tolist(),
                interpolation='PCHIP passes original poses; C1, not C2',
                original_source_frame_count=args.original_source_frame_count,
                source_positions_modified=False), indent=2)+'\n')
        motion_planned_modes = torch.as_tensor(candidate_modes,device=device)
        reference_manager = ReferenceManager(candidate_modes,motion_fps,motion_start_frame,
                                             args.motion_speed,args.motion_ramp_duration,
                                             settle_duration=0.3,
                                             early_touchdown=args.early_touchdown,
                                             settle_on_incoming=False,
                                             align_takeoff_frame=args.align_takeoff_frame,
                                             continuous_source_clock=False,
                                             takeoff_lead=args.takeoff_lead)
        from fabrics_sim.reference.completion import MotionCompletion
        motion_completion = MotionCompletion(len(motion_trajectory)-1)
        args.output_path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(args.output_path.with_suffix(".contact_plan.npz"), modes=candidate_modes,
                            sole_height=heights, foot_speed=speeds, ambiguous=ambiguous, fps=motion_fps,
                            method=contact_plan_method, geometry_method=contact_geometry_method)
        print(f"Experimental contact plan: method={contact_plan_method}, "
              f"modes={np.bincount(candidate_modes,minlength=4).tolist()}, "
              f"geometry ambiguous frames={int(ambiguous.sum())}/{len(ambiguous)}",flush=True)
    if args.motion_start_in_motion and bounded_events is not None:
        from fabrics_sim.reference.bounded_steps import MeasuredSteps
        goals = np.stack([_to_numpy(reference_feet_transforms(
            fabric, motion_trajectory[e['end_frame']:e['end_frame']+1])[0,1-e['support'],:3])
            for e in bounded_events])
        measured_steps = MeasuredSteps(bounded_events, goals, continuous=True, track_source_pose=True)
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        args.output_path.with_suffix('.steps_plan.json').write_text(json.dumps(dict(
            events=bounded_events, landing_positions_world=goals.tolist(), continuous=True,
            landing_goal_policy="source_pose",
            full_motion=args.track_full_motion,
            ends_mid_swing=bool(modes[-1] != 3),
            contact_source=str(args.contacts_pkl or args.motion_pkl),
            contact_plan_verified=False,
            interpolation=args.motion_interpolation,
            original_source_frame_count=args.original_source_frame_count,
            source_positions_modified=False), indent=2)+'\n')
    if args.motion_start_in_motion:
        # Imported pose data has not been certified dynamically feasible.
        # Retain the ordinary task acceleration caps unless explicitly opted in.
        fabric.dynamics_tasks.preserve_reference_acceleration = args.preserve_reference_acceleration
        fabric.dynamics_tasks.source_pelvis_metric_scale = args.source_pelvis_weight
        q, qd, initial_reference_acc = motion_sampler.kinematics(
            float(motion_start_frame), motion_fps * args.motion_speed, 0.)
        WholeBodyReference(q, qd, initial_reference_acc).apply(fabric)
        initial_center_of_mass = fabric.center_of_mass_taskmap.position_target.clone()
        com_balance_target = initial_center_of_mass.clone()
        print(f"Dynamic source start: frame={motion_start_frame}, velocity_norm={float(qd.norm()):.6f}", flush=True)
    qdd_fabric = fabric.compute_acceleration(q, qd)

    (
        mass_np,
        bias_np,
        left_j_np,
        right_j_np,
        isaac_joint_names,
        simulation_app,
        world,
        robot,
        foot_contact_view,
    ) = (
        initialize_isaac_dynamics(
            q,
            qd,
            headless=not args.show_isaac_window,
            enable_self_collisions=False if args.disable_self_collisions else None,
            report_contacts=args.report_self_contacts,
            physics_iterations=args.physics_iterations,
            diagnose_contacts=args.diagnose_contacts,
            motor_armature=args.motor_armature,
            prime_source_contacts=args.motion_start_in_motion,
        )
    )
    if args.motion_start_in_motion:
        q, qd, _, _, _, _ = read_live_isaac_snapshot(robot, isaac_joint_names, device, q.dtype)
        qdd_fabric = fabric.compute_acceleration(q, qd)
    mass = torch.as_tensor(mass_np, device=device, dtype=q.dtype)
    if fabric.dynamics_tasks is not None:
        armature_order = [isaac_joint_names.index(name) for name in G1_JOINT_NAMES]
        fabric.dynamics_tasks.model.validate_mass_matrix(
            q, mass, joint_armature=_to_numpy(robot.get_armatures())[:, armature_order])
        if args.fabric_priority_mode != 'none':
            fabric.dynamics_tasks.priority_joint_armature = torch.as_tensor(
                _to_numpy(robot.get_armatures())[:, armature_order], device=device, dtype=q.dtype)
    self_contact_events = []
    foot_contact_report = []
    contact_report_subscription = None
    if args.report_self_contacts:
        from omni.physx import get_physx_simulation_interface
        from pxr import PhysicsSchemaTools
        reported_pairs = set()
        def report_self_contacts(headers, data):
            foot_contact_report.clear()
            for header in headers:
                actors = [str(PhysicsSchemaTools.intToSdfPath(actor))
                          for actor in (header.actor0, header.actor1)]
                if any(actor.endswith('/'+name) for actor in actors
                       for name in (LEFT_FOOT_BODY,RIGHT_FOOT_BODY)):
                    foot_contact_report.append(dict(step=step,actors=actors,
                        colliders=[str(PhysicsSchemaTools.intToSdfPath(c)) for c in
                                   (header.collider0,header.collider1)],
                        contacts=[dict(position=list(data[i].position),normal=list(data[i].normal),
                                       impulse=list(data[i].impulse),separation=data[i].separation)
                                  for i in range(header.contact_data_offset,
                                                 header.contact_data_offset+header.num_contact_data)]))
                if not all(actor.startswith("/World/G1/") for actor in actors) or not header.num_contact_data:
                    continue
                pair = tuple(sorted(str(PhysicsSchemaTools.intToSdfPath(collider))
                                    for collider in (header.collider0, header.collider1)))
                separation = min(data[i].separation for i in range(
                    header.contact_data_offset, header.contact_data_offset + header.num_contact_data))
                if separation < 0. and pair not in reported_pairs:
                    reported_pairs.add(pair)
                    event = (str(step * PD_TIMESTEP), *pair, str(separation))
                    self_contact_events.append(event)
                    print(f"SELF CONTACT: t={event[0]} pair={pair} separation={separation:.6g}", flush=True)
        contact_report_subscription = get_physx_simulation_interface().subscribe_contact_report_events(report_self_contacts)
    foot_landing_geometry = None
    if args.landing_speed_limit is not None or (args.motion_start_in_motion and bounded_events is not None):
        from omni.usd import get_context
        from fabrics_sim.solver.landing_approach import collision_support_points
        foot_landing_geometry=[tuple(q.new_tensor(a) for a in collision_support_points(
            get_context().get_stage(),f'/World/G1/{link}')) for link in (LEFT_FOOT_BODY,RIGHT_FOOT_BODY)]
        if len(foot_landing_geometry[0][0])!=len(foot_landing_geometry[1][0]):
            raise ValueError('G1 landing diagnostics require matching left/right collider point counts')
        if args.landing_speed_limit is not None:
            print(f'Landing speed envelope: {args.landing_speed_limit:g} m/s; '
                  f'{len(foot_landing_geometry[0][0])} collision points per foot',flush=True)
        if args.motion_start_in_motion and bounded_events is not None:
            plan_path = args.output_path.with_suffix('.steps_plan.json')
            plan = json.loads(plan_path.read_text())
            plan['collision_points'] = [_to_numpy(row[0]).tolist() for row in foot_landing_geometry]
            plan['collision_radii'] = [_to_numpy(row[1]).tolist() for row in foot_landing_geometry]
            plan['clearance_geometry'] = 'enabled USD sphere/cube foot colliders'
            plan_path.write_text(json.dumps(plan,indent=2)+'\n')
    if args.swing_midpoint_clearance:
        from fabrics_sim.reference.swing_clearance import swing_clearance_reference
        original_start = motion_sampler.kinematics(float(motion_start_frame), motion_fps*args.motion_speed, 0.)
        geometry = [(_to_numpy(g[0]),_to_numpy(g[1])) for g in foot_landing_geometry]
        clearance_feasibility = None
        if args.track_full_motion:
            from fabrics_sim.reference.swing_clearance import feasible_swing_clearance_reference
            control_motion_trajectory, clearance_feasibility = feasible_swing_clearance_reference(
                fabric, control_motion_trajectory, G1_JOINT_NAMES, bounded_events,
                args.swing_midpoint_clearance, collision_geometry=geometry)
        else:
            control_motion_trajectory = swing_clearance_reference(
                fabric, control_motion_trajectory, G1_JOINT_NAMES, bounded_events,
                args.swing_midpoint_clearance, collision_geometry=geometry)
        if args.tracking_profile is not None:
            from landing_clearance import feasible_swing_clearance_reference as landing_clearance_reference
            fifth = dict(bounded_events[4], end_frame=362)
            control_motion_trajectory, landing_tail_report = landing_clearance_reference(
                fabric, control_motion_trajectory, G1_JOINT_NAMES, [fifth], .085, collision_geometry=geometry)
            args.output_path.with_suffix('.landing_tail.json').write_text(json.dumps(landing_tail_report, indent=2)+'\n')
            # Later landing transients: retain the source order and all source
            # targets. Extend clearance only inside the existing double-support
            # interval, ending once the original foot slows below 0.2 m/s.
            later_tail_reports = []
            for event_index in range(6, len(bounded_events)):
                event = bounded_events[event_index]
                next_takeoff = (bounded_events[event_index+1]['takeoff_frame']
                                if event_index+1 < len(bounded_events) else len(motion_trajectory)-1)
                end = event['end_frame']
                last = min(end+20, next_takeoff-1, len(motion_trajectory)-2)
                selected = end
                side = 1-event['support']
                link = fabric.dynamics_tasks.model.kinematics.get_link_index(('left','right')[side]+'_ankle_roll_link')
                for sample in range(end+1, last):
                    speeds=[]
                    for probe in (sample, sample+1):
                        source_q, source_v, _ = motion_sampler.kinematics(float(probe), motion_fps, 0.)
                        source_state = fabric.dynamics_tasks.model.snapshot(source_q)
                        speeds.append(float((source_state['origin_jac'][0,link]@source_v[0]).norm()))
                    if max(speeds) <= .2:
                        selected=sample
                        break
                if selected == end:
                    later_tail_reports.append(dict(step=event_index+1, extended=False))
                    continue
                extended = dict(event, end_frame=selected)
                if event_index+1>=args.later_clearance_first_step:
                    from landing_clearance_smooth import feasible_swing_clearance_reference as smooth_landing_clearance
                    control_motion_trajectory, feasibility = smooth_landing_clearance(
                        fabric, control_motion_trajectory, G1_JOINT_NAMES, [extended], args.later_clearance_target,
                        collision_geometry=geometry,fall_fraction=args.later_clearance_fall_fraction)
                else:
                    control_motion_trajectory, feasibility = landing_clearance_reference(
                        fabric, control_motion_trajectory, G1_JOINT_NAMES, [extended], .085, collision_geometry=geometry)
                later_tail_reports.append(dict(step=event_index+1, extended=True,
                    original_end_frame=end, end_frame=selected, feasibility=feasibility))
            args.output_path.with_suffix('.later_tails.json').write_text(json.dumps(later_tail_reports, indent=2)+'\n')

            if args.reference_correction_rmse_limit:
                from bounded_reference_correction import bound_reference_correction
                control_motion_trajectory,correction_report=bound_reference_correction(control_motion_trajectory,motion_trajectory,
                    bounded_events,later_tail_reports,first_step=args.later_clearance_first_step,joint_rmse_limit=args.reference_correction_rmse_limit,first_step_limit=args.first_correction_rmse_limit)
                args.output_path.with_suffix('.reference_correction.json').write_text(json.dumps(correction_report,indent=2)+'\n')
            if args.reference_pose_candidate is not None:
                import hashlib
                with np.load(args.reference_pose_candidate) as candidate:
                    for key,array in [('baseline_sha256',control_motion_trajectory),('source_sha256',motion_trajectory)]:
                        if candidate[key].item()!=hashlib.sha256(_to_numpy(array).tobytes()).hexdigest():
                            raise ValueError('Reference candidate input mismatch: '+key)
                    prepared=candidate['control_q']
                    if prepared.shape!=tuple(control_motion_trajectory.shape) or not np.isfinite(prepared).all() or abs(float(candidate['fps'])-motion_fps)>1e-8:
                        raise ValueError('Invalid reference pose candidate')
                    control_motion_trajectory=control_motion_trajectory.new_tensor(prepared)
        motion_sampler = MotionInterpolator(control_motion_trajectory,
            method='bspline' if args.motion_interpolation == 'bspline-analytic' else 'pchip')
        if args.smooth_stride:
            from blended_spline import BlendedSplineReference
            motion_sampler=BlendedSplineReference(motion_sampler,control_motion_trajectory,stride=args.smooth_stride,
                start_frame=args.smooth_start*motion_fps,ramp_frames=args.smooth_ramp*motion_fps)
        if args.late_reference_smooth_stride:
            from late_tube_smoothing import transition_tube_spline
            motion_sampler=transition_tube_spline(motion_sampler,control_motion_trajectory,
                stride=args.late_reference_smooth_stride,start_frame=args.late_reference_smooth_start*motion_fps,
                ramp_frames=args.late_reference_smooth_ramp*motion_fps)
        if args.joint_correction_overlay is not None:
            from joint_correction_overlay import JointCorrectionOverlay
            motion_sampler=JointCorrectionOverlay(motion_sampler,control_motion_trajectory,motion_trajectory,args.joint_correction_overlay)
        if args.secondary_joint_correction_overlay is not None:
            import hashlib
            if args.joint_correction_overlay is None:raise ValueError('Secondary overlay requires its recorded parent')
            with np.load(args.secondary_joint_correction_overlay) as correction:
                if correction['parent_overlay_sha256'].item()!=hashlib.sha256(args.joint_correction_overlay.read_bytes()).hexdigest():
                    raise ValueError('Secondary correction parent mismatch')
            motion_sampler=JointCorrectionOverlay(motion_sampler,control_motion_trajectory,motion_trajectory,args.secondary_joint_correction_overlay)
        np.savez_compressed(args.output_path.with_suffix('.control_reference.npz'),
            control_q=_to_numpy(control_motion_trajectory),source_q=_to_numpy(motion_trajectory),
            fps=motion_fps,smooth_stride=args.smooth_stride,smooth_start=args.smooth_start,smooth_ramp=args.smooth_ramp)
        adapted_start = motion_sampler.kinematics(float(motion_start_frame), motion_fps*args.motion_speed, 0.)
        if any(not torch.allclose(a,b,atol=2e-5,rtol=0.) for a,b in
               zip(adapted_start,original_start)):
            raise ValueError('Swing clearance correction must preserve the spawned initial q/v/a')
        delta = control_motion_trajectory[:,7:]-motion_trajectory[:,7:]
        report = dict(mid_swing_clearance_m=args.swing_midpoint_clearance,
            per_swing_feasibility=clearance_feasibility,
            peak_joint_rmse_rad=float(delta.square().mean(-1).sqrt().max()),
            max_joint_change_rad=float(delta.abs().max()), original_source_retained_for_audit=True,
            landing_endpoints_preserved=True, pelvis_preserved=True,
            geometry='enabled USD sphere/cube foot colliders',
            adaptation_policy='Fill each source midpoint clearance deficit with one smooth bump',
            changed_source_frames=int((delta.abs().amax(-1)>1e-7).sum()))
        from reference_pose_metadata import reference_pose_metadata
        report.update(reference_pose_metadata(_to_numpy(control_motion_trajectory),_to_numpy(motion_trajectory),G1_JOINT_NAMES,bounded_events))
        if args.reference_pose_candidate is not None:
            report['adaptation_policy']='Explicit prepared full-body reference candidate; original PKL retained for all fidelity checks'
            report['reference_pose_candidate']=str(args.reference_pose_candidate)
        args.output_path.with_suffix('.swing_clearance.json').write_text(json.dumps(report,indent=2)+'\n')
        print('Control reference midpoint swing clearance: '+json.dumps(report),flush=True)
    if args.prepared_contact_plan is not None:
        if not args.direct_wbc or not args.motion_start_in_motion or args.reference_pose_candidate is None or reference_manager is not None:
            raise ValueError('Prepared contact intentions require direct source WBC with an explicit prepared pose reference')
        from prepared_contact_plan import load_prepared_contact_plan
        prepared_modes,contact_report=load_prepared_contact_plan(args.prepared_contact_plan,args.reference_pose_candidate,_to_numpy(motion_planned_modes))
        motion_planned_modes=motion_planned_modes.new_tensor(prepared_modes)
        args.output_path.with_suffix('.prepared_contacts.json').write_text(json.dumps(contact_report,indent=2)+'\n')
        print('Prepared contact intentions: '+json.dumps(contact_report),flush=True)
    prepared_source_clock = None
    if args.prepared_source_clock is not None:
        if (args.tracking_profile is None or args.reference_pose_candidate is None
                or args.clock_fourth_speed is not None or args.clock_fifth_speed is not None or args.contact_pause_end is not None or args.extra_contact_pause
                or args.motion_speed != 1. or motion_start_frame != 0):
            raise ValueError('Prepared source clock requires an unpaused full-source profile at unit motion speed')
        import hashlib
        from smooth_clock import smooth_clock as original_source_clock
        from g1_prepared_clock import PreparedSourceClock
        clock_parameters = (args.clock_transition_start,args.clock_transition_ramp,args.clock_final_speed,
                            args.clock_second_start,args.clock_second_ramp,args.clock_second_speed,
                            args.clock_third_start,args.clock_third_ramp,args.clock_third_speed)
        prepared_source_clock = PreparedSourceClock(args.prepared_source_clock,args.reference_pose_candidate,
            hashlib.sha256(_to_numpy(motion_trajectory).tobytes()).hexdigest(),motion_fps,
            lambda time: original_source_clock(time,*clock_parameters))
        args.output_path.with_suffix('.prepared_clock.json').write_text(json.dumps(prepared_source_clock.report,indent=2)+'\n')
    bias_force = torch.as_tensor(bias_np, device=device, dtype=q.dtype)
    left_j = torch.as_tensor(left_j_np, device=device, dtype=q.dtype)
    right_j = torch.as_tensor(right_j_np, device=device, dtype=q.dtype)
    projector = G1FabricContactProjector(
        num_envs=1,
        device=device,
        dtype=q.dtype,
        dt=FABRIC_TIMESTEP,
        detector_config=ContactDetectorConfig(
            force_on=20.0,
            force_off=10.0,
            confirm_steps=1,
        ),
        projection_config=ProjectionConfig(
            regularization=args.regularization,
            joint_correction_weight=args.projection_joint_correction_weight,
            contact_kp=args.contact_stabilization_kp,
            contact_kd=args.contact_stabilization_kd,
            contact_stabilization_mask=(1.0, 1.0, float(args.reference_manager),
                                        *([1.0] * 3 if args.stabilize_foot_rotation else [0.0] * 3)),
            contact_stabilization_max_acceleration=(
                args.contact_stabilization_max_acceleration
            ),
            filter_alpha_jdot_v=args.jdot_filter_alpha,
        ),
    )
    measured_contact_detector = BatchedFootContactDetector(
        num_envs=1,
        device=device,
        config=ContactDetectorConfig(
            force_on=20.0,
            force_off=10.0,
            confirm_steps=max(1,round(args.contact_switch_confirm_updates*.02/FABRIC_TIMESTEP)) if args.reference_manager else args.contact_switch_confirm_updates,
            impact_force_on=100. if args.motion_start_in_motion else None,
        ),
    )
    if motion_trajectory is not None:
        # The robot is spawned on its two feet. Seed the hysteresis state so
        # the one-sample-delayed PhysX force signal does not erase support.
        measured_contact_detector.contact.fill_(True)
    foot_normal_forces = torch.zeros((1, 2), device=device, dtype=q.dtype)
    contact_anchors = reference_feet_transforms(fabric, q)
    initial_planned_mode = None
    if motion_trajectory is not None:
        initial_planned_mode = motion_planned_modes[
            motion_start_frame:motion_start_frame + 1
        ]
    commanded_contact_mode = (
        torch.full((1,), 3, device=device, dtype=torch.int64)
        if motion_trajectory is not None
        else None
    )
    if args.motion_start_in_motion:
        touching = source_start_contacts(contact_anchors)
        measured_contact_detector.contact.copy_(touching)
        commanded_contact_mode = touching[:, 0].long() + 2 * touching[:, 1].long()
        print(f"Dynamic source start: geometric initial contact mode={commanded_contact_mode.tolist()}; subsequent contacts use measured forces", flush=True)
    contact_ready_counter = torch.zeros(
        1, device=device, dtype=torch.int64
    )
    single_ready_count = 0
    single_release_time = None
    from fabrics_sim.solver.contact_cycle import ContactCycle
    contact_cycle = ContactCycle() if args.contact_cycle_test else None
    single_initial_feet = contact_anchors.clone()
    # The first finite-difference Jdot*v sample is intentionally zero because
    # there is no previous live Jacobian yet.
    zero_bias = torch.zeros((1, 6), device=device, dtype=q.dtype)
    if acceleration_only:
        initial_mode = commanded_contact_mode if commanded_contact_mode is not None else initial_planned_mode
        fabric.set_contact_state(torch.stack(((initial_mode & 1) != 0,
                                             (initial_mode & 2) != 0), dim=1))
        qdd_fabric = fabric.compute_acceleration(q, qd)
    output = projector.step(
        acc_fabric=qdd_fabric,
        mass_matrix=preserve_com_acceleration(mass,
            fabric.dynamics_tasks.last_snapshot['com_jac'], args.com_acceleration_preservation_weight)
            if args.com_acceleration_preservation_weight else mass,
        generalized_velocity=qd,
        left_foot_jacobian=left_j,
        right_foot_jacobian=right_j,
        foot_normal_forces=foot_normal_forces,
        left_jdot_v=zero_bias,
        right_jdot_v=zero_bias,
        left_contact_pose_error=zero_bias,
        right_contact_pose_error=zero_bias,
        contact_mode_override=(
            commanded_contact_mode
            if commanded_contact_mode is not None
            else initial_planned_mode
        ),
    )
    previous_contact_active = torch.stack(
        (output.left_contact, output.right_contact), dim=1
    )
    if fabric.dynamics_tasks is not None:
        fabric.set_contact_state(previous_contact_active)
        if not acceleration_only:
            qdd_fabric = fabric.compute_acceleration(q, qd)
    qdd_dc = qdd_fabric if args.g1_md else output.acc_dc
    contact_jacobian = torch.cat((left_j, right_j), dim=1)
    jdot_qdot = torch.cat((zero_bias, zero_bias), dim=1)
    qp_acceleration_weight = torch.ones(
        35, device=device, dtype=q.dtype
    )
    qp_acceleration_weight[:6] = args.qp_base_acceleration_weight
    qp_acceleration_weight[:3] = initial_linear_weight
    qp_type = G1DynamicsAccelerationQP
    if args.tracking_profile is not None:
        from unload_qp import G1DynamicsAccelerationQP as qp_type
    acceleration_qp = qp_type(
        solver_backend=args.qp_solver,
        heel_length=.05 if args.use_sole_wrench_geometry else .06,
        half_width=.025 if args.use_sole_wrench_geometry else .04,
        max_iter=2000,
        rho=100.0,
        # A 1e-6 wrench weight leaves the acceleration/wrench Hessian
        # poorly conditioned (roughly six orders of magnitude apart) and
        # repeatedly stalls ADMM at its iteration limit during gait contact
        # transitions.  This remains small relative to the acceleration
        # tracking term, while providing enough curvature for a stable solve.
        wrench_weight=1.0e-4,
        acc_weight=preserve_com_acceleration(qp_acceleration_weight,
            fabric.dynamics_tasks.last_snapshot['com_jac'], args.com_acceleration_preservation_weight)
            if args.com_acceleration_preservation_weight else qp_acceleration_weight,
        validation_tolerance=args.qp_validation_tolerance,
        friction_validation_tolerance=1.0,
        effort_validation_tolerance=1.0,
        box_validation_tolerance=1.0,
    )
    if args.qp_reuse_workspace:
        acceleration_qp.solver.reuse_workspace = True
    if args.qp_active_set_predictor:
        acceleration_qp.solver.active_set_predictor = True
    base_qp_solve = acceleration_qp.solve
    if args.tracking_profile == 'amp-full':
        acceleration_qp.solver.single_solve = True
        base_qp_solve = partial(solve_single_tracking_qp, acceleration_qp)
    if args.tracking_tube_soft_weight:
        from g1_soft_tracking_tube import SoftTrackingTubeBackend
        acceleration_qp.solver = SoftTrackingTubeBackend(args.tracking_tube_soft_weight,
            soft_rows=((True, False, False) if args.tracking_tube_soft_joint_only else
                       (True, False, True) if args.tracking_tube_soft_joint_orientation else
                       (True, True, True)))
    acceleration_limits = (
        args.qp_acceleration_limit_scale
        * fabric.accel_limits.to(device=device, dtype=q.dtype)
    )
    if not args.acceleration_and_braking_limits:
        acceleration_limits = torch.full_like(acceleration_limits, float('inf'))
    import xml.etree.ElementTree as ET
    urdf_joint_limits = {node.attrib['name']:node.find('limit')
                         for node in ET.parse(fabric.urdf_path).getroot().findall('joint')}
    joint_velocity_limits = q.new_tensor([float(urdf_joint_limits[name].attrib['velocity'])
                                         for name in G1_JOINT_NAMES])
    nominal_kp, nominal_kd = unitree_g1_pd_gains(device)
    kp_batch, kd_batch = explicit_pd_gains(
        mass, nominal_kp, nominal_kd, PD_TIMESTEP, bandwidth=args.servo_bandwidth
    )
    kp, kd = kp_batch[0], kd_batch[0]
    if args.direct_wbc and not args.tracking_servo:
        kp = torch.zeros_like(kp)
    effort_limits = torch.as_tensor(
        unitree_g1_effort_limits(),
        device=device,
        dtype=q.dtype,
    )
    # Reserve part of each motor's capacity for the 500 Hz feedback term.
    legacy_effort_limits=effort_limits.clone()
    qp_effort_limits = 0.8 * legacy_effort_limits
    if math.isfinite(args.model_effort_start):
        if args.model_effort_start<0 or not math.isfinite(args.model_effort_ramp) or args.model_effort_ramp<=0:
            raise ValueError('Invalid model effort transition')
        from g1_model_effort import model_effort_limits
        simulator_effort_limits=_to_numpy(robot.get_max_efforts(joint_names=list(G1_JOINT_NAMES)))
        effort_limits=q.new_tensor(model_effort_limits(fabric.urdf_path,G1_JOINT_NAMES,simulator_effort_limits))
        if bool((legacy_effort_limits>effort_limits+1e-5).any()):
            raise ValueError('Legacy cap exceeds the active model; refuse transition')
        print('Active model effort limits in Fabric joint order:',effort_limits.tolist(),flush=True)
        args.output_path.with_suffix('.effort_model.json').write_text(json.dumps(dict(
            joint_names=list(G1_JOINT_NAMES),urdf=str(fabric.urdf_path),
            simulator_limits=np.asarray(simulator_effort_limits).tolist(),
            model_limits=effort_limits.tolist(),legacy_software_caps=legacy_effort_limits.tolist(),
            transition_source_time=args.model_effort_start,ramp_source_time=args.model_effort_ramp,
            qp_fraction=.8,simulator_limits_modified=False),indent=2)+'\n')
    initial_feet = reference_feet_transforms(fabric, q)
    initial_friction_G, initial_friction_b = (
        build_single_support_wrench_constraints(
            contact_mode=output.contact_mode,
            foot_rotation_world=_quat_xyzw_to_matrix(
                initial_feet[..., 3:7]
            ),
            friction_coefficient=acceleration_qp.friction_coefficient,
            yaw_friction_coefficient=(
                acceleration_qp.yaw_friction_coefficient
            ),
            toe_length=acceleration_qp.toe_length,
            heel_length=acceleration_qp.heel_length,
            half_width=acceleration_qp.half_width,
            sole_depth=.035 if args.use_sole_wrench_geometry else 0.,
            coupled_yaw=args.reference_manager,
            maximum_normal_force=(
                acceleration_qp.maximum_normal_force
            ),
        )
    )
    composed_policy = None
    if args.g1_md:
        from fabrics_sim.solver.fabric_mit import compose_task_policy
        qdd_dc, composed_policy = compose_task_policy(
            fabric.dynamics_tasks.last_requests, qd,
            fabric.dynamics_tasks.config["regularization"])
    initial_qp_reference = qdd_dc
    if args.qp_fabric_metric:
        acceleration_qp.acc_weight = fabric.dynamics_tasks.last_snapshot['fabric_acc_metric']
        initial_qp_reference = qdd_fabric
    if args.qp_com_feedback_scale:
        acceleration_qp.acc_weight, initial_qp_reference = track_com_feedback(
            acceleration_qp.acc_weight, initial_qp_reference, fabric.dynamics_tasks.last_snapshot,
            com_feedback_scale(args.qp_com_feedback_scale, 0., start=args.qp_com_feedback_start,
                               ramp=args.qp_com_feedback_ramp), axes=args.qp_com_feedback_axes)
    qp_output = base_qp_solve(
        acc_reference=initial_qp_reference,
        mass_matrix=mass,
        bias_force=bias_force,
        contact_jacobian=contact_jacobian,
        jdot_qdot=jdot_qdot,
        contact_mode=output.contact_mode,
        acc_lower=-acceleration_limits,
        acc_upper=acceleration_limits,
        effort_limits=qp_effort_limits,
        known_joint_torque=-kd*qd[:,6:] if args.direct_wbc and not args.tracking_servo else None,
        acceleration_constraints=joint_velocity_constraints(qd[:,6:],joint_velocity_limits,FABRIC_TIMESTEP)
            if args.joint_velocity_limits else None,
        friction_G=initial_friction_G,
        friction_b=initial_friction_b,
        task_requests=(None if acceleration_only else [composed_policy] if args.g1_md else
                    None if fabric.dynamics_tasks is None else fabric.dynamics_tasks.last_requests),
        task_tracks_predicted=not args.task_track_nominal,
        priority_tasks=('center_of_mass',) if args.g1_com_priority else (),
        priority_requests=[r for r in fabric.dynamics_tasks.last_requests
                           if r.name == 'center_of_mass'] if args.g1_com_priority and not acceleration_only else None,
        predicted_acc_limit=acceleration_limits if args.limit_predicted_acceleration else None,
    )
    initial_speed_ok = (not args.joint_velocity_limits or bool(
        ((qd[:,6:]+FABRIC_TIMESTEP*qp_output.predicted_acc[:,6:]).abs() <= joint_velocity_limits+1e-4).all()))
    if not bool(qp_output.validated.all()) or not initial_speed_ok:
        raise RuntimeError(
            "Acceleration QP failed validation: "
            f"converged={qp_output.converged.tolist()}, "
            f"base_error={qp_output.base_dynamics_error.tolist()}, "
            "joint_dynamics_error="
            f"{qp_output.joint_dynamics_error.tolist()}, "
            "contact_error="
            f"{qp_output.contact_acceleration_error.tolist()}, "
            f"friction_violation={qp_output.friction_violation.tolist()}, "
            f"effort_violation={qp_output.effort_violation.tolist()}."
        )
    qdd_dynamics_feasible = qp_output.acc
    joint_torque_feedforward = qp_output.joint_torque
    initial_joint_feedforward = joint_torque_feedforward.clone()
    # Seed the command integrator once from the initial measured joint state.
    # The first loop update at t=0 performs the first integration step; do
    # not advance it here as well. Never integrate the unactuated base.
    joint_velocity = qd[:, 6:35]
    joint_velocity_desired = joint_velocity.clone()
    joint_position = q[:, 7:36]
    joint_position_desired = joint_position.clone()
    integration_initial_position = joint_position_desired.clone()
    integration_initial_velocity = joint_velocity_desired.clone()

    # MIT-style command: inverse-dynamics feedforward plus joint-space PD.
    joint_torque_pd = (
        torch.zeros_like(joint_velocity) if args.tracking_servo else -kd*joint_velocity if args.direct_wbc else
        kp * (joint_position_desired - joint_position)
        + kd * (joint_velocity_desired - joint_velocity)
    )
    joint_torque_unclipped = joint_torque_feedforward + joint_torque_pd
    joint_torque = torch.clamp(
        joint_torque_unclipped,
        min=-effort_limits,
        max=effort_limits,
    )

    # Send the 29-D effort command in Isaac DOF order. Physics and joint PD run
    # at 500 Hz; the complete state/dynamics/FABRICS/QP calculation is refreshed
    # at the selected control rate (every physics step at the 500 Hz default).
    fabric_to_isaac_joint = torch.tensor(
        [G1_JOINT_NAMES.index(name) for name in isaac_joint_names],
        device=device,
        dtype=torch.long,
    )
    position_desired_isaac = joint_position_desired[
        :, fabric_to_isaac_joint
    ]
    velocity_desired_isaac = joint_velocity_desired[
        :, fabric_to_isaac_joint
    ]
    kp_isaac = kp[fabric_to_isaac_joint]
    kd_isaac = kd[fabric_to_isaac_joint]
    effort_limits_isaac = effort_limits[fabric_to_isaac_joint]
    limit_order = torch.argsort(fabric_to_isaac_joint)
    joint_position_limits = torch.as_tensor(
        _to_numpy(robot.get_dof_limits()), device=device, dtype=q.dtype
    )[:, limit_order]
    if motion_trajectory is not None and args.left_shoulder_roll_offset:
        shoulder = G1_JOINT_NAMES.index("left_shoulder_roll_joint")
        values = motion_trajectory[:, 7 + shoulder]
        if ((values < joint_position_limits[0, shoulder, 0]) |
                (values > joint_position_limits[0, shoulder, 1])).any():
            raise ValueError("Shoulder clearance offset exceeds simulator joint limits in the reference trajectory.")
    # Conservative command-speed ceiling for this tracking experiment.
    joint_command_speed_limit = torch.full_like(joint_position_desired, 4.0)
    # Gains above belong to our explicit PD calculation, not PhysX drives.
    # Keep the internal drive stiffness/damping zero throughout the run.
    zero_drive_gains = np.zeros((1, len(isaac_joint_names)), dtype=np.float32)
    robot.set_gains(
        kps=zero_drive_gains,
        kds=zero_drive_gains,
    )
    drive_kp, drive_kd = robot.get_gains()
    if np.any(_to_numpy(drive_kp) != 0.0) or np.any(_to_numpy(drive_kd) != 0.0):
        raise RuntimeError("PhysX joint drives must be disabled for explicit torque control.")
    print(f"actuation: explicit {1./PD_TIMESTEP:g} Hz PD + QP feedforward; PhysX drive gains=0")
    torque_feedforward_isaac = joint_torque_feedforward[
        :, fabric_to_isaac_joint
    ]
    physics_steps = int(round(args.duration / PD_TIMESTEP))
    joint_torque_isaac_tensor = torch.zeros(
        (1, len(G1_JOINT_NAMES)),
        device=device,
        dtype=q.dtype,
    )
    fabric_steps_per_update = int(round(FABRIC_TIMESTEP / PD_TIMESTEP))
    render_steps_per_update = max(1, int(round(world.get_rendering_dt() / PD_TIMESTEP)))
    initial_physics_time = world.current_time
    if fabric_steps_per_update < 1:
        raise RuntimeError("FABRIC_TIMESTEP must be at least PD_TIMESTEP.")
    executed_steps = 0
    contact_sample_dt = PD_TIMESTEP
    stopped_at_motion_end = False
    stopped_at_tracking_steps = False
    stopped_at_unload_end = False
    transfer_reference = None
    unload_bridge = None
    from fabrics_sim.reference.mocap_transition import MocapContactTransition
    mocap_transition = MocapContactTransition(fabric, G1_JOINT_NAMES) if args.g1_md and args.mocap_pipeline else None
    from fabrics_sim.reference.contact_transfer import TransferStartGate
    transfer_start_gate = TransferStartGate()
    source_step_rollout = None
    probe_unload_report = None
    source_step_start = None
    support_switch = None
    source_step_number = 0
    direct_unload_event = None
    direct_unload_deadline_ramp = .05
    direct_unload_started = 0.
    source_steps_finished = False
    pending_source_event = None
    stopped_at_source_step_end = False
    control_updates = 0
    phase_frame_position = float(motion_start_frame)
    previous_wall_motion_time = 0.0
    phase_hold_updates = 0
    reported_nonconvergence = False
    torque_continuity_solver = None
    tracking_tube_solver = acceleration_qp.solver
    if math.isfinite(args.torque_continuity_start):
        from g1_torque_continuity import TorqueContinuitySolver
        torque_continuity_solver = TorqueContinuitySolver(acceleration_qp.solver)
        acceleration_qp.solver = torque_continuity_solver
    tracking_history = []
    servo_history = []
    fidelity_stop=None
    for step in range(physics_steps):
        if not simulation_app.is_running():
            break

        if step % fabric_steps_per_update == 0:
            (
                q,
                qd,
                mass,
                bias_force,
                left_j,
                right_j,
            ) = read_live_isaac_snapshot(
                robot,
                isaac_joint_names,
                device,
                q.dtype,
            )

            net_foot_forces = torch.as_tensor(
                _to_numpy(foot_contact_view.get_net_contact_forces(dt=contact_sample_dt)),
                device=device,
                dtype=q.dtype,
            )
            if net_foot_forces.shape != (2, 3):
                raise RuntimeError(
                    "Expected left/right net foot forces [2,3], got "
                    f"{tuple(net_foot_forces.shape)}."
                )
            # Support must come from the ground, not foot-foot/body collisions.
            ground_contact_forces = foot_contact_view.get_contact_force_matrix(dt=contact_sample_dt)
            ground_foot_forces = torch.as_tensor(
                ground_support_forces(_to_numpy(ground_contact_forces)), device=device, dtype=q.dtype)
            foot_normal_forces = ground_foot_forces[:, 2].clamp_min(0.).reshape(1, 2)
            contact_data = (foot_contact_view.get_contact_force_data(dt=contact_sample_dt)
                            if args.diagnose_contacts else None)

            com_replan_diagnostic = (dict(centroidal_replan_attempted=False,
                centroidal_replan_accepted=False,centroidal_replan_reason='')
                if None is not None else {})

            control_time = control_updates * FABRIC_TIMESTEP
            elapsed_motion_time = control_time
            if motion_trajectory is not None:
                warming_up = args.fixed_joint_test or args.weight_transfer_test or bool(args.single_support_test) or control_time < args.motion_warmup_duration
                wall_motion_time = max(0.0, control_time - args.motion_warmup_duration)
                elapsed_motion_time = ramped_motion_time(wall_motion_time, args.motion_ramp_duration)
                previous_motion_time = ramped_motion_time(wall_motion_time-FABRIC_TIMESTEP, args.motion_ramp_duration)
                next_motion_time = ramped_motion_time(wall_motion_time+FABRIC_TIMESTEP, args.motion_ramp_duration)
                clock_rate, clock_acceleration = 1., 0.
                if args.tracking_profile is not None:
                    from smooth_clock import smooth_clock
                    if prepared_source_clock is not None:
                        original_speed_clock = smooth_clock
                        def smooth_clock(time,*parameters,base=original_speed_clock):
                            return prepared_source_clock.sample(time,base(time,*parameters))
                    if args.clock_fourth_speed is not None:
                        from g1_late_clock import late_speed_transition
                        if args.clock_third_speed is None or args.clock_fourth_start<args.clock_third_start+args.clock_third_ramp:
                            raise ValueError('Fourth source-clock transition must follow a completed third transition')
                        original_speed_clock=smooth_clock
                        def smooth_clock(time,*parameters,base=original_speed_clock):
                            return late_speed_transition(base(time,*parameters),time,
                                start=args.clock_fourth_start,ramp=args.clock_fourth_ramp,
                                previous_speed=args.clock_third_speed,speed=args.clock_fourth_speed)
                    if args.clock_fifth_speed is not None:
                        from g1_late_clock import late_speed_transition
                        if args.clock_fourth_speed is None or args.clock_fifth_start<args.clock_fourth_start+args.clock_fourth_ramp:
                            raise ValueError('Fifth source-clock transition must follow a completed fourth transition')
                        fourth_speed_clock=smooth_clock
                        def smooth_clock(time,*parameters,base=fourth_speed_clock):
                            return late_speed_transition(base(time,*parameters),time,
                                start=args.clock_fifth_start,ramp=args.clock_fifth_ramp,
                                previous_speed=args.clock_fourth_speed,speed=args.clock_fifth_speed)
                    if args.contact_pause_end is not None:
                        from landing_pause_clock import LandingPauseClock
                        if args.contact_pause_early_resume:
                            from early_resume_clock import EarlyResumeLandingClock as LandingPauseClock
                        if 'source_contact_pause' not in locals():
                            source_contact_pause=LandingPauseClock(args.contact_pause_start,args.contact_pause_end,args.contact_pause_resume,args.contact_pause_landings)
                        speed_clock=smooth_clock
                        input_phase=speed_clock(wall_motion_time,args.clock_transition_start,args.clock_transition_ramp,args.clock_final_speed,args.clock_second_start,args.clock_second_ramp,args.clock_second_speed,args.clock_third_start,args.clock_third_ramp,args.clock_third_speed)[0]
                        source_contact_pause.update(input_phase,len(measured_steps.landings) if measured_steps is not None else 0)
                        def smooth_clock(time,*parameters):
                            phase,rate,acceleration=speed_clock(time,*parameters)
                            out,phase_rate,phase_acceleration=source_contact_pause.sample(phase)
                            return out,phase_rate*rate,phase_acceleration*rate**2+phase_rate*acceleration
                    if args.extra_contact_pause:
                        from g1_landing_clock_chain import LandingClockChain
                        if 'extra_contact_clocks' not in locals():
                            extra_contact_clocks = LandingClockChain(args.extra_contact_pause, args.extra_contact_pause_resume)
                        before_extra_clock = smooth_clock
                        extra_input = before_extra_clock(wall_motion_time,args.clock_transition_start,args.clock_transition_ramp,args.clock_final_speed,args.clock_second_start,args.clock_second_ramp,args.clock_second_speed,args.clock_third_start,args.clock_third_ramp,args.clock_third_speed)
                        extra_contact_clocks.update(extra_input,len(measured_steps.landings) if measured_steps is not None else 0)
                        def smooth_clock(time,*parameters,base=before_extra_clock):
                            return extra_contact_clocks.sample(base(time,*parameters))
                    elapsed_motion_time,clock_rate,clock_acceleration=smooth_clock(wall_motion_time,args.clock_transition_start,args.clock_transition_ramp,args.clock_final_speed,args.clock_second_start,args.clock_second_ramp,args.clock_second_speed,args.clock_third_start,args.clock_third_ramp,args.clock_third_speed)
                    previous_motion_time=smooth_clock(wall_motion_time-FABRIC_TIMESTEP,args.clock_transition_start,args.clock_transition_ramp,args.clock_final_speed,args.clock_second_start,args.clock_second_ramp,args.clock_second_speed,args.clock_third_start,args.clock_third_ramp,args.clock_third_speed)[0]
                    next_motion_time=smooth_clock(wall_motion_time+FABRIC_TIMESTEP,args.clock_transition_start,args.clock_transition_ramp,args.clock_final_speed,args.clock_second_start,args.clock_second_ramp,args.clock_second_speed,args.clock_third_start,args.clock_third_ramp,args.clock_third_speed)[0]
                if args.fixed_joint_test or args.weight_transfer_test or args.single_support_test:
                    elapsed_motion_time = 0.0
                phase_held = False
                pending_phase_mode = None
                preview_release_ready = True
                preview_release_reason = ""
                frame_position = min(
                    motion_start_frame
                    + elapsed_motion_time * motion_fps * args.motion_speed,
                    motion_trajectory.shape[0] - 1,
                )
                if reference_manager is not None:
                    from fabrics_sim.reference.balance import support_balance_ready
                    observed_feet = reference_feet_transforms(fabric,q)
                    observed_rotations = _quat_xyzw_to_matrix(observed_feet[...,3:])
                    observed_corners = observed_feet[...,:3].unsqueeze(-2)+(observed_rotations.unsqueeze(-3)@corners[...,None]).squeeze(-1)
                    sole_clearance = observed_corners[...,2].amin(-1)[0].detach().cpu().numpy()
                    actual_com,actual_com_jac = fabric.center_of_mass_taskmap._position_and_jacobian(q)
                    actual_com_velocity=(actual_com_jac@qd[...,None]).squeeze(-1)
                    # No capture-point/LIPM gate in direct mocap tracking.
                    # None omits only the optional balance heuristic; the manager
                    # still checks measured support and unloading before release.
                    balance_ready = None if args.mocap_pipeline else support_balance_ready(actual_com[0].detach().cpu().numpy(), actual_com_velocity[0].detach().cpu().numpy(), observed_feet[0].detach().cpu().numpy(), max_speed=0.06)
                    if mocap_transition is not None and reference_manager.mode == 3:
                        balance_ready = [False, False]
                        if mocap_transition.transfer is not None and mocap_transition.ready(
                                control_time, actual_com, actual_com_velocity, q):
                            balance_ready[mocap_transition.transfer.support] = True
                    if args.mocap_unload_test:
                        # Keep both constraints: this experiment measures unloading only.
                        balance_ready = [False, False]
                        if (args.mocap_first_step_report is None and transfer_reference is not None
                                and control_time >= transfer_reference.start_time+3.5):
                            stopped_at_unload_end = True
                            frame_position = managed_clock.frame
                            print("Double-support unloading diagnostic completed", flush=True)
                            break
                    if (transfer_reference is not None and transfer_reference.release_time is None
                            and not transfer_reference.ready_for_release(control_time,actual_com,actual_com_velocity,q)):
                        balance_ready[transfer_reference.support] = False
                    measured_left, measured_right, measured_mode = measured_contact_detector.update(foot_normal_forces)
                    measured_contacts = torch.stack((measured_left,measured_right),dim=1)
                    try:
                        # This diagnostic holds the first lift; source coverage remains at takeoff.
                        # Before release, the unchanged manager verifies measured load and balance.
                        if transfer_reference is None or transfer_reference.release_time is None:
                            managed_clock = reference_manager.step(foot_normal_forces[0].cpu().numpy(),
                                measured_contacts[0].cpu().numpy(),FABRIC_TIMESTEP,warming_up,sole_clearance,balance_ready,
                                sole_max_height=_to_numpy(observed_corners[...,2].amax(-1)[0]) if args.early_touchdown else None,
                                touchdown_contact=loaded_sole_contacts(observed_feet,tuple(map(_to_numpy,contact_data)))
                                    if patch_manager is not None else None)
                            if (transfer_reference is not None and managed_clock.mode != 3
                                    and (args.coordinated_source_start or args.unified_source_step_test)):
                                from fabrics_sim.reference.source_sequence import hold_takeoff_clock
                                managed_clock = hold_takeoff_clock(managed_clock,reference_manager.takeoff_frame)
                        else:
                            measured_support = (bool(measured_contacts[0, transfer_reference.support])
                                if source_step_rollout is None else source_step_rollout.has_measured_support(
                                    measured_contacts[0].cpu().numpy()))
                            if not measured_support:
                                raise RuntimeError("Whole-body transfer lost its measured support")
                            from dataclasses import replace
                            managed_clock = replace(managed_clock, previous_frame=managed_clock.frame,
                                                    next_frame=managed_clock.frame, rate=0.)
                            if support_switch is not None:
                                mode = support_switch.update(control_time,_to_numpy(foot_normal_forces[0]),
                                    _to_numpy(measured_contacts[0]),actual_com,actual_com_velocity,q)
                                managed_clock = replace(managed_clock,mode=mode,support_mode=support_switch.support+1,phase="SUPPORT_SWITCH")
                            elif source_step_rollout is not None:
                                elapsed = control_time-source_step_start
                                step_mode = source_step_rollout.contact_mode(_to_numpy(foot_normal_forces[0]),
                                    sole_clearance, _to_numpy(observed_corners[...,2].amax(-1)[0]))
                                if elapsed > source_step_rollout.duration+.5*FABRIC_TIMESTEP:
                                    if not source_step_rollout.touchdown:
                                        raise RuntimeError("Source step reached its endpoint without confirmed touchdown")
                                    if getattr(source_step_rollout,"is_tail",False):
                                        source_steps_finished = True
                                    elif source_step_number >= args.source_step_count and (args.source_step_count > 1 or args.source_contact_events):
                                        source_steps_finished = True
                                    elif args.source_support_switch:
                                        from fabrics_sim.reference.source_sequence import next_source_event
                                        pending_source_event = None
                                        if args.source_contact_events and not source_recovery_active(source_step_number,
                                                args.source_recovery_after_step,args.source_recovery_step_count):
                                            pending_source_event = next_source_event(_to_numpy(motion_planned_modes),
                                                source_step_rollout.end_frame)
                                        if pending_source_event is not None and pending_source_event['kind']=='tail':
                                            from fabrics_sim.reference.source_tail import DoubleSupportSourceRollout
                                            from g1_armature import unitree_armatures
                                            source_step_rollout = DoubleSupportSourceRollout(fabric,
                                                source_step_rollout.references[-1],G1_JOINT_NAMES,control_motion_trajectory,
                                                pending_source_event['start_frame'],pending_source_event['end_frame'],
                                                source_step_rollout.support,FABRIC_TIMESTEP,
                                                q.new_tensor(unitree_armatures(G1_JOINT_NAMES)),
                                                args.output_path.with_suffix('.source_tail_plan.json'),
                                                dict(lower=joint_position_limits[:,:,0],upper=joint_position_limits[:,:,1],
                                                    dt=FABRIC_TIMESTEP,bandwidth=args.joint_limit_barrier_bandwidth,
                                                    velocity_damper_time=args.joint_limit_time_constant,
                                                    impact_recovery=args.joint_limit_impact_recovery),
                                                acceleration_limits,duration=args.source_tail_duration)
                                            source_step_start=control_time
                                            elapsed=0.
                                            step_mode=3
                                            print("Final source tail started: "+json.dumps(pending_source_event),flush=True)
                                        else:
                                            from fabrics_sim.reference.support_switch import SupportSwitchReference
                                            next_support=(1-source_step_rollout.support if pending_source_event is None
                                                else pending_source_event['support'])
                                            support_switch = SupportSwitchReference(fabric,source_step_rollout.references[-1],
                                                G1_JOINT_NAMES,observed_feet,next_support,control_time,FABRIC_TIMESTEP,
                                                dict(lower=joint_position_limits[:,:,0],upper=joint_position_limits[:,:,1],
                                                    dt=FABRIC_TIMESTEP,bandwidth=args.joint_limit_barrier_bandwidth,
                                                    velocity_damper_time=args.joint_limit_time_constant,
                                                    impact_recovery=args.joint_limit_impact_recovery),
                                                com_height_offset=.01 if source_step_number==1 else args.later_transfer_height_offset,
                                                lift_height=.04 if source_step_number==1 else args.later_transfer_lift_height,
                                                pelvis_roll_offset=(args.recovery_left_transfer_roll_offset
                                                    if source_step_rollout.fixed_source_frame is not None and next_support==0 else 0.))
                                            switch_report=dict(support_switch.transfer.preflight,source_event=pending_source_event)
                                            args.output_path.with_suffix(".switch_plan.json" if source_step_number==1
                                                else f".switch_{source_step_number}_plan.json").write_text(json.dumps(switch_report,indent=2)+"\n")
                                            print(f"Support switch started: time={control_time:.3f}, support={support_switch.support}",flush=True)
                                    elif args.coordinated_source_start:
                                        raise RuntimeError("Landing reference stopped but the next load-transfer/lift trajectory is not planned"
                                            if source_step_rollout.stop_duration else
                                            "First landing reached but the next load-transfer trajectory is not planned")
                                    else:
                                        stopped_at_source_step_end = True
                                        print(f"First source step reached frame {source_step_rollout.end_frame} with measured touchdown",flush=True)
                                        break
                                source_frame = source_step_rollout.frame(elapsed)
                                previous_frame = source_step_rollout.frame(max(0.,elapsed-FABRIC_TIMESTEP))
                                next_frame = source_step_rollout.frame(elapsed+FABRIC_TIMESTEP)
                                managed_clock = replace(managed_clock,frame=source_frame,previous_frame=previous_frame,
                                    next_frame=next_frame,rate=(next_frame-previous_frame)/(2*FABRIC_TIMESTEP*motion_fps),
                                    mode=step_mode,support_mode=source_step_rollout.support+1,phase=("SOURCE_STEPS_HOLD" if source_steps_finished else "LANDING_STOP" if elapsed > source_step_rollout.swing_duration
                                        else "SOURCE_LANDING" if step_mode==3 else "SOURCE_SWING"))
                    except (ValueError,RuntimeError) as exc:
                        save_reference_failure(args,exc,control_time,q,qd,foot_normal_forces,
                                               tracking_history,servo_history)
                        raise
                    frame_position = managed_clock.frame
                    phase_held = managed_clock.rate < 1e-5 and not warming_up
                    commanded_contact_mode.fill_(managed_clock.mode)
                    phase_hold_updates += int(phase_held)
                    source_complete = motion_completion.update(
                        frame_position, managed_clock.mode,
                        reference_manager.settling and not getattr(source_step_rollout,"is_tail",False),
                        foot_normal_forces[0].detach().cpu().numpy(),
                        actual_com_velocity[0].detach().cpu().numpy(), FABRIC_TIMESTEP)
                    if measured_steps is not None:
                        measured_steps.update(frame_position, control_time, int(measured_mode.item()),
                            _to_numpy(foot_normal_forces[0]), sole_clearance,
                            _to_numpy(observed_feet[0,:,:3]), FABRIC_TIMESTEP)
                        source_complete = source_complete and measured_steps.complete
                    if args.until_motion_end and source_complete:
                        stopped_at_motion_end = True
                        print(f"Source endpoint and stable final support confirmed at {control_time:.2f}s",flush=True)
                        break
                if reference_manager is None and args.require_support_load and not warming_up:
                    phase_frame_position, pending_phase_mode, phase_held = advance_contact_phase(
                        phase_frame_position,
                        max(0., elapsed_motion_time - previous_wall_motion_time) * motion_fps * args.motion_speed,
                        motion_planned_modes, int(commanded_contact_mode.item()),
                    )
                    frame_position = phase_frame_position
                    phase_hold_updates += int(phase_held)
                previous_wall_motion_time = elapsed_motion_time
                motion_frame = int(round(frame_position))
                reference_q = sample_motion(frame_position)
                audit_reference_q = interpolate_motion_trajectory(motion_trajectory, frame_position)
                if 'tube_source_sampler' not in locals():
                    from fabrics_sim.reference.interpolation import MotionInterpolator
                    tube_source_sampler=MotionInterpolator(motion_trajectory,method=args.tube_reference_method)
                    if args.tube_smooth_stride:
                        from blended_spline import BlendedSplineReference
                        tube_source_sampler=BlendedSplineReference(tube_source_sampler,motion_trajectory,stride=args.tube_smooth_stride,
                            start_frame=args.smooth_start*motion_fps,ramp_frames=args.smooth_ramp*motion_fps)
                    if args.late_tube_smooth_stride:
                        from late_tube_smoothing import transition_tube_spline
                        tube_source_sampler=transition_tube_spline(tube_source_sampler,motion_trajectory,
                            stride=args.late_tube_smooth_stride,start_frame=args.late_tube_smooth_start*motion_fps,
                            ramp_frames=args.late_tube_smooth_ramp*motion_fps)
                previous_frame_stride = (elapsed_motion_time-previous_motion_time)*motion_fps*args.motion_speed
                next_frame_stride = (next_motion_time-elapsed_motion_time)*motion_fps*args.motion_speed
                previous_reference_q = (
                    None
                    if warming_up or phase_held or frame_position <= motion_start_frame
                    else sample_motion(
                        max(
                            float(motion_start_frame),
                            frame_position - previous_frame_stride,
                        ),
                    )
                )
                next_reference_q = (
                    None
                    if warming_up or phase_held
                    or frame_position >= motion_trajectory.shape[0] - 1
                    else sample_motion(
                        min(
                            float(motion_trajectory.shape[0] - 1),
                            frame_position + next_frame_stride,
                        ),
                    )
                )
                if managed_clock is not None:
                    previous_reference_q = None if warming_up else sample_motion(managed_clock.previous_frame)
                    next_reference_q = None if warming_up else sample_motion(managed_clock.next_frame)
                reference_kinematics = None
                if args.motion_interpolation in ('pchip-analytic', 'bspline-analytic') and fabric.dynamics_tasks is not None:
                    if warming_up:
                        frame_rate = frame_acceleration = 0.
                    elif args.motion_start_in_motion:
                        # A source clip starts with nonzero velocity. Clipping a
                        # fictitious previous time to zero would halve this rate
                        # and inject a spurious acceleration on the first tick.
                        # A full clip may end in mid-swing: retain its terminal
                        # derivative while auditing the final sample.
                        frame_rate = motion_fps * args.motion_speed * clock_rate if (
                            args.track_full_motion or frame_position < motion_trajectory.shape[0]-1) else 0.
                        frame_acceleration = motion_fps * args.motion_speed * clock_acceleration
                    elif managed_clock is not None:
                        frame_rate = (managed_clock.next_frame-managed_clock.previous_frame)/(2*FABRIC_TIMESTEP)
                        frame_acceleration = (managed_clock.next_frame-2*frame_position+managed_clock.previous_frame)/FABRIC_TIMESTEP**2
                    else:
                        frame_rate = 0. if phase_held else (previous_frame_stride+next_frame_stride)/(2*FABRIC_TIMESTEP)
                        frame_acceleration = 0. if phase_held else (next_frame_stride-previous_frame_stride)/FABRIC_TIMESTEP**2
                    _,reference_v,reference_a = motion_sampler.kinematics(frame_position,frame_rate,frame_acceleration)
                    reference_kinematics = (reference_v,reference_a)
                if unified_reference:
                    whole_body_reference = WholeBodyReference(reference_q, reference_v, reference_a)
                    stopped_for_transfer = (managed_clock is not None and managed_clock.phase == "PREPARE_STEP" and phase_held
                        and reference_v.abs().max() < 1e-4 and reference_a.abs().max() < 1e-3)
                    measured_transfer_ready = (transfer_start_gate.update(stopped_for_transfer,
                        actual_com_velocity, qd[:, 6:], foot_normal_forces, FABRIC_TIMESTEP)
                        if args.mocap_unload_test and transfer_reference is None else True)
                    try:
                        if ((args.unified_transfer_test or args.coordinated_source_start or args.mocap_unload_test) and transfer_reference is None
                                and measured_transfer_ready
                                and managed_clock.phase == "PREPARE_STEP" and phase_held
                                and reference_v.abs().max() < 1e-4 and reference_a.abs().max() < 1e-3):
                            from fabrics_sim.reference.contact_transfer import ContactTransferReference
                            support = int(reference_manager.modes[reference_manager.pending_takeoff])-1
                            transfer_initial = (WholeBodyReference(q.clone(), torch.zeros_like(qd), torch.zeros_like(qd))
                                                if args.mocap_unload_test else whole_body_reference)
                            transfer_reference = ContactTransferReference(
                                fabric, transfer_initial, G1_JOINT_NAMES, observed_feet, support, control_time,
                                preflight_dt=FABRIC_TIMESTEP,
                                joint_bounds=dict(lower=joint_position_limits[:, :, 0],
                                    upper=joint_position_limits[:, :, 1],dt=FABRIC_TIMESTEP,
                                    bandwidth=args.joint_limit_barrier_bandwidth,
                                    velocity_damper_time=args.joint_limit_time_constant,
                                    impact_recovery=args.joint_limit_impact_recovery) if args.acceleration_and_braking_limits else None)
                            if args.mocap_unload_test:
                                from fabrics_sim.reference.bridge import WholeBodyReferenceBridge
                                unload_bridge = WholeBodyReferenceBridge(
                                    WholeBodyReference(q.clone(), qd.clone(), torch.zeros_like(qd)),
                                    transfer_reference.sample(control_time, 3), .5)
                                transfer_reference.preflight.update(initial_state="measured q and v; initial acceleration assumed zero",
                                    scope="double_support_unloading_only", bridge_dynamics_certified=False,
                                    measured_settle_seconds=transfer_start_gate.elapsed)
                            args.output_path.parent.mkdir(parents=True, exist_ok=True)
                            args.output_path.with_suffix('.transfer_plan.json').write_text(
                                json.dumps(transfer_reference.preflight, indent=2)+"\n")
                            print("Transfer reference preflight: "+json.dumps(transfer_reference.preflight),flush=True)
                            print(f"Whole-body transfer started: time={control_time:.3f}s, frame={frame_position:.3f}, support={support}", flush=True)
                        if transfer_reference is not None:
                            if source_step_rollout is None:
                                whole_body_reference = transfer_reference.sample(control_time, managed_clock.mode)
                                if unload_bridge is not None:
                                    whole_body_reference = unload_bridge.sample(control_time-transfer_reference.start_time, whole_body_reference)
                                if ((args.unified_source_step_test or args.coordinated_source_start) and transfer_reference.release_time is not None
                                        and control_time >= transfer_reference.release_time+3.):
                                    from fabrics_sim.reference.source_step_rollout import SourceStepRollout
                                    from g1_armature import unitree_armatures
                                    source_step_rollout = SourceStepRollout(fabric,whole_body_reference,G1_JOINT_NAMES,
                                        control_motion_trajectory,reference_manager.takeoff_frame,reference_manager.landing_frame,
                                        transfer_reference.support,FABRIC_TIMESTEP,
                                        q.new_tensor(unitree_armatures(G1_JOINT_NAMES)),args.output_path.with_suffix('.step_plan.json'),
                                        candidate_options=(dict(duration=1.5,
                                            landing_height=.63 if args.source_step_landing_fraction < 1 else .61,
                                            swing_phase_blend=.5 if args.source_step_landing_fraction < 1 else .7,
                                            landing_com_fraction=.45, landing_fraction=args.source_step_landing_fraction,
                                            landing_search_depth=args.source_step_landing_search_depth,
                                            terminal_capture='double_support',
                                            pelvis_pitch_offset=-.2 if args.source_step_landing_fraction < 1 else -.4)
                                            | source_step_profile.get('first',{})
                                            if args.coordinated_source_start else None),
                                        strict_landing=bool(args.tracking_steps),
                                        stop_duration=args.source_step_stop_duration,
                                        acceleration_limits=acceleration_limits,
                                        joint_bounds=dict(lower=joint_position_limits[:, :, 0],
                                            upper=joint_position_limits[:, :, 1],dt=FABRIC_TIMESTEP,
                                            bandwidth=args.joint_limit_barrier_bandwidth,
                                            velocity_damper_time=args.joint_limit_time_constant,
                                            impact_recovery=args.joint_limit_impact_recovery) if args.acceleration_and_braking_limits else None)
                                    fabric.dynamics_tasks.preserve_reference_acceleration = True
                                    source_step_start = control_time
                                    source_step_number = 1
                                    print(f"First source swing started: time={control_time:.3f}s",flush=True)
                            if support_switch is not None:
                                whole_body_reference = support_switch.sample(control_time)
                                if (source_step_number < args.source_step_count
                                        and support_switch.transfer.release_time is not None
                                        and control_time >= support_switch.transfer.release_time+3.):
                                    from fabrics_sim.reference.source_sequence import next_source_landing
                                    from fabrics_sim.reference.source_step_rollout import SourceStepRollout
                                    from g1_armature import unitree_armatures
                                    start_frame = source_step_rollout.end_frame
                                    recovering = source_recovery_active(source_step_number,
                                        args.source_recovery_after_step,args.source_recovery_step_count)
                                    takeoff, end_frame = ((start_frame,start_frame) if recovering else
                                        (pending_source_event["takeoff_frame"],pending_source_event["end_frame"])
                                        if pending_source_event is not None else next_source_landing(
                                            _to_numpy(motion_planned_modes),start_frame,support_switch.support))
                                    next_rollout = SourceStepRollout(fabric,whole_body_reference,G1_JOINT_NAMES,
                                        control_motion_trajectory,start_frame,end_frame,support_switch.support,FABRIC_TIMESTEP,
                                        q.new_tensor(unitree_armatures(G1_JOINT_NAMES)),
                                        args.output_path.with_suffix(f'.step_{source_step_number+1}_plan.json'),
                                        candidate_options=dict(duration=2.,landing_height=args.following_step_landing_height,
                                            swing_phase_blend=.85,landing_com_fraction=(args.recovery_landing_com_fraction if recovering else .45),
                                            clearance=.01,body_join="distributed",
                                            landing_fraction=(1. if recovering else args.following_step_landing_fraction if source_step_number==1
                                                else args.later_step_landing_fraction),
                                            landing_search_depth=args.source_step_landing_search_depth,
                                            terminal_capture='double_support',pelvis_pitch_offset=-.2)
                                            | source_step_profile.get('following',{}),
                                        stop_duration=(args.source_step_stop_duration if source_step_number==1
                                            else args.later_step_stop_duration),acceleration_limits=acceleration_limits,
                                        strict_landing=bool(args.tracking_steps),
                                        fixed_source_frame=start_frame if recovering else None,
                                        maximum_step_distance=args.recovery_maximum_step_distance if recovering else None,
                                        joint_bounds=dict(lower=joint_position_limits[:,:,0],upper=joint_position_limits[:,:,1],
                                            dt=FABRIC_TIMESTEP,bandwidth=args.joint_limit_barrier_bandwidth,
                                            velocity_damper_time=args.joint_limit_time_constant,
                                            impact_recovery=args.joint_limit_impact_recovery) if args.acceleration_and_braking_limits else None)
                                    source_step_rollout = next_rollout
                                    source_step_start = control_time
                                    source_step_number += 1
                                    support_switch = None
                                    print(f"Source step {source_step_number} started: time={control_time:.3f}, frames={start_frame}-{end_frame}, takeoff={takeoff}",flush=True)
                            elif source_step_rollout is not None:
                                whole_body_reference = (source_step_rollout.references[-1] if source_steps_finished
                                    else source_step_rollout.sample(control_time-source_step_start))
                            reference_q, reference_v, reference_a = whole_body_reference.q, whole_body_reference.v, whole_body_reference.a
                            reference_kinematics = (reference_v, reference_a)
                    except (ValueError, RuntimeError) as exc:
                        save_reference_failure(args, exc, control_time, q, qd, foot_normal_forces,
                                               tracking_history, servo_history)
                        raise
                    if mocap_transition is not None:
                        try:
                            whole_body_reference = mocap_transition.sample(
                                whole_body_reference,
                                WholeBodyReference(q, qd, qp_output.predicted_acc),
                                observed_feet, control_time, managed_clock, reference_manager,
                                motion_fps, stopped_for_transfer, actual_com_velocity, foot_normal_forces, FABRIC_TIMESTEP)
                            reference_q, reference_v, reference_a = (whole_body_reference.q,
                                whole_body_reference.v, whole_body_reference.a)
                            reference_kinematics = (reference_v, reference_a)
                        except (ValueError, RuntimeError) as exc:
                            save_reference_failure(args, exc, control_time, q, qd, foot_normal_forces,
                                                   tracking_history, servo_history)
                            raise
                    if elapsed_motion_time>=args.measured_stance_reference_start:
                        from measured_stance_reference import MeasuredStanceReference
                        if 'measured_stance_reference' not in locals():
                            measured_stance_reference=MeasuredStanceReference(
                                'src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf',
                                G1_JOINT_NAMES,_to_numpy(motion_planned_modes),motion_fps,
                                ramp=args.measured_stance_reference_ramp,limit=args.measured_stance_reference_limit)
                        measured_stance_reference.observe(frame_position,_to_numpy(q[0]),
                            _to_numpy(foot_normal_forces[0]),tuple(map(_to_numpy,contact_data)))
                        whole_body_reference=measured_stance_reference.project(frame_position,frame_rate,
                            frame_acceleration,motion_sampler,whole_body_reference)
                        reference_q,reference_v,reference_a=whole_body_reference.q,whole_body_reference.v,whole_body_reference.a
                        reference_kinematics=(reference_v,reference_a)
                    if elapsed_motion_time >= args.stance_heading_start:
                        from g1_stance_heading import StanceHeadingReference
                        try:
                            if 'stance_heading_reference' not in locals():
                                stance_heading_reference = StanceHeadingReference(
                                    'src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf',
                                    G1_JOINT_NAMES, _to_numpy(motion_planned_modes), motion_fps,
                                    ramp=args.stance_heading_ramp, limit=args.stance_heading_limit)
                            stance_heading_reference.observe(frame_position, _to_numpy(q[0]),
                                _to_numpy(foot_normal_forces[0]), tuple(map(_to_numpy, contact_data)))
                            whole_body_reference = stance_heading_reference.project(
                                frame_position, frame_rate, frame_acceleration, motion_sampler, whole_body_reference)
                            reference_q, reference_v, reference_a = (whole_body_reference.q,
                                whole_body_reference.v, whole_body_reference.a)
                            reference_kinematics = (reference_v, reference_a)
                        except (ValueError, RuntimeError) as exc:
                            save_reference_failure(args, exc, control_time, q, qd, foot_normal_forces,
                                                   tracking_history, servo_history)
                            raise
                    reference_state = whole_body_reference.apply(fabric)
                else:
                    reference_state = configure_motion_targets(
                        fabric,
                        reference_q,
                        previous_reference_q,
                        next_reference_q,
                        FABRIC_TIMESTEP,
                        reference_momentum=reference_manager is None or args.source_momentum_reference,
                        reference_kinematics=reference_kinematics,
                    )
                    if reference_manager is not None:
                        from fabrics_sim.reference.posture import crouched_posture
                        minimum_knee=.2 if args.reference_footsteps == "source" else .5
                        posture, crouch_blend = crouched_posture(reference_q[:,7:],G1_JOINT_NAMES,control_time,
                                                                 minimum_knee=minimum_knee)
                        previous_posture,_ = crouched_posture(
                            (reference_q if previous_reference_q is None else previous_reference_q)[:,7:],
                            G1_JOINT_NAMES,control_time-FABRIC_TIMESTEP,minimum_knee=minimum_knee)
                        next_posture,_ = crouched_posture(
                            (reference_q if next_reference_q is None else next_reference_q)[:,7:],
                            G1_JOINT_NAMES,control_time+FABRIC_TIMESTEP,minimum_knee=minimum_knee)
                        posture_v,posture_a = finite_difference_reference(
                            previous_posture,posture,next_posture,FABRIC_TIMESTEP,False,False)
                        if reference_kinematics is not None:
                            from fabrics_sim.reference.posture import crouched_posture_kinematics
                            posture,posture_v,posture_a,crouch_blend = crouched_posture_kinematics(
                                reference_q[:,7:],reference_v[:,6:],reference_a[:,6:],
                                G1_JOINT_NAMES,control_time,minimum_knee=minimum_knee)
                        fabric.set_joint_posture_target(posture,posture_v,posture_a)
                        # Adapted slow steps do not share the source clip's angular
                        # momentum; asking for that momentum conflicts with balance.
                        if not args.source_momentum_reference:
                            fabric.set_centroidal_momentum_target(torch.zeros(1,3,device=device))
                        if args.reference_footsteps == "source":
                            # Stabilize the torso during dynamic transfer. A pelvis-
                            # only orientation target leaves waist pitch available
                            # to satisfy COM/hand tasks until it hits a joint stop.
                            body_model=fabric.dynamics_tasks.model
                            body_index=body_model.kinematics.get_link_index('torso_link')
                            if reference_state is not None:
                                body_quat=reference_state['transforms'][:,body_index,3:7]
                                body_jac=reference_state['angular_jac'][:,body_index]
                                body_v=(body_jac@reference_v[...,None]).squeeze(-1)
                                body_a=(body_jac@reference_a[...,None]).squeeze(-1)+reference_state['angular_jac_bias'][:,body_index]
                            else:
                                body_quat=body_model.snapshot(reference_q)['transforms'][:,body_index,3:7].clone()
                                before_quat=(body_quat if previous_reference_q is None else
                                    body_model.snapshot(previous_reference_q)['transforms'][:,body_index,3:7].clone())
                                after_quat=(body_quat if next_reference_q is None else
                                    body_model.snapshot(next_reference_q)['transforms'][:,body_index,3:7].clone())
                                body_v,body_a=angular_reference(before_quat,body_quat,after_quat,FABRIC_TIMESTEP,
                                    previous_reference_q is None,next_reference_q is None)
                            fabric.set_body_orientation_target(body_quat,body_v,body_a,link='torso_link')
                if warming_up:
                    planned_contact_mode = torch.full_like(
                        commanded_contact_mode, 3
                    )
                    support_frame = motion_start_frame
                    support_reference_q = reference_q
                    support_mode = planned_contact_mode
                else:
                    planned_contact_mode = motion_planned_modes[
                        motion_frame:motion_frame + 1
                    ]
                    lookahead_end_frame = int(round(min(
                        frame_position
                        + args.com_lookahead
                        * motion_fps
                        * args.motion_speed,
                        motion_trajectory.shape[0] - 1,
                    )))
                    support_frame = lookahead_end_frame
                    support_reference_q = (
                        reference_q
                        if support_frame == motion_frame
                        else motion_trajectory[
                            support_frame:support_frame + 1
                        ]
                    )
                    support_mode = motion_planned_modes[
                        support_frame:support_frame + 1
                    ]
                    if pending_phase_mode is not None:
                        planned_contact_mode = torch.full_like(commanded_contact_mode, pending_phase_mode)
                    if phase_held:
                        # Do not look through to the opposite support phase
                        # while waiting at this phase boundary.
                        support_mode = planned_contact_mode
                        support_reference_q = reference_q
                # Use forces from the same physics snapshot as q and qd.
                # An extra controller-tick delay postpones touchdown and
                # loss-of-support handling by 20 ms.
                if reference_manager is None:
                    measured_left, measured_right, measured_mode = measured_contact_detector.update(foot_normal_forces)
                    measured_contacts = torch.stack((measured_left, measured_right),dim=1)
                else:
                    planned_contact_mode = motion_planned_modes[motion_frame:motion_frame+1]
                    support_mode = torch.full_like(commanded_contact_mode, managed_clock.support_mode)
                    support_reference_q = reference_q
                if args.single_support_test:
                    support_index = 0 if args.single_support_test == "left" else 1
                    support_mode_value = support_index + 1
                    live_com, com_jac = fabric.center_of_mass_taskmap._position_and_jacobian(q)
                    com_velocity = torch.bmm(com_jac, qd.unsqueeze(-1)).squeeze(-1)
                    support_jac = left_j if support_index == 0 else right_j
                    foot_speed = torch.linalg.vector_norm(torch.bmm(support_jac[:, :3], qd.unsqueeze(-1)).squeeze(-1), dim=1)
                    ready = single_support_ready(live_com, com_velocity, com_balance_target,
                                                 single_initial_feet[:, support_index, :3], foot_speed,
                                                 foot_normal_forces, support_index)
                    ready &= measured_contacts[:, support_index]
                    ready &= control_time >= args.motion_warmup_duration + cycle_preparation + (args.support_transfer_duration if contact_cycle is None else 0.)
                    if contact_cycle is not None:
                        swing = 1 - support_index
                        swing_jac = right_j if swing else left_j
                        swing_speed = torch.linalg.vector_norm((swing_jac @ qd[..., None]).squeeze(-1)).item()
                        cycle_feet = reference_feet_transforms(fabric, q)
                        clearance = (cycle_feet[0, swing, 2] - single_initial_feet[0, swing, 2]).item()
                        settled = bool((foot_normal_forces >= 20.).all()) and com_velocity.norm().item() < .05 and swing_speed < .03 and (live_com-com_balance_target).norm().item() < .03
                        old_phase = contact_cycle.phase
                        contact_cycle.update(control_time, ready=bool(ready.all()), clearance=clearance,
                                             swing_load=foot_normal_forces[0, swing].item(),
                                             swing_speed=swing_speed, settled=settled)
                        if contact_cycle.phase != old_phase:
                            print(f"contact cycle: {old_phase} -> {contact_cycle.phase} at {control_time:.3f}s", flush=True)
                        if single_release_time is None and contact_cycle.phase == "lift":
                            # The landing target is the settled sole pose, not
                            # the pre-drop initialization height.
                            single_initial_feet = cycle_feet.clone()
                            single_release_time = control_time
                            commanded_contact_mode.fill_(support_mode_value)
                    single_ready_count = single_ready_count + 1 if bool(ready.all()) else 0
                    if contact_cycle is None and single_release_time is None and single_ready_count >= 10:
                        single_release_time = control_time
                        commanded_contact_mode.fill_(support_mode_value)
                        print(f"single support: release at {control_time:.3f}s, support={args.single_support_test}")
                    if single_release_time is not None:
                        planned_contact_mode = torch.full_like(commanded_contact_mode, support_mode_value)
                    if contact_cycle is not None and contact_cycle.phase in ("settle", "done"):
                        planned_contact_mode.fill_(3)
                    if control_updates % 25 == 0:
                        print(f"single support gate: ready_count={single_ready_count}, released={single_release_time is not None}, "
                              f"com_error={torch.linalg.vector_norm(live_com-com_balance_target).item():.4f}, "
                              f"com_speed={torch.linalg.vector_norm(com_velocity).item():.4f}, "
                              f"foot_speed={foot_speed.item():.4f}, forces={foot_normal_forces.tolist()}")
                if not unified_reference and reference_manager is not None and managed_clock.mode in (1,2):
                    from fabrics_sim.reference.landing import LandingTransition
                    from fabrics_sim.reference.footstep import grounded_step
                    swing = 1 if managed_clock.mode == 1 else 0
                    key = (reference_manager.landing_frame,swing)
                    if key != landing_key:
                        landing_key = key
                        source_foot_bridge = None
                        source_rotation_bridge = None
                        endpoint = min(reference_manager.landing_frame or motion_frame,motion_trajectory.shape[0]-1)
                        source_feet = reference_feet_all[endpoint].detach().cpu().numpy()
                        if args.reference_footsteps == "source":
                            goal = source_landing_pose(source_feet[swing])
                        else:
                            goal = grounded_step(source_feet[1-swing], source_feet[swing],
                                                 contact_anchors[0,1-swing].detach().cpu().numpy())
                        jac = right_j if swing else left_j
                        twist=(jac@qd[...,None]).squeeze(-1)[0].detach().cpu().numpy()
                        measured_pose = reference_feet_transforms(fabric, q)[0, swing].detach().cpu().numpy()
                        swing_duration = 2.
                        if args.reference_footsteps == 'source':
                            swing_duration=max(.8,(endpoint-frame_position)/(motion_fps*args.motion_speed))
                            reference_manager.synchronize_swing(swing_duration)
                            source_com_bridge = source_foot_bridge = None
                        landing_transition=LandingTransition(control_time,measured_pose,twist,goal,
                                                             duration=swing_duration,clearance=.06,search_depth=.04,
                                                             search_duration=1.5)
                        if args.source_swing_reference:
                            from fabrics_sim.reference.phase_clock import SwingClock
                            if args.swing_touchdown_lead==0:
                                # One source phase for posture and feet. The
                                # measured-state bridge handles lift-off rates;
                                # a second clock would distort mid-swing timing.
                                source_foot_clock=reference_manager.swing_clock
                            else:
                                source_foot_clock=SwingClock(frame_position/motion_fps,managed_clock.rate,
                                                            endpoint/motion_fps,swing_duration,end_rate=0.)
                    pose,twist,acceleration = landing_transition.sample(control_time)
                    poses=reference_feet_transforms(fabric,reference_q).detach().clone()
                    poses[0,swing]=torch.as_tensor(pose,device=device,dtype=q.dtype)
                    # The stance rows are disabled in physical task requests.
                    v=torch.zeros(1,2,6,device=device,dtype=q.dtype);a=torch.zeros_like(v)
                    v[0,swing]=torch.as_tensor(twist,device=device,dtype=q.dtype)
                    a[0,swing]=torch.as_tensor(acceleration,device=device,dtype=q.dtype)
                    if args.reference_footsteps == 'compact':
                        fabric.set_feet_targets(poses[...,:3],poses[...,3:],v[...,:3],v[...,3:],a[...,:3],a[...,3:])
                    else:
                        # Source mode retains the coupled FK foot/COM path and
                        # sole orientation at the shared retimed source phase.
                        # Correct only the terminal sole clearance on flat ground.
                        from fabrics_sim.reference.landing import LandingTransition
                        endpoint_pose=reference_feet_all[endpoint,swing]
                        endpoint_rotation=_quat_xyzw_to_matrix(endpoint_pose[3:])
                        terminal_height=(endpoint_pose[:3,None]+endpoint_rotation@corners.T)[2].min()
                        clock=source_foot_clock if args.source_swing_reference else reference_manager.swing_clock
                        elapsed=control_time-landing_transition.start_time
                        fraction=1. if clock is None else min(1.,elapsed/clock.duration)
                        blend=10*fraction**3-15*fraction**4+6*fraction**5
                        correction=terminal_height.clamp(-.04,.04)
                        foot_positions=fabric.feet_pose_taskmap.position_target.clone()
                        foot_quaternions=fabric.feet_pose_taskmap.quaternion_target.clone()
                        # Reference derivatives are adjusted with the same blend.
                        features=fabric.fabrics_features['feet_pose']['feet_pose_attractor']
                        fv,fa=fabric.dynamics_tasks.references(features,torch.zeros(1,12,device=device))
                        fv,fa=fv.reshape(1,2,6).clone(),fa.reshape(1,2,6).clone()
                        if args.source_swing_reference:
                            foot_phase,foot_rate,foot_acc=clock.sample(elapsed)
                            foot_pose,foot_v,foot_a=source_foot_samplers[swing].kinematics(
                                foot_phase*motion_fps,foot_rate*motion_fps,foot_acc*motion_fps)
                            foot_positions[:,swing]=foot_pose[:,:3]
                            foot_quaternions[:,swing]=foot_pose[:,3:]
                            fv[:,swing]=foot_v;fa[:,swing]=foot_a
                        foot_positions[:,swing,2]-=blend*correction
                        duration=landing_transition.duration if clock is None else clock.duration
                        fv[:,swing,2]-=30*fraction**2*(1-fraction)**2/duration*correction
                        fa[:,swing,2]-=(60*fraction-180*fraction**2+120*fraction**3)/duration**2*correction
                        if args.source_swing_reference:
                            from fabrics_sim.reference.landing import contact_search
                            search=contact_search(elapsed-duration,landing_transition.search_depth,
                                                  landing_transition.search_duration)
                            for target,offset in zip((foot_positions[:,swing,2],fv[:,swing,2],fa[:,swing,2]),search):
                                target.add_(float(offset))
                        from fabrics_sim.reference.bridge import ReferenceBridge
                        bridge_duration=(duration if args.source_bridge_duration is None else
                                         min(duration,args.source_bridge_duration))
                        if source_foot_bridge is None:
                            to_np=lambda x:x.detach().cpu().numpy()
                            source_foot_bridge=ReferenceBridge(
                                measured_pose[:3]-to_np(foot_positions[0,swing]),
                                twist[:3]-to_np(fv[0,swing,:3]),
                                -to_np(fa[0,swing,:3]), bridge_duration)
                        offsets=source_foot_bridge.sample(control_time-landing_transition.start_time)
                        for target,offset in zip((foot_positions[0,swing],fv[0,swing,:3],fa[0,swing,:3]),offsets):
                            target+=torch.as_tensor(offset,device=device,dtype=q.dtype)
                        if args.source_swing_reference:
                            from fabrics_sim.reference.bridge import RotationReferenceBridge
                            to_np=lambda x:x.detach().cpu().numpy()
                            if source_rotation_bridge is None:
                                source_rotation_bridge=RotationReferenceBridge(measured_pose[3:],twist[3:],
                                    to_np(foot_quaternions[0,swing]),to_np(fv[0,swing,3:]),
                                    to_np(fa[0,swing,3:]),bridge_duration)
                            blended=source_rotation_bridge.sample(control_time-landing_transition.start_time,
                                to_np(foot_quaternions[0,swing]),to_np(fv[0,swing,3:]),to_np(fa[0,swing,3:]))
                            for target,value in zip((foot_quaternions[0,swing],fv[0,swing,3:],fa[0,swing,3:]),blended):
                                target.copy_(torch.as_tensor(value,device=device,dtype=q.dtype))
                        fabric.set_feet_targets(foot_positions,foot_quaternions,
                            fv[:,:,:3],fv[:,:,3:],fa[:,:,:3],fa[:,:,3:])
                controller_contact_plan=planned_contact_mode
                if args.gate_contact_transitions and measured_steps is not None and elapsed_motion_time>=args.landing_gated_unload_start:
                    from landing_gated_unload import intended_unload_mode,anticipatory_unload_mode
                    current_feet=reference_feet_transforms(fabric,q)
                    original_feet=reference_feet_transforms(fabric,audit_reference_q)
                    contact_plan_value,contact_plan_reason=intended_unload_mode(measured_steps,frame_position,
                        _to_numpy(current_feet[0,:,:3]),_to_numpy(original_feet[0,:,:3]),int(planned_contact_mode.item()))
                    if elapsed_motion_time>=args.anticipate_unload_start:
                        contact_plan_value,contact_plan_anticipated=anticipatory_unload_mode(measured_steps,frame_position,
                            _to_numpy(current_feet[0,:,:3]),_to_numpy(original_feet[0,:,:3]),
                            _to_numpy(foot_normal_forces[0]),motion_fps,contact_plan_value,horizon=args.anticipate_unload_horizon,hysteresis=args.anticipate_unload_hysteresis,allow_overdue=args.anticipate_unload_overdue,error_limit=args.anticipate_unload_error_limit)
                    controller_contact_plan=torch.full_like(planned_contact_mode,contact_plan_value)
                if reference_manager is None:
                    commanded_contact_mode, contact_ready_counter = (
                        update_contact_mode_state_machine(
                            commanded_contact_mode,
                            controller_contact_plan,
                            measured_contacts,
                            contact_ready_counter,
                            args.contact_switch_confirm_updates,
                            # Dynamic walking still needs measured touchdown/load gates;
                            # --require-support-load also changes the legacy source clock.
                            foot_normal_forces=foot_normal_forces if (
                                args.require_support_load or args.motion_start_in_motion) else None,
                        )
                    )
                (
                    balance_support_mode,
                    balance_support_reference_q,
                ) = select_balance_support(
                    commanded_contact_mode,
                    measured_mode,
                    reference_q,
                    support_reference_q,
                    support_mode,
                )
                if unified_reference:
                    # All targets were set together above. No COM filter, foot bridge,
                    # crouch override, or hand/pelvis shift may replace individual rows.
                    com_balance_target = fabric.center_of_mass_taskmap.position_target.clone()
                    features = fabric.fabrics_features['center_of_mass']['center_of_mass_attractor']
                    com_target_velocity = features['velocity'].clone()
                    com_target_acceleration = features['acceleration'].clone()
                else:
                    raw_com_target = support_aware_com_target(
                        fabric,
                        reference_q,
                        balance_support_reference_q,
                        balance_support_mode,
                        support_anchors=contact_anchors[..., :3] if reference_manager is not None else None,
                        anchored_contacts=torch.stack(((commanded_contact_mode == 1) | (commanded_contact_mode == 3),
                                                       (commanded_contact_mode == 2) | (commanded_contact_mode == 3)), dim=1),
                    )
                    if reference_manager is not None and managed_clock.support_mode in (1,2):
                        support_index = managed_clock.support_mode-1
                        raw_com_target[:,:2] = .95*contact_anchors[:,support_index,:2]+.05*raw_com_target[:,:2]
                    if reference_manager is not None and reference_manager.settling:
                        support_index = reference_manager.landing_support-1
                        # Give the landed foot a sustained share of load before
                        # asking it to become the next support. Keeping COM fully
                        # over the old ankle would make the settle gate impossible.
                        raw_com_target[:,:2] = (.8*contact_anchors[:,support_index,:2]
                                               +.2*contact_anchors[:,1-support_index,:2])
                    if reference_manager is not None:
                        raw_com_target[:,2] = (fabric.compute_center_of_mass(reference_q).detach()[:,2]-.005*crouch_blend
                                              if args.reference_footsteps == "source" else initial_center_of_mass[:,2]-.06*crouch_blend)
                    if args.weight_transfer_test:
                        transfer_target = support_aware_com_target(
                            fabric, reference_q, reference_q, torch.full_like(commanded_contact_mode, 2))
                        phase = max(0., min(1., (control_time - args.motion_warmup_duration) / 3.))
                        blend = phase**3 * (10. - 15. * phase + 6. * phase**2)
                        raw_com_target = initial_center_of_mass + blend * (transfer_target - initial_center_of_mass)
                    if args.single_support_test:
                        transfer_target = initial_center_of_mass.clone()
                        transfer_target[:, :2] = (.95 * single_initial_feet[:, support_index, :2]
                                                 + .05 * initial_center_of_mass[:, :2])
                        phase = max(0., min(1., (control_time - args.motion_warmup_duration - cycle_preparation) / args.support_transfer_duration))
                        blend = phase**3 * (10. - 15. * phase + 6. * phase**2)
                        raw_com_target = initial_center_of_mass + blend * (transfer_target - initial_center_of_mass)
                        if cycle_preparation:
                            from fabrics_sim.solver.contact_cycle import smooth_height
                            depth, depth_v, depth_a = smooth_height(control_time-args.motion_warmup_duration,
                                                                   cycle_preparation, 0., args.cycle_crouch_depth)
                            raw_com_target[:, 2] -= depth
                            posture = reference_q[:, 7:].clone()
                            posture_v, posture_a = torch.zeros_like(posture), torch.zeros_like(posture)
                            for side in ("left", "right"):
                                for joint, gain in (("hip_pitch", -8.), ("knee", 16.), ("ankle_pitch", -8.)):
                                    index = G1_JOINT_NAMES.index(f"{side}_{joint}_joint")
                                    posture[:, index] += gain * depth
                                    posture_v[:, index], posture_a[:, index] = gain * depth_v, gain * depth_a
                            fabric.set_joint_posture_target(posture, posture_v, posture_a)
                        if contact_cycle is not None and contact_cycle.landing_time is not None:
                            from fabrics_sim.solver.contact_cycle import smooth_height
                            return_blend, _, _ = smooth_height(control_time-contact_cycle.landing_time, 3., 0., 1.)
                            raw_com_target[:, :2] = ((1-return_blend)*raw_com_target[:, :2]
                                                    + return_blend*initial_center_of_mass[:, :2])
                        if single_release_time is not None:
                            # One-second quintic vertical lift; no PKL joint command or integrator reset.
                            s = min(1., max(0., control_time - single_release_time))
                            p = single_initial_feet[..., :3].clone()
                            v = torch.zeros_like(p)
                            a = torch.zeros_like(p)
                            swing = 1 - support_index
                            p[:, swing, 2] += .04 * s**3 * (10. - 15.*s + 6.*s*s)
                            v[:, swing, 2] = .04 * 30.*s*s*(1.-s)**2
                            a[:, swing, 2] = .04 * 60.*s*(1.-s)*(1.-2.*s)
                            if contact_cycle is not None:
                                dz, vz, az = contact_cycle.reference(control_time)
                                p[:, swing, 2] = single_initial_feet[:, swing, 2] + dz
                                v[:, swing, 2], a[:, swing, 2] = vz, az
                            fabric.set_feet_targets(p, single_initial_feet[..., 3:], v, torch.zeros_like(v), a, torch.zeros_like(a))
                    previous_com_target = com_balance_target
                    previous_com_target_velocity = com_target_velocity
                    previous_source_com_acceleration = com_target_acceleration
                    smoothing = 1.0 - math.exp(
                        -FABRIC_TIMESTEP / args.com_target_time_constant
                    )
                    com_balance_target = (
                        previous_com_target
                        + smoothing * (raw_com_target - previous_com_target)
                    )
                    com_target_velocity = (
                        com_balance_target - previous_com_target
                    ) / FABRIC_TIMESTEP
                    com_target_acceleration = (
                        com_target_velocity - previous_com_target_velocity
                    ) / FABRIC_TIMESTEP
                    source_swing=(args.reference_footsteps == 'source' and managed_clock.mode in (1, 2))
                    if (fabric.dynamics_tasks is not None and (not source_swing)):
                        # The force objective consumes this acceleration. A first-
                        # order position filter differentiated twice produces an
                        # impulsive force request at support-target switches.
                        reference_speed=max(.2,float(previous_com_target_velocity.abs().max()))
                        com_balance_target, com_target_velocity, com_target_acceleration = bounded_reference_step(
                            previous_com_target, previous_com_target_velocity, raw_com_target,
                            FABRIC_TIMESTEP, args.com_target_time_constant,
                            max_speed=reference_speed if args.reference_footsteps == "source" else .2)
                    if source_swing:
                        source_com=fabric.compute_center_of_mass(reference_q).detach()
                        previous_com=fabric.compute_center_of_mass(previous_reference_q).detach()
                        next_com=fabric.compute_center_of_mass(next_reference_q).detach()
                        com_balance_target=source_com
                        com_target_velocity,com_target_acceleration=finite_difference_reference(
                            previous_com,source_com,next_com,FABRIC_TIMESTEP,False,False)
                        from fabrics_sim.reference.bridge import ReferenceBridge
                        if source_com_bridge is None:
                            to_np=lambda x:x[0].detach().cpu().numpy()
                            source_com_bridge=ReferenceBridge(
                                to_np(previous_com_target-source_com),
                                to_np(previous_com_target_velocity-com_target_velocity),
                                to_np(previous_source_com_acceleration-com_target_acceleration),
                                landing_transition.duration)
                        offsets=source_com_bridge.sample(control_time-landing_transition.start_time)
                        for target,offset in zip((com_balance_target,com_target_velocity,com_target_acceleration),offsets):
                            target+=torch.as_tensor(offset,device=device,dtype=q.dtype)
                    fabric.set_center_of_mass_target(
                        com_balance_target,
                        com_target_velocity,
                        com_target_acceleration,
                    )
                    if reference_manager is not None:
                        # Preserve the hand gesture relative to COM when shortening
                        # steps; absolute PKL hand positions would pull the body
                        # toward footsteps that the manager intentionally shortened.
                        if reference_state is not None:
                            source_com=reference_state['com']
                            source_v=(reference_state['com_jac']@reference_v[...,None]).squeeze(-1)
                            source_a=(reference_state['com_jac']@reference_a[...,None]).squeeze(-1)+reference_state['com_jac_bias']
                        else:
                            source_com = fabric.compute_center_of_mass(reference_q).detach()
                            previous_com = source_com if previous_reference_q is None else fabric.compute_center_of_mass(previous_reference_q).detach()
                            next_com = source_com if next_reference_q is None else fabric.compute_center_of_mass(next_reference_q).detach()
                            source_v, source_a = finite_difference_reference(
                                previous_com, source_com, next_com, FABRIC_TIMESTEP,
                                previous_reference_q is None, next_reference_q is None)
                        features = fabric.fabrics_features["hand_pose"]["hand_pose_attractor"]
                        hand_v, hand_a = fabric.dynamics_tasks.references(features, torch.zeros(1,12,device=device))
                        hand_v, hand_a = hand_v.reshape(1,2,6).clone(), hand_a.reshape(1,2,6).clone()
                        hand_v[:,:,:3] += (com_target_velocity-source_v)[:,None]
                        hand_a[:,:,:3] += (com_target_acceleration-source_a)[:,None]
                        fabric.set_hand_targets(
                            fabric.hand_pose_taskmap.position_target+(com_balance_target-source_com)[:,None],
                            fabric.hand_pose_taskmap.quaternion_target,
                            hand_v[:,:,:3], hand_v[:,:,3:], hand_a[:,:,:3], hand_a[:,:,3:])
                    if args.align_balance_pelvis:
                        align_pelvis_balance_reference(
                            fabric, reference_q, previous_reference_q, next_reference_q, FABRIC_TIMESTEP,
                            com_balance_target, com_target_velocity, com_target_acceleration)
                # Keep full-rate numeric history, but avoid hundreds of GPU
                # scalar synchronizations per second just for console output.
                if control_updates % max(1, round(.1/FABRIC_TIMESTEP)) == 0:
                    print(
                        f"time={control_time:.2f}s, frame={motion_frame}, "
                        f"warmup={warming_up}, phase_held={phase_held}, "
                        f"planned_mode={int(planned_contact_mode.item())}, "
                        f"mode={int(commanded_contact_mode.item())}, "
                        f"support_mode={int(balance_support_mode.item())}, "
                        f"measured={measured_contacts[0].tolist()}, "
                        f"joint_rmse={float((q[:,7:]-audit_reference_q[:,7:]).square().mean().sqrt()):.4f}, "
                        f"com_target={com_balance_target[0].detach().cpu().tolist()}", flush=True)
            else:
                planned_contact_mode = None
                if fabric.dynamics_tasks is not None:
                    # Detect once before both Fabric and projector. A separate
                    # detector preserves confirmation counters across overrides.
                    _, _, planned_contact_mode = measured_contact_detector.update(foot_normal_forces)

            joint_velocity = qd[:, 6:35]
            joint_position = q[:, 7:36]

            live_feet = reference_feet_transforms(fabric, q)
            if reference_manager is not None and warming_up:
                # Let the spawn settle onto the floor, then keep the settled
                # support poses fixed. This updates reference anchors only.
                contact_anchors = live_feet.detach().clone()

            # A support foot is constrained to the pose at which that
            # contact became active.  Refresh only a newly activated foot;
            # otherwise Baumgarte stabilization would chase the moving
            # reference trajectory instead of correcting physical slip.
            if motion_trajectory is not None or fabric.dynamics_tasks is not None:
                active_mode = commanded_contact_mode if motion_trajectory is not None else planned_contact_mode
                requested_left = (
                    (active_mode == 1)
                    | (active_mode == 3)
                )
                requested_right = (
                    (active_mode == 2)
                    | (active_mode == 3)
                )
                requested_contact_active = torch.stack(
                    (requested_left, requested_right), dim=1
                )
                rising_contact = (
                    requested_contact_active & ~previous_contact_active
                )
                contact_anchors = torch.where(
                    rising_contact.unsqueeze(-1),
                    live_feet,
                    contact_anchors,
                )
                if fabric.dynamics_tasks is not None:
                    fabric.set_contact_state(requested_contact_active)
            if elapsed_motion_time>=args.collision_barrier_bandwidth_start:
                if not math.isfinite(args.collision_barrier_bandwidth) or args.collision_barrier_bandwidth<=0:
                    raise ValueError('Collision barrier bandwidth must be finite and positive')
                cb_phase=min(1.,max(0.,(elapsed_motion_time-args.collision_barrier_bandwidth_start)/.1))
                fabric.dynamics_tasks.collision_barrier_bandwidth=8.+(args.collision_barrier_bandwidth-8.)*cb_phase**3*(10-15*cb_phase+6*cb_phase**2)
            if args.convex_wrist_collision and elapsed_motion_time>=args.convex_wrist_collision_start:
                fabric.dynamics_tasks.convex_wrist_geometry=delayed_convex_geometry
            if (elapsed_motion_time > args.hip_yaw_barrier_start
                    and fabric.dynamics_tasks.hip_yaw_barrier_geometry is None):
                fabric.dynamics_tasks.enable_hip_yaw_barrier(
                    args.hip_yaw_barrier_bandwidth, args.hip_yaw_barrier_margin)
            qdd_fabric = fabric.compute_acceleration(q, qd)
            left_contact_bias = right_contact_bias = None
            task_acceleration_constraints = None
            sole_barrier_diagnostic = ({})
            if fabric.dynamics_tasks is not None:
                # Use the current-state directional derivative already
                # evaluated by the physical task model. A filtered previous-
                # tick Jacobian introduces lag during support transitions.
                snapshot = fabric.dynamics_tasks.last_snapshot
                model = fabric.dynamics_tasks.model
                foot_indices = [model.kinematics.get_link_index(name)
                                for name in (LEFT_FOOT_BODY, RIGHT_FOOT_BODY)]
                contact_bias = torch.cat((snapshot["origin_jac_bias"][:, foot_indices],
                                          snapshot["angular_jac_bias"][:, foot_indices]), dim=-1)
                left_contact_bias, right_contact_bias = contact_bias[:, 0], contact_bias[:, 1]
                task_acceleration_constraints = fabric.dynamics_tasks.last_collision_constraints
            landing_diagnostic = {}
            momentum_bound_diagnostic = {}
            if args.landing_speed_limit is not None:
                count=len(foot_landing_geometry[0][0])
                landing_diagnostic=dict(landing_surface_height=np.full((1,count),np.nan),
                    landing_surface_speed=np.full((1,count),np.nan),landing_speed_active=np.zeros((1,count),dtype=bool))
                landing_mode = int(commanded_contact_mode.item()) if args.motion_start_in_motion else managed_clock.mode
                landing_enabled = args.motion_start_in_motion or reference_manager.swing_clear
                if (landing_mode in (1,2) and landing_enabled
                        and control_time >= args.landing_speed_limit_start):
                    from fabrics_sim.solver.landing_approach import landing_approach_constraints
                    swing=1 if landing_mode==1 else 0
                    C,d,height,speed,active=landing_approach_constraints(
                        live_feet[:,swing,:3],_quat_xyzw_to_matrix(live_feet[:,swing,3:]),
                        (left_j,right_j)[swing],contact_bias[:,swing],qd,*foot_landing_geometry[swing],
                        touchdown_speed=args.landing_speed_limit)
                    if task_acceleration_constraints is None:
                        task_acceleration_constraints=C,d
                    else:
                        old_C,old_d=task_acceleration_constraints
                        task_acceleration_constraints=torch.cat((old_C,C),1),torch.cat((old_d,d),1)
                    landing_diagnostic=dict(landing_surface_height=_to_numpy(height),
                        landing_surface_speed=_to_numpy(speed),landing_speed_active=_to_numpy(active))
            pose_error = contact_pose_error(live_feet, contact_anchors)
            contact_row_mask = None
            contact_patches = {}
            contact_patch_tasks = {}
            support_flatten_weights = q.new_zeros(1, 2)
            rolling_bias_correction = torch.zeros((1, 2, 3), device=device, dtype=q.dtype)
            rolling_bias_matched = torch.zeros((1, 2), device=device, dtype=torch.bool)
            if unified_reference and patch_manager is not None:
                # The outgoing foot can roll only after the incoming foot
                # actually carries the load and physical toe points still touch.
                if toe_contact is not None and not bool(requested_contact_active[0,toe_foot]):
                    toe_contact = None
                    toe_foot = None
                candidate = None
                if managed_clock is not None and managed_clock.mode == 3 and com_preview is not None:
                    candidate = reference_manager.outgoing_foot(managed_clock.support_mode)
                if toe_contact is None and patch_manager is not None and candidate in patch_manager.patches:
                    toe_foot=candidate
                    toe_contact=patch_manager.patches[candidate]
                if toe_contact is not None and toe_contact.flatten:
                    rotation = _quat_xyzw_to_matrix(live_feet[:,toe_foot,3:])
                    sole_z = (live_feet[:,toe_foot,:3,None]+rotation@corners.T)[:,2]
                    angular_jac = (left_j if toe_foot == 0 else right_j)[:,3:]
                    angular_speed = (angular_jac@qd[...,None]).norm()
                    flat = bool(sole_z.max()<=.002 and sole_z.min()>=-.002
                                and foot_normal_forces[0,toe_foot]>=20. and angular_speed<=.2)
                    toe_contact.flat_seconds = toe_contact.flat_seconds+FABRIC_TIMESTEP if flat else 0.
                    if toe_contact.flat_seconds>=.06-1e-9:
                        contact_anchors[:,toe_foot] = live_feet[:,toe_foot].detach()
                        pose_error = contact_pose_error(live_feet,contact_anchors)
                        print(f"Flat stance restored: foot={toe_foot}, time={control_time:.3f}s",flush=True)
                        toe_contact,toe_foot = None,None
                if toe_contact is None and candidate is not None:
                    forces, points, normals, distances, counts, starts = map(_to_numpy,contact_data)
                    start = int(starts.reshape(-1)[candidate]); count = int(counts.reshape(-1)[candidate])
                    rotation = _quat_xyzw_to_matrix(live_feet[:,candidate,3:])
                    local = (_to_numpy(rotation[0]).T @ (points[start:start+count]-_to_numpy(live_feet[0,candidate,:3])).T).T
                    from fabrics_sim.solver.toe_contact import has_touching_toe_line, has_loaded_toe_line
                    # Incoming load is checked below. The outgoing toe may
                    # already be unloaded while its measured manifold still
                    # touches the floor; do not pin the heel in that case.
                    touching = has_touching_toe_line(local,distances[start:start+count])
                    source_rollover = False
                    load = foot_normal_forces[0]
                    ready = managed_clock.mode == 3 and touching and bool(load.sum()>=100.) and bool(
                        load[1-candidate]>=args.toe_transfer_load_fraction*load.sum())
                    if source_rollover:
                        # Actual loaded corners establish this line. No
                        # airborne support or pitch contact moment is assumed.
                        ready=bool(load[candidate]>=100.) and has_loaded_toe_line(
                            local,forces[start:start+count],distances[start:start+count])
                    toe_ready_seconds = toe_ready_seconds+FABRIC_TIMESTEP if ready else 0.
                    # Once this foot is almost unloaded, holding all six flat
                    # contact coordinates can prevent its heel from releasing
                    # while the incoming heel takes an impact. Geometry must
                    # still establish BOTH touching toe endpoints, and the
                    # other foot must carry the load. Only heel pitch is freed;
                    # the manager's full-foot release gates remain unchanged.
                    unloaded_toe_ready = (ready and managed_clock.mode == 3
                                          and float(load[candidate]) <= 5.)
                    if toe_contact is None and (unloaded_toe_ready or toe_ready_seconds>=.06-1e-9):
                        from fabrics_sim.solver.toe_contact import ToeContact
                        toe_foot = candidate
                        if patch_manager is not None:
                            from fabrics_sim.solver.observed_sole_contact import ObservedSoleContact
                            toe_contact = ObservedSoleContact(live_feet[:,candidate,:3],rotation,control_time)
                        else:
                            toe_contact = ToeContact(live_feet[:,candidate,:3],rotation,control_time)
                        print(f"Toe-line support: foot={toe_foot}, time={control_time:.2f}s",flush=True)
                        toe_ready_seconds = 0.
                elif toe_contact is None:
                    toe_ready_seconds = 0.
                if toe_contact is not None:
                    pass
                if patch_manager is not None:
                    angular_speeds=[float((jac[:,3:]@qd[...,None]).norm()) for jac in (left_j,right_j)]
                    contact_patches,restored=patch_manager.update(requested_contact_active,live_feet,
                        angular_speeds,tuple(map(_to_numpy,contact_data)),control_time,FABRIC_TIMESTEP,
                        toe_foot,toe_contact)
                    for foot in restored:contact_anchors[:,foot]=live_feet[:,foot].detach()
                    if restored:pose_error=contact_pose_error(live_feet,contact_anchors)
                elif toe_contact is not None:
                    contact_patches={toe_foot:toe_contact}
                if contact_patches:
                    contact_row_mask=torch.ones(1,12,device=device,dtype=torch.bool)
                for patch_foot,patch in contact_patches.items():
                    if not getattr(patch,'polygon_support',False):
                        contact_row_mask[:,6*patch_foot+4] = False
                    if patch.point_support:
                        contact_row_mask[:,6*patch_foot+3:6*patch_foot+6] = False
                    jac = left_j if patch_foot == 0 else right_j
                    foot_bias = left_contact_bias if patch_foot == 0 else right_contact_bias
                    toe_reference = None
                    mapped_jac,mapped_bias,mapped_error,toe_task = patch.evaluate(
                        live_feet[:,patch_foot,:3],_quat_xyzw_to_matrix(live_feet[:,patch_foot,3:]),
                        jac,foot_bias,qd,control_time,orientation_reference=toe_reference)
                    if control_time >= args.rolling_contact_kinematics_start:
                        patch_points = (patch.vertices if getattr(patch, 'vertices', None) is not None
                                        else patch.local_point[:, None])
                        omega = (jac[:, 3:] @ qd[..., None]).squeeze(-1)
                        correction, matched = rolling_sphere_bias_correction(
                            _quat_xyzw_to_matrix(live_feet[:, patch_foot, 3:]), omega,
                            patch.frame, patch_points, *foot_landing_geometry[patch_foot])
                        mapped_bias = mapped_bias.clone()
                        mapped_bias[:, :3] += correction
                        rolling_bias_correction[:, patch_foot] = correction
                        rolling_bias_matched[:, patch_foot] = matched
                    if patch_foot == 0:
                        left_j,left_contact_bias = mapped_jac,mapped_bias
                    else:
                        right_j,right_contact_bias = mapped_jac,mapped_bias
                    pose_error[:,patch_foot] = mapped_error
                    if task_feedback_enabled:
                        flatten_task = toe_task
                        if unified_reference:
                            from fabrics_sim.reference.whole_body import support_orientation_request
                            foot_request = next(r for r in fabric.dynamics_tasks.last_requests if r.name == 'feet_pose')
                            toe_task = support_orientation_request(
                                foot_request, patch_foot, patch.frame, patch.point_support,
                                getattr(patch, 'polygon_support', False))
                        elif patch_manager is not None:
                            toe_task.name += f'_{patch_foot}'
                        if (unified_reference and elapsed_motion_time > args.support_flatten_start
                                and patch.flatten and not getattr(patch, 'polygon_support', False)
                                and bool(foot_normal_forces[0, patch_foot] >= 20.)
                                and int(planned_contact_mode.item()) & (1 << patch_foot)):
                            # Only a measured loaded, planned stance foot receives
                            # this soft landing-recovery objective. Its contact
                            # rows and physical wrench cone remain measured ones.
                            flat_phase = min(1., (elapsed_motion_time-args.support_flatten_start)/.1)
                            flat_blend = flat_phase**3*(10-15*flat_phase+6*flat_phase**2)
                            if args.support_flatten_gain_scale!=1. or args.support_flatten_feedback_limit!=25.:
                                from g1_support_flatten import flat_support_target
                                flatten_task.desired=flat_support_target(
                                    _quat_xyzw_to_matrix(live_feet[:,patch_foot,3:]),
                                    getattr(patch,'sole_frame',patch.frame),patch.frame,
                                    mapped_jac[:,3:],qd,gain_scale=args.support_flatten_gain_scale,
                                    feedback_limit=args.support_flatten_feedback_limit)
                            toe_task.desired = torch.lerp(toe_task.desired, flatten_task.desired, flat_blend)
                            toe_task.metric = torch.lerp(toe_task.metric, flatten_task.metric, flat_blend)
                            support_flatten_weights[:, patch_foot] = flat_blend
                        contact_patch_tasks[patch_foot]=toe_task
                        fabric.dynamics_tasks.last_requests.append(toe_task)
            if control_time>=args.late_contact_stabilization_start:
                from dataclasses import replace as replace_projection
                contact_phase=min(1.,max(0.,(control_time-args.late_contact_stabilization_start)/.04))
                contact_blend=contact_phase**3*(10-15*contact_phase+6*contact_phase**2)
                projector.projection_config=replace_projection(projector.projection_config,
                    contact_kp=args.contact_stabilization_kp+contact_blend*(args.late_contact_stabilization_kp-args.contact_stabilization_kp),
                    contact_kd=args.contact_stabilization_kd+contact_blend*(args.late_contact_stabilization_kd-args.contact_stabilization_kd),
                    contact_stabilization_mask=(1.,1.,1.,1.,1.,1.))
            output = projector.step(
                acc_fabric=qdd_fabric,
                mass_matrix=preserve_com_acceleration(mass,
                    fabric.dynamics_tasks.last_snapshot['com_jac'], args.com_acceleration_preservation_weight)
                    if args.com_acceleration_preservation_weight else mass,
                generalized_velocity=qd,
                left_foot_jacobian=left_j,
                right_foot_jacobian=right_j,
                foot_normal_forces=foot_normal_forces,
                left_jdot_v=left_contact_bias,
                right_jdot_v=right_contact_bias,
                left_contact_pose_error=pose_error[:, 0],
                right_contact_pose_error=pose_error[:, 1],
                contact_row_mask=contact_row_mask,
                contact_mode_override=(
                    commanded_contact_mode
                    if motion_trajectory is not None
                    else planned_contact_mode
                ),
            )
            current_contact_active = torch.stack(
                (output.left_contact, output.right_contact), dim=1
            )
            # In force-detection mode the new mode is known only after the
            # projector update.  Capture its current pose for subsequent
            # controller ticks.
            detected_rising_contact = (
                current_contact_active & ~previous_contact_active
            )
            contact_anchors = torch.where(
                detected_rising_contact.unsqueeze(-1),
                live_feet,
                contact_anchors,
            )
            previous_contact_active = current_contact_active
            qdd_dc = qdd_fabric if args.g1_md else output.acc_dc
            # Acceleration-only tracking uses this projected acceleration as
            # the QP target, with the same contact rows and stabilized bias.
            contact_jacobian = torch.cat((left_j, right_j), dim=1)
            # The projector's finite-difference estimator and the QP must use
            # the same Jdot*v value. Recover it from the reported residual:
            # residual_before = J*a_fabric + Jdot*v plus the same Baumgarte
            # stabilization terms used by the projector. Passing the
            # recovered bias to the QP preserves that stabilized target.
            jdot_qdot = (
                output.contact_residual_before
                - torch.bmm(
                    contact_jacobian * output.active_row_mask.unsqueeze(-1),
                    qdd_fabric.unsqueeze(-1),
                ).squeeze(-1)
            )
            friction_G, friction_b = (
                build_single_support_wrench_constraints(
                    contact_mode=output.contact_mode,
                    foot_rotation_world=_quat_xyzw_to_matrix(
                        live_feet[..., 3:7]
                    ),
                    friction_coefficient=(
                        acceleration_qp.friction_coefficient
                    ),
                    yaw_friction_coefficient=(
                        acceleration_qp.yaw_friction_coefficient
                    ),
                    toe_length=acceleration_qp.toe_length,
                    heel_length=acceleration_qp.heel_length,
                    half_width=acceleration_qp.half_width,
                    sole_depth=.035 if args.use_sole_wrench_geometry else 0.,
                    coupled_yaw=args.reference_manager,
                    maximum_normal_force=(
                        acceleration_qp.maximum_normal_force
                    ),
                )
            )
            command_acc_lower = -acceleration_limits.unsqueeze(0).clone()
            friction_blocks=split_foot_wrench_constraints(friction_G,friction_b)
            for patch_foot,patch in contact_patches.items():
                # Toe wrench coordinates are local to its fixed horizontal
                # frame at its material anchor. Each foot has its own patch.
                if elapsed_motion_time>=args.contact_friction_refine_start:
                    from inscribed_patch_wrench import observed_patch_constraints
                    patch_G,patch_b=observed_patch_constraints(patch,
                        _quat_xyzw_to_matrix(live_feet[:,patch_foot,3:]),
                        acceleration_qp.friction_coefficient,acceleration_qp.maximum_normal_force,
                        args.contact_friction_rays,
                        minimum_normal_force=20. if not patch.polygon_support and int(output.contact_mode.item()) in (1,2) else 0.,
                        yaw_limit=None if patch.polygon_support else acceleration_qp.yaw_friction_coefficient)
                    block=patch_G.new_zeros(1,patch_G.shape[1],12)
                    block[:,:,6*patch_foot:6*(patch_foot+1)]=patch_G
                    friction_blocks[patch_foot]=(block,patch_b)
                    continue
                if getattr(patch,'polygon_support',False):
                    polygon_G,polygon_b=patch.wrench_constraints(
                        _quat_xyzw_to_matrix(live_feet[:,patch_foot,3:]),
                        acceleration_qp.friction_coefficient,acceleration_qp.maximum_normal_force)
                    block=polygon_G.new_zeros(1,polygon_G.shape[1],12)
                    block[:,:,6*patch_foot:6*(patch_foot+1)]=polygon_G
                    friction_blocks[patch_foot]=(block,polygon_b)
                    continue
                line_G,line_b = build_single_support_wrench_constraints(
                    contact_mode=output.contact_mode,
                    foot_rotation_world=torch.eye(3,device=device,dtype=q.dtype).expand(1,2,3,3),
                    friction_coefficient=acceleration_qp.friction_coefficient,
                    yaw_friction_coefficient=acceleration_qp.yaw_friction_coefficient,
                    toe_length=0.,heel_length=0.,half_width=0. if patch.point_support else patch.half_width,
                    sole_depth=0.,coupled_yaw=True,
                    maximum_normal_force=acceleration_qp.maximum_normal_force)
                rows=slice(20*patch_foot,20*(patch_foot+1))
                friction_blocks[patch_foot]=(line_G[:,rows],line_b[:,rows])
            friction_G=torch.cat([block[0] for block in friction_blocks],dim=1)
            friction_b=torch.cat([block[1] for block in friction_blocks],dim=1)
            if elapsed_motion_time>=args.support_load_floor_start:
                from support_load_floor import support_load_floor
                load_phase=min(1.,max(0.,(elapsed_motion_time-args.support_load_floor_start)/.06))
                load_blend=load_phase**3*(10-15*load_phase+6*load_phase**2)
                support_minimum = args.support_load_floor
                if args.late_support_load_floor is not None and elapsed_motion_time > args.late_support_load_floor_start:
                    late_load_phase = min(1., (elapsed_motion_time-args.late_support_load_floor_start)/args.late_support_load_floor_ramp)
                    late_load_blend = late_load_phase**3*(10-15*late_load_phase+6*late_load_phase**2)
                    support_minimum += late_load_blend*(args.late_support_load_floor-support_minimum)
                load_G,load_b=support_load_floor(output.contact_mode,foot_normal_forces,
                    support_minimum*load_blend,acceleration_qp.maximum_normal_force,
                    preferred_mode=controller_contact_plan if args.support_load_floor_transfer else None)
                friction_G=torch.cat((friction_G,load_G),dim=1)
                friction_b=torch.cat((friction_b,load_b),dim=1)

            command_acc_upper = acceleration_limits.unsqueeze(0).clone()
            if args.direct_wbc:
                # Only a one-step prediction for physical limit checks/logging;
                # these values are not sent as joint position commands.
                joint_position_desired = joint_position.clone()
                joint_velocity_desired = joint_velocity.clone()
            if args.acceleration_and_braking_limits and not args.fixed_joint_test and not args.direct_wbc:
                try:
                    command_acc_lower[:, 6:], command_acc_upper[:, 6:] = integration_acceleration_bounds(
                        joint_position_desired, joint_velocity_desired,
                        joint_position_limits[:, :, 0], joint_position_limits[:, :, 1],
                        joint_command_speed_limit, acceleration_limits[6:], FABRIC_TIMESTEP,
                    )
                except ValueError:
                    failure_path = args.output_path.with_suffix(".failure.npz")
                    failure_path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(failure_path,
                        source_frame_count=np.asarray(getattr(args,'source_frame_count',0)),
                        source_start_frame=np.asarray(motion_start_frame),
                        controller_options=np.asarray(json.dumps(vars(args), default=str, sort_keys=True)),
                        failure_reason=np.asarray("infeasible command integration bounds"),
                        control_time=np.asarray(control_updates * FABRIC_TIMESTEP),
                        q=q.detach().cpu().numpy(), qd=qd.detach().cpu().numpy(),
                        command_position=joint_position_desired.cpu().numpy(),
                        command_velocity=joint_velocity_desired.cpu().numpy(),
                        joint_position_limits=joint_position_limits.cpu().numpy())
                    print(f"[fail] command integration bounds; saved: {failure_path}", flush=True)
                    raise
            integration_seed_position = joint_position_desired.clone()
            # Physical stopping-distance bounds are separate from command bounds.
            physical_lower = torch.full_like(qd, -float("inf"))
            physical_upper = torch.full_like(qd, float("inf"))
            damper_time = None
            if args.joint_limit_time_constant is not None:
                blend = min(1., max(0., (control_time-args.motion_warmup_duration)/.5))
                if blend > 0:
                    damper_time = args.joint_limit_time_constant * (3*blend**2-2*blend**3)
            physical_braking_active=args.acceleration_and_braking_limits or elapsed_motion_time>=args.physical_braking_start
            try:
                if physical_braking_active:
                    braking_joint_start = (12 if (args.physical_braking_upper_only or args.physical_braking_waist_only)
                                           and not args.acceleration_and_braking_limits else 0)
                    braking_joint_end = (15 if args.physical_braking_waist_only
                                         and not args.acceleration_and_braking_limits else joint_position.shape[1])
                    braking_slice = slice(braking_joint_start, braking_joint_end)
                    acceleration_slice = slice(6+braking_joint_start, 6+braking_joint_end)
                    if args.physical_braking_joints is not None:
                        braking_slice, acceleration_slice = named_braking_coordinates(
                            G1_JOINT_NAMES, args.physical_braking_joints)
                    physical_lower[:, acceleration_slice], physical_upper[:, acceleration_slice] = measured_joint_acceleration_bounds(
                        joint_position[:,braking_slice], joint_velocity[:,braking_slice], joint_position_limits[:, braking_slice, 0],
                        joint_position_limits[:, braking_slice, 1], FABRIC_TIMESTEP, velocity_damper_time=damper_time,
                        barrier_bandwidth=args.joint_limit_barrier_bandwidth,
                        impact_recovery=args.joint_limit_impact_recovery)
            except ValueError as exc:
                save_reference_failure(args,exc,control_time,q,qd,foot_normal_forces,
                                       tracking_history,servo_history)
                raise
            if physical_braking_active and not args.acceleration_and_braking_limits and args.physical_braking_upper_only:
                physical_lower[:,:18]=-float('inf')
                physical_upper[:,:18]=float('inf')
            integration_seed_velocity = joint_velocity_desired.clone()
            servo_bias, servo_acc_gain = integrated_servo_model(
                joint_position_desired, joint_velocity_desired,
                joint_position, joint_velocity, kp, kd, FABRIC_TIMESTEP,
            )
            qp_reference = qdd_dc if args.g1_md or args.fixed_joint_test or args.direct_wbc else synchronize_command_acceleration(
                qdd_dc, joint_position_desired, joint_velocity_desired,
                joint_position, joint_velocity,
                stiffness=20. if args.fast_command_sync else 1.,
                damping=8. if args.fast_command_sync else 2.,
                limit=30. if args.fast_command_sync else 5.,
            )
            wrench_reference = None
            wrench_weights = None
            # COM tracking alone need not unload the outgoing foot: contact
            # moments can maintain a different load split. Request transfer
            # here, without imposing this extra objective throughout startup
            # and swing. Release still requires measured load and balance.
            managed_load_transfer = (reference_manager is not None and reference_manager.phase == 'PREPARE_STEP')
            if args.mocap_pipeline:
                # Ask only the outgoing measured contact to unload. The full
                # dynamics QP determines all other forces from the rigid-body equations.
                event = reference_manager.pending_takeoff if managed_load_transfer and args.mocap_unload_weight > 0 else None
                if event is None:
                    direct_unload_event = None
                else:
                    if event != direct_unload_event:
                        direct_unload_event, direct_unload_started = event, control_time
                    phase = min(1., max(0., control_time-direct_unload_started))
                    weight = (mocap_transition.unload_weight(control_time) if mocap_transition is not None
                              else phase**3 * (10.-15.*phase+6.*phase**2))
                    outgoing = 2-int(reference_manager.modes[event])
                    wrench_reference = torch.zeros((1,12),device=device,dtype=q.dtype)
                    wrench_weights = torch.zeros_like(wrench_reference)
                    if args.mocap_unload_test:
                        wrench_weights[:,6*outgoing+2] = weight
                    else:
                        wrench_weights[:,6*outgoing:6*outgoing+6] = weight
            retain_source_support = False
            if (control_time>=args.retain_planned_support_start and int(output.contact_mode.item())==3
                    and int(controller_contact_plan.item())==3 and measured_steps is not None):
                # A measured landing gate can extend double support beyond
                # the raw source plan. Retain support according to that same
                # effective plan used by the contact state machine.
                pending_events=[event for event in measured_steps.events
                    if event['takeoff_frame']>frame_position and (event['takeoff_frame']-frame_position)/motion_fps<=.15]
                if pending_events:
                    event=pending_events[0];outgoing=1-event['support']
                    # Existing ground contact is required; this force objective never creates contact rows.
                    if float(foot_normal_forces[0,outgoing])>=10.:
                        retain_source_support=True
                        wrench_reference=torch.zeros((1,12),device=device,dtype=q.dtype)
                        wrench_weights=torch.zeros_like(wrench_reference)
                        wrench_reference[:,6*outgoing+2]=args.retain_planned_support_force
                        wrench_weights[:,6*outgoing+2]=1.
            unload_intended_mode=int(planned_contact_mode.item())
            unload_intended_reason='source_plan'
            if elapsed_motion_time>=args.landing_gated_unload_start and measured_steps is not None:
                from landing_gated_unload import intended_unload_mode
                unload_original_feet=reference_feet_transforms(fabric,audit_reference_q)
                unload_intended_mode,unload_intended_reason=intended_unload_mode(measured_steps,frame_position,
                    _to_numpy(live_feet[0,:,:3]),_to_numpy(unload_original_feet[0,:,:3]),unload_intended_mode)
            if elapsed_motion_time>=args.anticipate_unload_start and measured_steps is not None and int(output.contact_mode.item())==3:
                from landing_gated_unload import anticipatory_unload_mode
                unload_original_feet=reference_feet_transforms(fabric,audit_reference_q)
                unload_intended_mode,anticipated=anticipatory_unload_mode(measured_steps,frame_position,
                    _to_numpy(live_feet[0,:,:3]),_to_numpy(unload_original_feet[0,:,:3]),
                    _to_numpy(foot_normal_forces[0]),motion_fps,unload_intended_mode,horizon=args.anticipate_unload_horizon,hysteresis=args.anticipate_unload_hysteresis,allow_overdue=args.anticipate_unload_overdue,error_limit=args.anticipate_unload_error_limit)
                if anticipated:unload_intended_reason='anticipate_next_takeoff_after_load_hold'
            dynamic_unloading = (args.motion_start_in_motion and args.dynamic_unload_weight > 0
                and int(output.contact_mode.item()) == 3 and unload_intended_mode in (1,2))
            if dynamic_unloading:
                outgoing = 2-unload_intended_mode
                unload_ramp=args.late_unload_ramp if elapsed_motion_time>=args.late_unload_start else .05
                if direct_unload_event != outgoing:
                    direct_unload_event, direct_unload_started = outgoing, control_time
                    direct_unload_deadline_ramp = unload_ramp
                    if elapsed_motion_time >= args.unload_deadline_start and measured_steps is not None:
                        from g1_unload_timing import unload_ramp_for_source
                        direct_unload_deadline_ramp = unload_ramp_for_source(
                            measured_steps.events, frame_position, frame_rate,
                            outgoing, unload_ramp, FABRIC_TIMESTEP)
                if elapsed_motion_time >= args.unload_deadline_start and measured_steps is not None:
                    unload_ramp = direct_unload_deadline_ramp
                phase = min(1.,max(0.,(control_time-direct_unload_started)/unload_ramp))
                blend = phase**3*(10.-15.*phase+6.*phase**2)
                wrench_reference = torch.zeros((1,12),device=device,dtype=q.dtype)
                wrench_weights = torch.zeros_like(wrench_reference)
                wrench_weights[:,6*outgoing:6*outgoing+6] = blend
            elif args.motion_start_in_motion:
                direct_unload_event = None
            # The new transfer has its own measured gate; its force objective
            # must stay active after the source scheduler has stopped advancing.
            acceleration_qp.wrench_weight = (args.mocap_unload_weight if wrench_weights is not None else
                .01 if support_switch is not None or source_step_number > 1 else 1e-4)
            if dynamic_unloading:
                acceleration_qp.wrench_weight = args.late_unload_weight if elapsed_motion_time>=args.late_unload_start else args.dynamic_unload_weight
                if elapsed_motion_time >= args.unload_deadline_start and args.unload_deadline_weight is not None:
                    acceleration_qp.wrench_weight = args.unload_deadline_weight
            if retain_source_support:acceleration_qp.wrench_weight=args.retain_planned_support_weight
            height_priority = (False)
            swing_com_priority = (False)
            com_priority_projection = None
            priority_tasks=("center_of_mass",) if (args.balance_priority or height_priority or swing_com_priority or
                (args.transfer_balance_priority and (transfer_reference is not None or
                 (com_preview is not None and int(output.contact_mode.item()) == 3
                  and reference_manager.phase in ("PREPARE_STEP","LANDING","SETTLE"))))) else ()
            priority_rows={"center_of_mass": (2,)} if height_priority and com_priority_projection is None else {}
            if args.centroidal_balance_priority and 'center_of_mass' in priority_tasks:
                priority_tasks += ('centroidal_momentum',)
            if args.horizontal_balance_priority and 'center_of_mass' in priority_tasks:
                priority_rows['center_of_mass']=(0,1)
            if ((args.swing_clearance_priority or args.swing_position_priority or args.swing_pose_priority)
                    and (com_preview is not None or args.motion_start_in_motion)
                    and int(output.contact_mode.item()) in (1, 2)):
                priority_tasks += ("feet_pose",)
                if not args.swing_pose_priority:
                    priority_rows["feet_pose"] = (0, 1, 2, 6, 7, 8) if args.swing_position_priority else (2, 8)
            if 'feet_pose' in priority_tasks and elapsed_motion_time>=args.full_swing_pose_start:
                priority_rows['feet_pose']=tuple(range(12))
            if task_feedback_enabled and (elapsed_motion_time>=args.transition_swing_priority_start):
                from transition_swing_task import activate_transition_swing
                transition_rows=activate_transition_swing(fabric.dynamics_tasks.last_requests,
                    int(output.contact_mode.item()),unload_intended_mode,foot_normal_forces[0])
                if transition_rows is not None:
                    priority_tasks+=("feet_pose",)
                    priority_rows["feet_pose"]=transition_rows
            if priority_tasks and (args.upper_body_priority or args.whole_body_posture_priority):
                priority_tasks += ("body_orientation", "joint_posture")
                priority_rows["joint_posture"] = tuple(range(0 if args.whole_body_posture_priority else 12,29))
            if priority_tasks and (args.body_orientation_priority or args.balance_body_priority):
                priority_tasks += ("body_orientation", "source_pelvis_orientation")
            if priority_tasks and args.landing_orientation_priority and int(output.contact_mode.item()) == 3:
                for patch_foot,patch in contact_patches.items():
                    if patch.flatten:
                        # Point/edge contacts expose real rotational freedom.
                        # Completing that rotation creates a bearing sole;
                        # hard contact rows and available moments stay unchanged.
                        priority_tasks += (contact_patch_tasks[patch_foot].name,)
            if elapsed_motion_time>=args.support_orientation_priority_start:
                for support_request in contact_patch_tasks.values():
                    priority_tasks += (support_request.name,)
                    support_scale=args.support_orientation_metric_scale
                    if args.support_orientation_roll_pitch_only:
                        # A point/edge contact may freely tilt. Keep those
                        # source-foot orientation rows in the first solve;
                        # yaw remains secondary during loaded support.
                        priority_rows[support_request.name]=(0,1)
                        tilt_phase=min(1.,max(0.,(elapsed_motion_time-args.support_orientation_priority_start)/.06))
                        support_scale*=tilt_phase**3*(10-15*tilt_phase+6*tilt_phase**2)
                    support_request.metric *= support_scale
            priority_levels = None
            # Body selection can overlap the whole-body/upper-body selection.
            # Each task must occur once in the hierarchy's partition.
            priority_tasks = tuple(dict.fromkeys(priority_tasks))
            if priority_tasks and (args.separate_balance_priority or args.posture_last_priority):
                remaining = list(priority_tasks)
                levels = []
                if args.separate_balance_priority and 'center_of_mass' in remaining:
                    balance = (('center_of_mass','body_orientation','source_pelvis_orientation')
                               if args.balance_body_priority else
                               ('center_of_mass','centroidal_momentum')
                               if args.centroidal_balance_priority else ('center_of_mass',))
                    levels.append(balance)
                    remaining = [name for name in remaining if name not in balance]
                if args.posture_last_priority:
                    remaining = [name for name in remaining if name != 'joint_posture']
                    if 'joint_posture' not in priority_tasks:
                        priority_tasks += ('joint_posture',)
                    priority_rows.pop('joint_posture', None)
                if remaining:
                    levels.append(tuple(remaining))
                if args.posture_last_priority:
                    levels.append(('joint_posture',))
                priority_levels = tuple(levels)
            if support_switch is not None or source_step_number > 1:
                priority_tasks = ("center_of_mass", "joint_posture")
                priority_rows = {"joint_posture": tuple(range(12))}
                priority_levels = (("center_of_mass",), ("joint_posture",))
            if elapsed_motion_time>=args.stance_posture_secondary_start and 'joint_posture' in priority_tasks:
                # Keep the complete posture task in the final solve. The first
                # stage must let loaded legs realize balance/contact tracking.
                loaded_mode=int(output.contact_mode.item())
                primary_joint_rows=list(range(12,29))
                if not loaded_mode&1:primary_joint_rows+=list(range(6))
                if not loaded_mode&2:primary_joint_rows+=list(range(6,12))
                priority_rows['joint_posture']=tuple(sorted(primary_joint_rows))
            if elapsed_motion_time>=args.leg_posture_secondary_start and 'joint_posture' in priority_tasks:
                # Foot-space tracking and fixed contact anchors determine the
                # legs at the current floating-base pose. Keep source leg
                # posture as a secondary objective rather than locking its
                # incompatible source-base solution in the same first stage.
                priority_rows['joint_posture']=tuple(range(12,29))
            from fabrics_sim.reference.source_priority import source_swing_priority
            source_priority = source_swing_priority(args.source_swing_position_priority or args.source_swing_pose_priority,
                source_step_rollout is not None, support_switch is not None,
                source_steps_finished, int(output.contact_mode.item()), pose=args.source_swing_pose_priority,
                body_priority="balance" if args.balance_body_priority else "swing" if args.body_orientation_priority else None)
            if source_priority is not None:
                priority_tasks, priority_rows, priority_levels = source_priority
                com_priority_projection = None
            if control_time>=args.swing_balance_priority_start:
                priority_tasks=('feet_pose',) if args.swing_only_priority else ('center_of_mass','feet_pose')
                priority_rows={'feet_pose':tuple(range(12)) if elapsed_motion_time>=args.full_swing_pose_start else (0,1,2,6,7,8)}
                priority_levels=None
                com_priority_projection=None
            if task_feedback_enabled and (elapsed_motion_time>=args.late_momentum_priority_start):
                priority_tasks=tuple(dict.fromkeys(priority_tasks+('centroidal_momentum',)))
                priority_levels=None
                if args.late_momentum_roll_only:
                    priority_rows['centroidal_momentum']=(0,)
                for request in fabric.dynamics_tasks.last_requests:
                    if request.name=='centroidal_momentum':
                        scale=args.late_momentum_metric_scale
                        if args.late_momentum_feedback_damping is not None:
                            if not math.isfinite(args.late_momentum_feedback_damping) or args.late_momentum_feedback_damping<=0:
                                raise ValueError('Momentum feedback damping must be finite and positive')
                            dynamics=fabric.dynamics_tasks
                            if args.late_momentum_zero_target:
                                request.desired=-args.late_momentum_feedback_damping*dynamics.last_snapshot['momentum']
                            else:
                                request.desired=dynamics.momentum_rate_reference-args.late_momentum_feedback_damping*(dynamics.last_snapshot['momentum']-dynamics.momentum_reference)
                            mp_phase=min(1.,max(0.,(elapsed_motion_time-args.late_momentum_priority_start)/.06))
                            scale*=mp_phase**3*(10-15*mp_phase+6*mp_phase**2)
                        if args.late_momentum_total_rate_limit is not None:
                            if not math.isfinite(args.late_momentum_total_rate_limit) or args.late_momentum_total_rate_limit<=0:
                                raise ValueError('Momentum rate limit must be finite and positive')
                            request.desired=request.desired.clamp(-args.late_momentum_total_rate_limit,args.late_momentum_total_rate_limit)
                        request.metric*=scale
            if task_feedback_enabled and (elapsed_motion_time>=args.late_pelvis_priority_start):
                priority_tasks=tuple(dict.fromkeys(priority_tasks+('source_pelvis_orientation',)))
                priority_levels=None
                if args.late_pelvis_axes is not None:
                    if args.late_pelvis_yaw_only or len(set(args.late_pelvis_axes))!=len(args.late_pelvis_axes):
                        raise ValueError('Pelvis axes must be distinct and cannot combine with yaw-only')
                    priority_rows['source_pelvis_orientation']=tuple('xyz'.index(axis) for axis in args.late_pelvis_axes)
                if args.late_pelvis_yaw_only:
                    priority_rows['source_pelvis_orientation']=(2,)
                if args.late_pelvis_metric_scale!=1.:
                    if not math.isfinite(args.late_pelvis_metric_scale) or args.late_pelvis_metric_scale<=0:
                        raise ValueError('Late pelvis metric scale must be finite and positive')
                    for request in fabric.dynamics_tasks.last_requests:
                        if request.name=='source_pelvis_orientation':
                            request.metric*=args.late_pelvis_metric_scale
            if elapsed_motion_time>=args.late_balance_hierarchy_start:
                balance_names=(('source_pelvis_orientation',) if args.late_balance_hierarchy=='body'
                    else ('center_of_mass',) if args.late_balance_hierarchy in ('y','xy','xyz')
                    else ('center_of_mass','body_orientation','source_pelvis_orientation'))
                priority_tasks=tuple(dict.fromkeys(priority_tasks+balance_names))
                if 'center_of_mass' in balance_names:
                    priority_rows['center_of_mass'] = ((0,1,2) if args.late_balance_hierarchy in ('xyz','xyz-body')
                        else (1,) if args.late_balance_hierarchy == 'y' else (0,1))
                remaining=tuple(n for n in priority_tasks if n not in balance_names)
                priority_levels=(balance_names,remaining) if remaining else (balance_names,)
            if task_feedback_enabled and (args.joint_limit_avoidance_weight > 0):
                avoidance_request=joint_limit_avoidance_request(
                    joint_position, joint_velocity, joint_position_limits[:, :, 0],
                    joint_position_limits[:, :, 1], args.joint_limit_avoidance_weight)
                if elapsed_motion_time>=args.joint_limit_priority_start:
                    # A full-rank first stage otherwise leaves this braking task
                    # no acceleration freedom in the lower-priority solve.
                    brake_phase=min(1.,max(0.,(elapsed_motion_time-args.joint_limit_priority_start)/.06))
                    avoidance_request.metric*=brake_phase**3*(10-15*brake_phase+6*brake_phase**2)
                    if args.joint_limit_priority_upper_only:
                        avoidance_request.metric[:, 12:, 12:] *= args.joint_limit_priority_scale
                    else:
                        avoidance_request.metric *= args.joint_limit_priority_scale
                    priority_tasks=tuple(dict.fromkeys(priority_tasks+(avoidance_request.name,)))
                    if args.joint_limit_priority_upper_only:
                        priority_rows[avoidance_request.name]=tuple(range(12,29))
                    if priority_levels is not None:
                        priority_levels=(priority_levels[0]+(avoidance_request.name,),)+priority_levels[1:]
                fabric.dynamics_tasks.last_requests.append(avoidance_request)
            if args.joint_velocity_limits:
                speed_C,speed_d = joint_velocity_constraints(joint_velocity,joint_velocity_limits,FABRIC_TIMESTEP)
                task_acceleration_constraints = ((speed_C,speed_d) if task_acceleration_constraints is None else
                    (torch.cat((task_acceleration_constraints[0],speed_C),dim=1),
                     torch.cat((task_acceleration_constraints[1],speed_d),dim=1)))
            knee_guard_diagnostic = {}
            if args.support_knee_min_angle is not None:
                from g1_support_knee import support_knee_constraints
                knee_guard_active = elapsed_motion_time >= args.support_knee_guard_start
                knee_C, knee_d, knee_fields = support_knee_constraints(
                    joint_position, joint_velocity,
                    current_contact_active if knee_guard_active else torch.zeros_like(current_contact_active),
                    minimum_angle=args.support_knee_min_angle,
                    bandwidth=args.support_knee_guard_bandwidth,
                    knee_indices=tuple(G1_JOINT_NAMES.index(f'{side}_knee_joint') for side in ('left', 'right')))
                knee_guard_diagnostic = {key: _to_numpy(value).copy() for key, value in knee_fields.items()}
                if knee_guard_active:
                    task_acceleration_constraints = ((knee_C, knee_d) if task_acceleration_constraints is None else
                        (torch.cat((task_acceleration_constraints[0], knee_C), dim=1),
                         torch.cat((task_acceleration_constraints[1], knee_d), dim=1)))
            if task_feedback_enabled and (args.g1_md):
                from fabrics_sim.solver.fabric_mit import compose_task_policy
                try:
                    qdd_fabric, composed_policy = compose_task_policy(
                        fabric.dynamics_tasks.last_requests, qd,
                        fabric.dynamics_tasks.config["regularization"])
                except (ValueError, RuntimeError, FloatingPointError) as exc:
                    save_reference_failure(args, exc, control_time, q, qd, foot_normal_forces,
                                           tracking_history, servo_history)
                    raise
                qdd_dc = qdd_fabric
                # Preserve integration history, but continuously pull command
                # error back toward measurements. The existing QP correction
                # map excludes this synchronization from physical acceleration.
                qp_reference = synchronize_command_acceleration(
                    qdd_fabric, joint_position_desired, joint_velocity_desired,
                    joint_position, joint_velocity, stiffness=20., damping=8., limit=30.)
            # Diagnostic: independently bound vertical swing feedback so that
            # horizontal lag cannot consume the entire XYZ feedback allowance.
            feedback_phase = min(1., max(0., (elapsed_motion_time-args.vertical_feedback_start)/.02))
            feedback_blend = feedback_phase**3*(10.-15.*feedback_phase+6.*feedback_phase**2)
            if task_feedback_enabled and (feedback_blend > 0):
                dynamics = fabric.dynamics_tasks
                foot_map = fabric.feet_pose_taskmap
                foot_indices = [dynamics.model.kinematics.get_link_index(foot_map.frame_groups[n][0])
                    for n in foot_map.end_effector_order]
                foot_position = dynamics.last_snapshot['transforms'][:,foot_indices,:3]
                foot_params = fabric.fabric_params['feet_pose_attractor']
                for request in dynamics.last_requests:
                    if request.name != 'feet_pose':
                        continue
                    ref_velocity, ref_acceleration = dynamics.references(
                        fabric.fabrics_features['feet_pose']['feet_pose_attractor'], request.desired)
                    actual_velocity = (request.A @ qd[...,None]).squeeze(-1)
                    for foot, row in enumerate((2,8)):
                        feedback = (-foot_params['stiffness'][row]*(foot_position[:,foot,2]-foot_map.position_target[:,foot,2])
                            -foot_params['damping'][row]*(actual_velocity[:,row]-ref_velocity[:,row]))
                        desired = ref_acceleration[:,row]+feedback.clamp(-40.,40.)
                        request.desired[:,row] += feedback_blend*(desired-request.desired[:,row])
                        request.metric[:,row,row] *= 1.+3.*feedback_blend
            horizontal_phase=min(1.,max(0.,(elapsed_motion_time-args.horizontal_feedback_start)/.04))
            horizontal_blend=horizontal_phase**3*(10-15*horizontal_phase+6*horizontal_phase**2)
            if task_feedback_enabled and (horizontal_blend>0):
                dynamics=fabric.dynamics_tasks;foot_map=fabric.feet_pose_taskmap
                foot_indices=[dynamics.model.kinematics.get_link_index(foot_map.frame_groups[n][0]) for n in foot_map.end_effector_order]
                positions=dynamics.last_snapshot['transforms'][:,foot_indices,:3]
                foot_params=fabric.fabric_params['feet_pose_attractor']
                for request in dynamics.last_requests:
                    if request.name!='feet_pose':continue
                    rv,ra=dynamics.references(fabric.fabrics_features['feet_pose']['feet_pose_attractor'],request.desired)
                    actual=(request.A@qd[...,None]).squeeze(-1)
                    for foot in range(2):
                        for axis in range(2):
                            row=foot*6+axis
                            feedback=-foot_params['stiffness'][row]*(positions[:,foot,axis]-foot_map.position_target[:,foot,axis])-foot_params['damping'][row]*(actual[:,row]-rv[:,row])
                            new=ra[:,row]+feedback.clamp(-args.horizontal_feedback_limit,args.horizontal_feedback_limit)
                            request.desired[:,row]+=horizontal_blend*(new-request.desired[:,row])
                            request.metric[:,row,row]*=1+(args.horizontal_feedback_metric_scale-1)*horizontal_blend
            # An unloaded ankle can recover its pose without trying to impose
            # the source's stance geometry on the measured support contact.
            flight_mode = int(output.contact_mode.item())
            flight_feet = ([0] if flight_mode == 2 else [1] if flight_mode == 1 else [0,1] if flight_mode == 0 else [])
            feedback_phase = min(1.,max(0.,(elapsed_motion_time-args.flight_feedback_start)/args.flight_feedback_ramp))
            flight_blend = feedback_phase**3*(10.-15.*feedback_phase+6.*feedback_phase**2)
            if task_feedback_enabled and (flight_feet and flight_blend > 0):
                joint_params = fabric.fabric_params['joint_posture_attractor']
                for request in fabric.dynamics_tasks.last_requests:
                    if request.name != 'joint_posture':
                        continue
                    ref_v, ref_a = fabric.dynamics_tasks.references(
                        fabric.fabrics_features['joint_posture']['joint_posture_attractor'], request.desired)
                    feedback = (-joint_params['stiffness']*(q[:,7:]-fabric.joint_posture_taskmap.position_target)
                        -joint_params['damping']*(qd[:,6:]-ref_v))
                    for foot in flight_feet:
                        phase=min(1.,max(0.,(elapsed_motion_time-args.foot_feedback_starts[foot])/args.flight_feedback_ramp))
                        flight_blend=phase**3*(10-15*phase+6*phase**2)
                        rows = [index for index,name in enumerate(G1_JOINT_NAMES)
                            if name.startswith(('left_' if foot==0 else 'right_'))
                            and (('ankle_' in name) if args.foot_feedback_joints[foot]=='ankles' else index<12)]
                        new_desired = ref_a[:,rows]+feedback[:,rows].clamp(
                            -args.foot_feedback_limits[foot],args.foot_feedback_limits[foot])
                        request.desired[:,rows] += flight_blend*(new_desired-request.desired[:,rows])
            late_phase=min(1.,max(0.,(elapsed_motion_time-args.late_leg_feedback_start)/.04))
            late_blend=late_phase**3*(10-15*late_phase+6*late_phase**2)
            if task_feedback_enabled and (late_blend>0):
                selected=([0,1] if args.late_leg_feedback_side=='both' else [1])
                selected=[f for f in selected if args.late_leg_feedback_stance or f in flight_feet]
                rows=[j for f in selected for j in range(6*f,6*f+6)]
                if rows:
                    joint_params=fabric.fabric_params['joint_posture_attractor']
                    for request in fabric.dynamics_tasks.last_requests:
                        if request.name!='joint_posture':continue
                        rv,ra=fabric.dynamics_tasks.references(fabric.fabrics_features['joint_posture']['joint_posture_attractor'],request.desired)
                        feedback=-joint_params['stiffness']*(q[:,7:]-fabric.joint_posture_taskmap.position_target)-joint_params['damping']*(qd[:,6:]-rv)
                        new=ra[:,rows]+feedback[:,rows].clamp(-args.late_leg_feedback_limit,args.late_leg_feedback_limit)
                        request.desired[:,rows]+=late_blend*(new-request.desired[:,rows])
            phase=min(1.,max(0.,(elapsed_motion_time-args.waist_feedback_start)/.06));waist_blend=phase**3*(10-15*phase+6*phase**2)
            if task_feedback_enabled and (waist_blend>0):
                joint_params=fabric.fabric_params['joint_posture_attractor']
                waist_metric_scale = args.waist_metric_scale
                if args.late_waist_metric_scale is not None and elapsed_motion_time >= args.late_waist_metric_start:
                    late_waist_phase = min(1., (elapsed_motion_time-args.late_waist_metric_start)/args.late_waist_metric_ramp)
                    late_waist_blend = late_waist_phase**3*(10-15*late_waist_phase+6*late_waist_phase**2)
                    waist_metric_scale += late_waist_blend*(args.late_waist_metric_scale-waist_metric_scale)
                for request in fabric.dynamics_tasks.last_requests:
                    if request.name!='joint_posture':continue
                    rv,ra=fabric.dynamics_tasks.references(fabric.fabrics_features['joint_posture']['joint_posture_attractor'],request.desired)
                    feedback=-joint_params['stiffness']*(q[:,7:]-fabric.joint_posture_taskmap.position_target)-joint_params['damping']*(qd[:,6:]-rv)
                    rows=list(range(12,15));new=ra[:,rows]+feedback[:,rows].clamp(-args.waist_feedback_limit,args.waist_feedback_limit)
                    request.desired[:,rows]+=waist_blend*(new-request.desired[:,rows])
                    for row in rows:request.metric[:,row,row]*=1+(waist_metric_scale-1)*waist_blend
            com_feedback_phase=min(1.,max(0.,(elapsed_motion_time-args.com_feedback_start)/.06))
            com_feedback_blend=com_feedback_phase**3*(10-15*com_feedback_phase+6*com_feedback_phase**2)
            if task_feedback_enabled and (com_feedback_blend>0):
                if not math.isfinite(args.com_feedback_limit) or args.com_feedback_limit<=0 or not math.isfinite(args.com_feedback_gain_scale) or args.com_feedback_gain_scale<=0:
                    raise ValueError('COM feedback parameters must be finite and positive')
                dynamics=fabric.dynamics_tasks;state=dynamics.last_snapshot;params=fabric.fabric_params['center_of_mass_attractor']
                for request in dynamics.last_requests:
                    if request.name!='center_of_mass':continue
                    rv,ra=dynamics.references(fabric.fabrics_features['center_of_mass']['center_of_mass_attractor'],request.desired)
                    K=q.new_tensor(params['stiffness'])*args.com_feedback_gain_scale
                    D=q.new_tensor(params['damping'])*math.sqrt(args.com_feedback_gain_scale)
                    error=state['com']-fabric.center_of_mass_taskmap.position_target
                    velocity=(state['com_jac']@qd[...,None]).squeeze(-1)-rv
                    desired=ra+(-K*error-D*velocity).clamp(-args.com_feedback_limit,args.com_feedback_limit)
                    axes=slice(0,2) if args.com_feedback_axes=='xy' else slice(None)
                    request.desired[:,axes]+=com_feedback_blend*(desired[:,axes]-request.desired[:,axes])
            com_phase=min(1.,max(0.,(elapsed_motion_time-args.com_feedback_metric_start)/.06))
            com_blend=com_phase**3*(10-15*com_phase+6*com_phase**2)
            if task_feedback_enabled and (com_blend>0 and args.com_feedback_metric_scale!=1):
                for request in fabric.dynamics_tasks.last_requests:
                    if request.name=='center_of_mass':
                        request.metric*=1+(args.com_feedback_metric_scale-1)*com_blend
            global_phase=min(1.,max(0.,(elapsed_motion_time-args.global_posture_feedback_start)/.06))
            global_blend=global_phase**3*(10-15*global_phase+6*global_phase**2)
            if task_feedback_enabled and (global_blend>0):
                joint_params=fabric.fabric_params['joint_posture_attractor']
                for request in fabric.dynamics_tasks.last_requests:
                    if request.name!='joint_posture':continue
                    rv,ra=fabric.dynamics_tasks.references(fabric.fabrics_features['joint_posture']['joint_posture_attractor'],request.desired)
                    feedback=-joint_params['stiffness']*(q[:,7:]-fabric.joint_posture_taskmap.position_target)-joint_params['damping']*(qd[:,6:]-rv)
                    rows=slice(15,29) if args.global_posture_feedback_joints=='arms' else slice(None)
                    new=ra[:,rows]+feedback[:,rows].clamp(-args.global_posture_feedback_limit,args.global_posture_feedback_limit)
                    request.desired[:,rows]+=global_blend*(new-request.desired[:,rows])
                    indices=range(15,29) if args.global_posture_feedback_joints=='arms' else range(29)
                    for index in indices:request.metric[:,index,index]*=1+(args.global_posture_feedback_metric_scale-1)*global_blend
            stance_posture_scale = 1.
            if task_feedback_enabled and (elapsed_motion_time >= args.stance_posture_metric_start):
                stance_phase = min(1., (elapsed_motion_time-args.stance_posture_metric_start)
                                   / args.stance_posture_metric_ramp)
                stance_blend = stance_phase**3*(10.-15.*stance_phase+6.*stance_phase**2)
                stance_posture_scale += (args.stance_posture_metric_scale-1.)*stance_blend
                for request in fabric.dynamics_tasks.last_requests:
                    if request.name == 'joint_posture':
                        request.metric = scale_stance_posture_metric(
                            request.metric, current_contact_active, stance_posture_scale)
            upper_phase=min(1.,max(0.,(elapsed_motion_time-args.upper_posture_metric_start)/.06))
            leg_phase=min(1.,max(0.,(elapsed_motion_time-args.leg_posture_metric_start)/.06))
            leg_blend=leg_phase**3*(10-15*leg_phase+6*leg_phase**2)
            if task_feedback_enabled and leg_blend>0 and args.leg_posture_metric_scale!=1:
                for request in fabric.dynamics_tasks.last_requests:
                    if request.name=='joint_posture':
                        request.metric[:,:12,:12]*=1+(args.leg_posture_metric_scale-1)*leg_blend
            upper_blend=upper_phase**3*(10-15*upper_phase+6*upper_phase**2)
            upper_metric_scale = args.upper_posture_metric_scale
            if args.late_upper_metric_scale is not None and elapsed_motion_time >= args.late_upper_metric_start:
                upper_late_phase = min(1., (elapsed_motion_time-args.late_upper_metric_start)/args.late_upper_metric_ramp)
                upper_late_blend = upper_late_phase**3*(10-15*upper_late_phase+6*upper_late_phase**2)
                upper_metric_scale += upper_late_blend*(args.late_upper_metric_scale-upper_metric_scale)
            if task_feedback_enabled and (upper_blend>0 and upper_metric_scale!=1):
                for request in fabric.dynamics_tasks.last_requests:
                    if request.name=='joint_posture':
                        for joint_index in range(12,29):
                            request.metric[:,joint_index,joint_index]*=1+(upper_metric_scale-1)*upper_blend
            late_foot_phase=min(1.,max(0.,(elapsed_motion_time-args.late_foot_metric_start)/.04))
            late_foot_blend=late_foot_phase**3*(10-15*late_foot_phase+6*late_foot_phase**2)
            late_foot_multiplier=1+(args.late_foot_metric_scale-1)*late_foot_blend
            if elapsed_motion_time>args.late_foot_metric_end:
                if (args.late_foot_metric_end<args.late_foot_metric_start
                        or not math.isfinite(args.late_foot_metric_end_ramp) or args.late_foot_metric_end_ramp<=0
                        or not math.isfinite(args.late_foot_metric_end_scale) or args.late_foot_metric_end_scale<=0):
                    raise ValueError('Invalid foot task weight end transition')
                end_phase=min(1.,(elapsed_motion_time-args.late_foot_metric_end)/args.late_foot_metric_end_ramp)
                end_blend=end_phase**3*(10-15*end_phase+6*end_phase**2)
                late_foot_multiplier+=(args.late_foot_metric_end_scale-late_foot_multiplier)*end_blend
            if task_feedback_enabled and late_foot_multiplier!=1:
                for request in fabric.dynamics_tasks.last_requests:
                    if request.name=='feet_pose':request.metric*=late_foot_multiplier
            body_feedback_phase=min(1.,max(0.,(elapsed_motion_time-args.body_feedback_start)/.06))
            body_feedback_blend=body_feedback_phase**3*(10-15*body_feedback_phase+6*body_feedback_phase**2)
            if task_feedback_enabled and (body_feedback_blend>0):
                if not math.isfinite(args.body_feedback_limit) or args.body_feedback_limit<=0:
                    raise ValueError('Body feedback limit must be finite and positive')
                from fabrics_sim.fabrics.dynamics_aware import rotation_error as body_rotation_error
                from fabrics_sim.taskmaps.g1_end_effector_pose_taskmap import _quat_xyzw_to_matrix as body_reference_rotation
                dynamics=fabric.dynamics_tasks
                if dynamics.body_reference is None:
                    raise ValueError('Body feedback override requires the unified torso reference')
                body_q,body_v,body_a,body_link=dynamics.body_reference
                body_index=dynamics.model.kinematics.get_link_index(body_link)
                body_error=body_rotation_error(dynamics.last_snapshot['rotations'][:,body_index],body_reference_rotation(body_q))
                body_params=dynamics.config['body']
                for request in dynamics.last_requests:
                    if request.name=='body_orientation':
                        actual_omega=(request.A@qd[...,None]).squeeze(-1)
                        feedback=-q.new_tensor(body_params['stiffness'])*body_error-q.new_tensor(body_params['damping'])*(actual_omega-body_v)
                        desired=body_a+feedback.clamp(-args.body_feedback_limit,args.body_feedback_limit)
                        request.desired+=body_feedback_blend*(desired-request.desired)
            body_phase=min(1.,max(0.,(elapsed_motion_time-args.late_body_angular_metric_start)/.06))
            body_blend=body_phase**3*(10-15*body_phase+6*body_phase**2)
            if task_feedback_enabled and (body_blend>0 and args.late_body_angular_metric_scale!=1):
                if not math.isfinite(args.late_body_angular_metric_scale) or args.late_body_angular_metric_scale<=0:
                    raise ValueError('Late body angular metric must be finite and positive')
                for request in fabric.dynamics_tasks.last_requests:
                    if request.name=='body_orientation':
                        request.metric*=1+(args.late_body_angular_metric_scale-1)*body_blend
            linear_foot_phase=min(1.,max(0.,(elapsed_motion_time-args.late_foot_linear_feedback_start)/.06))
            linear_foot_blend=linear_foot_phase**3*(10-15*linear_foot_phase+6*linear_foot_phase**2)
            if task_feedback_enabled and linear_foot_blend>0 and args.late_foot_linear_feedback_limit is not None:
                from g1_foot_linear_feedback import linear_tracking_target
                dynamics=fabric.dynamics_tasks;taskmap=fabric.feet_pose_taskmap
                indices=[dynamics.model.kinematics.get_link_index(taskmap.frame_groups[n][0]) for n in taskmap.end_effector_order]
                params=fabric.fabric_params['feet_pose_attractor']
                for request in dynamics.last_requests:
                    if request.name!='feet_pose':continue
                    rv,ra=dynamics.references(fabric.fabrics_features['feet_pose']['feet_pose_attractor'],request.desired)
                    desired=linear_tracking_target(dynamics.last_snapshot['transforms'][:,indices,:3],
                        taskmap.position_target,(request.A@qd[...,None]).squeeze(-1).reshape(len(q),2,6)[:,:,:3],
                        rv.reshape(len(q),2,6)[:,:,:3],ra.reshape(len(q),2,6)[:,:,:3],
                        q.new_tensor(params['stiffness']).reshape(2,6)[:,:3],
                        q.new_tensor(params['damping']).reshape(2,6)[:,:3],args.late_foot_linear_feedback_limit)
                    rows=request.desired.reshape(len(q),2,6)
                    for side in (range(2) if args.late_foot_linear_feedback_side=='both' else
                                 [0 if args.late_foot_linear_feedback_side=='left' else 1]):
                        rows[:,side,:3]+=linear_foot_blend*(desired[:,side]-rows[:,side,:3])
            angular_phase=min(1.,max(0.,(elapsed_motion_time-args.late_foot_angular_metric_start)/.06))
            angular_blend=angular_phase**3*(10-15*angular_phase+6*angular_phase**2)
            if args.late_foot_angular_metric_end < args.late_foot_angular_metric_start:
                raise ValueError('Foot orientation weight window must end after its start')
            angular_end_phase=min(1.,max(0.,(elapsed_motion_time-args.late_foot_angular_metric_end)/.06))
            angular_blend*=1-angular_end_phase**3*(10-15*angular_end_phase+6*angular_end_phase**2)
            if task_feedback_enabled and (angular_blend>0 and (
                    args.late_foot_angular_metric_scale!=1 or args.late_foot_angular_feedback_limit is not None)):
                if not math.isfinite(args.late_foot_angular_metric_scale) or args.late_foot_angular_metric_scale<=0:
                    raise ValueError('Late foot angular metric must be finite and positive')
                for request in fabric.dynamics_tasks.last_requests:
                    if request.name=='feet_pose':
                        angular_rows=(3,4,5,9,10,11) if args.late_foot_angular_metric_side=='both' else ((3,4,5) if args.late_foot_angular_metric_side=='left' else (9,10,11))
                        if args.late_foot_angular_feedback_limit is not None:
                            from g1_foot_angular_feedback import angular_tracking_target
                            dynamics=fabric.dynamics_tasks;foot_map=fabric.feet_pose_taskmap
                            indices=[dynamics.model.kinematics.get_link_index(foot_map.frame_groups[n][0])
                                     for n in foot_map.end_effector_order]
                            rv,ra=dynamics.references(fabric.fabrics_features['feet_pose']['feet_pose_attractor'],request.desired)
                            angular_jac=request.A.reshape(len(q),2,6,35)[:,:,3:]
                            omega=(angular_jac@qd[:,None,:,None]).squeeze(-1)
                            params=fabric.fabric_params['feet_pose_attractor']
                            desired=angular_tracking_target(dynamics.last_snapshot['rotations'][:,indices],
                                _quat_xyzw_to_matrix(foot_map.quaternion_target),omega,
                                rv.reshape(len(q),2,6)[:,:,3:],ra.reshape(len(q),2,6)[:,:,3:],
                                q.new_tensor(params['stiffness']).reshape(2,6)[:,3:],
                                q.new_tensor(params['damping']).reshape(2,6)[:,3:],args.late_foot_angular_feedback_limit)
                            for side in sorted(set(row//6 for row in angular_rows)):
                                rows=slice(6*side+3,6*side+6)
                                request.desired[:,rows]+=angular_blend*(desired[:,side]-request.desired[:,rows])
                        for row in angular_rows:
                            request.metric[:,row,row]*=1+(args.late_foot_angular_metric_scale-1)*angular_blend
            # Bound the soft total task request, including feedforward spikes.
            # Unified reference q/v/a and physical acceleration constraints stay intact.
            total_phase=min(1.,max(0.,(elapsed_motion_time-args.posture_total_accel_start)/.06))
            total_blend=total_phase**3*(10-15*total_phase+6*total_phase**2)
            if task_feedback_enabled and (total_blend>0):
                if not math.isfinite(args.posture_total_accel_limit) or args.posture_total_accel_limit<=0:
                    raise ValueError('Posture total acceleration limit must be finite and positive')
                for request in fabric.dynamics_tasks.last_requests:
                    if request.name=='joint_posture':
                        bounded=request.desired.clamp(-args.posture_total_accel_limit,args.posture_total_accel_limit)
                        request.desired+=total_blend*(bounded-request.desired)
            if args.tracking_tube_soft_weight:
                acceleration_qp.solver.tube = None
                acceleration_qp.solver.braking = None
            soft_joint_braking_weight = 0.
            if (elapsed_motion_time>=args.tracking_tube_start and elapsed_motion_time<args.tracking_tube_end):
                from tracking_tube import tracking_tube_constraints
                tube_sampler = motion_sampler if args.tracking_tube_reference == 'control' else tube_source_sampler
                tube_q,tube_v,tube_a=tube_sampler.kinematics(frame_position,frame_rate,frame_acceleration)
                joint_tube_rate=args.tracking_tube_joint_bandwidth
                if elapsed_motion_time>=args.late_joint_tube_start:
                    if not math.isfinite(args.late_joint_tube_bandwidth) or args.late_joint_tube_bandwidth<=0:
                        raise ValueError('Late joint tube bandwidth must be finite and positive')
                    tube_phase=min(1.,max(0.,(elapsed_motion_time-args.late_joint_tube_start)/.06))
                    tube_blend=tube_phase**3*(10-15*tube_phase+6*tube_phase**2)
                    original_rate=args.tracking_tube_joint_bandwidth or args.tracking_tube_bandwidth
                    joint_tube_rate=original_rate+tube_blend*(args.late_joint_tube_bandwidth-original_rate)
                tube_rate=args.tracking_tube_bandwidth
                orientation_tube_rate=args.tracking_tube_orientation_bandwidth
                if elapsed_motion_time>=args.late_tracking_tube_start:
                    if not math.isfinite(args.late_tracking_tube_bandwidth) or args.late_tracking_tube_bandwidth<=0:
                        raise ValueError('Late tracking tube bandwidth must be positive')
                    phase=min(1.,max(0.,(elapsed_motion_time-args.late_tracking_tube_start)/.1))
                    blend=phase**3*(10.-15.*phase+6.*phase**2)
                    tube_rate+=(args.late_tracking_tube_bandwidth-tube_rate)*blend
                    joint_tube_rate=(joint_tube_rate or args.tracking_tube_bandwidth)*(1-blend)+args.late_tracking_tube_bandwidth*blend
                    orientation_tube_rate=(orientation_tube_rate or args.tracking_tube_bandwidth)*(1-blend)+args.late_tracking_tube_bandwidth*blend
                if elapsed_motion_time>=args.late_position_tube_start:
                    if not math.isfinite(args.late_position_tube_bandwidth) or args.late_position_tube_bandwidth<=0:
                        raise ValueError('Late position tube bandwidth must be positive')
                    # Freeze the other row bandwidths before changing position.
                    joint_tube_rate=joint_tube_rate or tube_rate
                    orientation_tube_rate=orientation_tube_rate or tube_rate
                    phase=min(1.,max(0.,(elapsed_motion_time-args.late_position_tube_start)/.1))
                    blend=phase**3*(10.-15.*phase+6.*phase**2)
                    tube_rate+=(args.late_position_tube_bandwidth-tube_rate)*blend
                tube_C,tube_d,tube_diagnostics=tracking_tube_constraints(q,qd,tube_q,tube_v,tube_a,bandwidth=tube_rate,joint_bandwidth=joint_tube_rate,orientation_bandwidth=orientation_tube_rate,
                    position_radius=(args.position_tube_radius if elapsed_motion_time>=args.position_tube_start else .195),
                    orientation_radius=(args.orientation_tube_radius
                        if elapsed_motion_time>=args.orientation_tube_start else .34))
                if elapsed_motion_time>=args.root_xy_tube_start:
                    from g1_root_xy_tube import root_xy_tube
                    xy_C,xy_d,xy_tube_diagnostics=root_xy_tube(q,qd,tube_q,tube_v,tube_a,
                        radius=args.root_xy_tube_radius,bandwidth=args.root_xy_tube_bandwidth)
                    task_acceleration_constraints=((xy_C,xy_d) if task_acceleration_constraints is None else
                        (torch.cat((task_acceleration_constraints[0],xy_C),dim=1),
                         torch.cat((task_acceleration_constraints[1],xy_d),dim=1)))
                if elapsed_motion_time >= args.joint_tube_end:
                    # No retry: choose this auxiliary constraint before the
                    # sole solve. Joint limits and source fidelity are separate.
                    tube_C=tube_C.clone();tube_d=tube_d.clone()
                    tube_C[:,0]=0.;tube_d[:,0]=0.
                if elapsed_motion_time>=args.position_tube_end:
                    # Keep physical constraints and original-source validation;
                    # only this auxiliary acceleration barrier is neutralized.
                    tube_C=tube_C.clone();tube_d=tube_d.clone()
                    tube_C[:,1]=0.;tube_d[:,1]=0.
                if elapsed_motion_time>=args.orientation_tube_end:
                    # Preserve the three-row recovery interface. This neutralizes
                    # only the auxiliary orientation CBF, not any physical row
                    # or the independent original-source fidelity auditor.
                    tube_C=tube_C.clone();tube_d=tube_d.clone()
                    tube_C[:,2]=0.;tube_d[:,2]=0.
                if args.tracking_tube_soft_weight and control_time >= args.tracking_tube_soft_start:
                    if args.late_tube_soft_weight is not None:
                        phase=min(1.,max(0.,(elapsed_motion_time-args.late_tube_soft_weight_start)/.06))
                        blend=phase**3*(10.-15.*phase+6.*phase**2)
                        tracking_tube_solver.weight=args.tracking_tube_soft_weight+blend*(args.late_tube_soft_weight-args.tracking_tube_soft_weight)
                    tube_scale = tube_d.new_tensor([.195**2,
                        (args.position_tube_radius if elapsed_motion_time>=args.position_tube_start else .195)**2,
                        math.sin((args.orientation_tube_radius if elapsed_motion_time>=args.orientation_tube_start else .34)/2)**2])
                    acceleration_qp.solver.tube = (tube_C, tube_d, tube_scale)
                else:
                    task_acceleration_constraints=((tube_C,tube_d) if task_acceleration_constraints is None else
                        (torch.cat((task_acceleration_constraints[0],tube_C),dim=1),torch.cat((task_acceleration_constraints[1],tube_d),dim=1)))
            if (elapsed_motion_time > args.soft_joint_braking_start
                    and acceleration_qp.solver.tube is not None):
                from g1_soft_joint_braking import soft_joint_braking_rows
                brake_phase=min(1.,(elapsed_motion_time-args.soft_joint_braking_start)/.06)
                soft_joint_braking_weight=args.soft_joint_braking_weight*brake_phase**3*(10-15*brake_phase+6*brake_phase**2)
                brake_C,brake_d=soft_joint_braking_rows(joint_position,joint_velocity,
                    joint_position_limits[:,:,0],joint_position_limits[:,:,1],
                    [G1_JOINT_NAMES.index(name) for name in args.soft_joint_braking_joints],
                    bandwidth=args.joint_limit_barrier_bandwidth)
                acceleration_qp.solver.braking=(brake_C,brake_d,soft_joint_braking_weight)
            acceleration_qp.priority_wrench_reference = (args.tracking_profile is not None
                and (retain_source_support or (dynamic_unloading and control_time >= .92)))
            if task_feedback_enabled and (elapsed_motion_time>=args.centroidal_preview_start):
                if 'late_centroidal_preview' not in locals():
                    if args.centroidal_preview_walking:
                        from walking_centroidal_preview import CentroidalPreview
                    elif args.centroidal_preview_persist_stance:
                        from persistent_centroidal_preview import CentroidalPreview
                    else:
                        from late_centroidal_preview import CentroidalPreview
                    late_centroidal_preview=CentroidalPreview(fabric,motion_sampler,_to_numpy(motion_planned_modes),motion_fps)
                if args.centroidal_preview_walking:
                    late_centroidal_preview.reference_clock=lambda offset: (
                        motion_start_frame+motion_fps*args.motion_speed*smooth_clock(wall_motion_time+offset,args.clock_transition_start,args.clock_transition_ramp,args.clock_final_speed,args.clock_second_start,args.clock_second_ramp,args.clock_second_speed,args.clock_third_start,args.clock_third_ramp,args.clock_third_speed)[0],
                        motion_fps*args.motion_speed*smooth_clock(wall_motion_time+offset,args.clock_transition_start,args.clock_transition_ramp,args.clock_final_speed,args.clock_second_start,args.clock_second_ramp,args.clock_second_speed,args.clock_third_start,args.clock_third_ramp,args.clock_third_speed)[1],
                        motion_fps*args.motion_speed*smooth_clock(wall_motion_time+offset,args.clock_transition_start,args.clock_transition_ramp,args.clock_final_speed,args.clock_second_start,args.clock_second_ramp,args.clock_second_speed,args.clock_third_start,args.clock_third_ramp,args.clock_third_speed)[2])
                try:
                    preview_acc,preview_rate=late_centroidal_preview.update(control_time,frame_position,q,qd,
                        live_feet,int(output.contact_mode.item()),contact_patches,friction_G,friction_b)
                except RuntimeError as exc:
                    args.output_path.with_suffix('.preview.json').write_text(json.dumps(late_centroidal_preview.records,indent=2)+'\n')
                    save_reference_failure(args,exc,control_time,q,qd,foot_normal_forces,tracking_history,servo_history)
                    raise
                preview_phase=min(1.,max(0.,(elapsed_motion_time-args.centroidal_preview_start)/.1))
                preview_blend=preview_phase**3*(10-15*preview_phase+6*preview_phase**2)
                for request in fabric.dynamics_tasks.last_requests:
                    if request.name=='center_of_mass':
                        axes=slice(0,2) if args.centroidal_preview_walking else slice(None)
                        request.desired[:,axes]=(1-preview_blend)*request.desired[:,axes]+preview_blend*q.new_tensor(preview_acc)[None,axes]
                    if request.name=='centroidal_momentum' and args.centroidal_preview_momentum:
                        request.desired[:]=(1-preview_blend)*request.desired+preview_blend*q.new_tensor(preview_rate)[None]
                if args.centroidal_preview_momentum:
                    priority_tasks=tuple(dict.fromkeys(priority_tasks+('centroidal_momentum',)))
                    priority_levels=None
                if control_updates%10==0:
                    args.output_path.with_suffix('.preview.json').write_text(json.dumps(late_centroidal_preview.records,indent=2)+'\n')
            if task_feedback_enabled and (elapsed_motion_time>=args.minimum_support_acceleration_start and int(output.contact_mode.item())!=0 and float(foot_normal_forces.sum())>=20.):
                support_phase=min(1.,max(0.,(elapsed_motion_time-args.minimum_support_acceleration_start)/.06))
                support_blend=support_phase**3*(10-15*support_phase+6*support_phase**2)
                if not math.isfinite(args.minimum_support_acceleration) or not -9.81<args.minimum_support_acceleration<=0:
                    raise ValueError('Minimum support acceleration must be in (-9.81,0]')
                if not math.isfinite(args.minimum_support_acceleration_metric_scale) or args.minimum_support_acceleration_metric_scale<=0:
                    raise ValueError('Support acceleration metric scale must be positive')
                for request in fabric.dynamics_tasks.last_requests:
                    if request.name=='center_of_mass':
                        supported=request.desired[:,2].clamp_min(args.minimum_support_acceleration)
                        request.desired[:,2]+=support_blend*(supported-request.desired[:,2])
                        request.metric[:,2,2]*=1+(args.minimum_support_acceleration_metric_scale-1)*support_blend
            if (task_feedback_enabled and args.maximum_support_acceleration is not None
                    and elapsed_motion_time >= args.maximum_support_acceleration_start
                    and int(output.contact_mode.item()) != 0 and float(foot_normal_forces.sum()) >= 20.):
                from g1_com_vertical_limit import limit_upward_target
                cap_phase=min(1.,max(0.,(elapsed_motion_time-args.maximum_support_acceleration_start)/args.maximum_support_acceleration_ramp))
                cap_blend=cap_phase**3*(10-15*cap_phase+6*cap_phase**2)
                for request in fabric.dynamics_tasks.last_requests:
                    if request.name=='center_of_mass':
                        request.desired=limit_upward_target(request.desired,args.maximum_support_acceleration,cap_blend)
            if (task_feedback_enabled and args.com_upward_speed_limit is not None
                    and elapsed_motion_time >= args.com_upward_speed_start
                    and int(output.contact_mode.item()) != 0 and float(foot_normal_forces.sum()) >= 20.):
                from g1_com_vertical_limit import limit_upward_speed_target
                speed_phase=min(1.,max(0.,(elapsed_motion_time-args.com_upward_speed_start)/args.com_upward_speed_ramp))
                speed_blend=speed_phase**3*(10-15*speed_phase+6*speed_phase**2)
                if elapsed_motion_time > args.com_upward_speed_end:
                    speed_end_phase = min(1., (elapsed_motion_time-args.com_upward_speed_end)/args.com_upward_speed_end_ramp)
                    speed_blend *= 1.-speed_end_phase**3*(10-15*speed_end_phase+6*speed_end_phase**2)
                for request in fabric.dynamics_tasks.last_requests:
                    if request.name=='center_of_mass':
                        velocity_z=(request.A@qd[...,None]).squeeze(-1)[:,2]
                        request.desired=limit_upward_speed_target(request.desired,velocity_z,
                            args.com_upward_speed_limit,args.com_upward_speed_bandwidth,
                            args.com_upward_speed_minimum_acceleration,speed_blend)
            if task_feedback_enabled and (elapsed_motion_time>=args.configuration_tracking_start):
                if not unified_reference or args.configuration_tracking_ramp<=0 or not math.isfinite(args.configuration_tracking_ramp):
                    raise ValueError('Configuration tracking requires a unified reference and a positive finite ramp')
                from configuration_tracking import configuration_tracking_request
                configuration_request=configuration_tracking_request(q,qd,whole_body_reference,fabric.dynamics_tasks.last_requests,args.configuration_angular_weight)
                configuration_phase=min(1.,max(0.,(elapsed_motion_time-args.configuration_tracking_start)/args.configuration_tracking_ramp))
                configuration_blend=configuration_phase**3*(10-15*configuration_phase+6*configuration_phase**2)
                if configuration_blend>0:
                    for request in fabric.dynamics_tasks.last_requests:
                        if request.name in priority_tasks and not (args.configuration_keep_feet and request.name=='feet_pose'):
                            request.metric*=1-configuration_blend
                    configuration_request.metric*=configuration_blend
                    fabric.dynamics_tasks.last_requests.append(configuration_request)
                    priority_tasks=tuple(dict.fromkeys(priority_tasks+(configuration_request.name,)))
                    priority_levels=None
            if task_feedback_enabled and (elapsed_motion_time>=args.tracking_margin_start):
                from tracking_margin import margin_weights,allocate_margin_weights
                margin_q,margin_v,_=tube_source_sampler.kinematics(frame_position,frame_rate,frame_acceleration)
                margin_scale,margin_risk=margin_weights(q,qd,margin_q,margin_v,
                    reaction_time=args.tracking_margin_reaction_time,maximum=args.tracking_margin_maximum)
                if 'filtered_margin_scale' not in locals():filtered_margin_scale=torch.ones_like(margin_scale)
                filtered_margin_scale+=(1-math.exp(-FABRIC_TIMESTEP/.025))*(margin_scale-filtered_margin_scale)
                margin_phase=min(1.,max(0.,(elapsed_motion_time-args.tracking_margin_start)/.1))
                allocate_margin_weights(fabric.dynamics_tasks.last_requests,filtered_margin_scale,
                    margin_phase**3*(10-15*margin_phase+6*margin_phase**2))
            if task_feedback_enabled and (elapsed_motion_time>=args.pelvis_orientation_replace_start):
                if not unified_reference or not math.isfinite(args.pelvis_orientation_replace_ramp) or args.pelvis_orientation_replace_ramp<=0:
                    raise ValueError('Pelvis orientation replacement requires unified references and a positive ramp')
                from pelvis_orientation_allocation import replace_torso_weight
                orientation_phase=min(1.,max(0.,(elapsed_motion_time-args.pelvis_orientation_replace_start)/args.pelvis_orientation_replace_ramp))
                orientation_blend=orientation_phase**3*(10-15*orientation_phase+6*orientation_phase**2)
                if orientation_blend>0:
                    replace_torso_weight(fabric.dynamics_tasks.last_requests,orientation_blend,
                        pelvis_scale=args.pelvis_orientation_replace_metric_scale)
                    if args.pelvis_orientation_replace_feedback_limit is not None:
                        from g1_foot_angular_feedback import angular_tracking_target
                        dynamics=fabric.dynamics_tasks
                        pelvis_index=dynamics.model.kinematics.get_link_index('pelvis')
                        for request in dynamics.last_requests:
                            if request.name!='source_pelvis_orientation':continue
                            rv,ra=dynamics.references(fabric.fabrics_features['pelvis_pose']['pelvis_pose_attractor'],q.new_zeros(len(q),6))
                            params=dynamics.config['body']
                            desired=angular_tracking_target(dynamics.last_snapshot['rotations'][:,pelvis_index],
                                _quat_xyzw_to_matrix(fabric.pelvis_pose_taskmap.orientation_target),
                                (request.A@qd[...,None]).squeeze(-1),rv[:,3:],ra[:,3:],
                                q.new_tensor(params['stiffness']),q.new_tensor(params['damping']),
                                args.pelvis_orientation_replace_feedback_limit)
                            request.desired+=orientation_blend*(desired-request.desired)
                    priority_tasks=tuple(dict.fromkeys(priority_tasks+('source_pelvis_orientation',)))
                    priority_levels=None
            root_task_weight = 0.
            root_task_axis_weights = q.new_zeros(len(q),3)
            if task_feedback_enabled and (elapsed_motion_time > args.root_position_task_start):
                root_phase = min(1., (elapsed_motion_time-args.root_position_task_start)
                                 / args.root_position_task_ramp)
                root_task_weight = args.root_position_task_weight*root_phase**3*(10-15*root_phase+6*root_phase**2)
                if args.root_position_task_reference == 'control':
                    if not unified_reference or whole_body_reference is None:
                        raise ValueError('Control root task requires a unified whole-body reference')
                if 'root_position_reference_sampler' not in locals():
                    from g1_root_position_reference import RootPositionReference
                    root_position_reference_sampler=RootPositionReference(motion_trajectory)
                root_task_reference=root_position_reference_sampler.sample(
                    frame_position,frame_rate,frame_acceleration,mode=args.root_position_task_reference,
                    control=whole_body_reference if unified_reference else None)
                root_request=root_position_request(q, qd, *root_task_reference,
                    root_task_weight,axes=args.root_position_task_axes)
                if args.root_position_task_late_weight is not None:
                    root_late_phase=min(1.,max(0.,(elapsed_motion_time-args.root_position_task_late_start)/args.root_position_task_late_ramp))
                    blend_root_position_weight(root_request,args.root_position_task_late_weight,
                        args.root_position_task_late_axes,root_late_phase**3*(10-15*root_late_phase+6*root_late_phase**2))
                root_task_axis_weights=root_request.metric.diagonal(dim1=-2,dim2=-1)*root_request.active
                fabric.dynamics_tasks.last_requests.append(root_request)
                priority_tasks = tuple(dict.fromkeys(priority_tasks+('root_position',)))
                if priority_levels is not None:
                    priority_levels = (priority_levels[0]+('root_position',),)+priority_levels[1:]
            shoulder_weight = shoulder_roll_multiplier(
                control_time, args.shoulder_roll_weight_start,
                args.shoulder_roll_weight_ramp, args.shoulder_roll_weight_scale)
            if task_feedback_enabled and (shoulder_weight > 1 and shoulder_guard is None):
                apply_shoulder_roll_weights(fabric.dynamics_tasks.last_requests,
                                            G1_JOINT_NAMES, shoulder_weight, args.shoulder_roll_weight_side)
            secondary_task_blend = 0.
            if task_feedback_enabled and (elapsed_motion_time > args.weighted_tasks_start):
                if priority_levels is not None or com_priority_projection is not None:
                    raise ValueError('Weighted transition requires a single unprojected priority stage')
                secondary_phase = min(1., (elapsed_motion_time-args.weighted_tasks_start)
                                      / args.weighted_tasks_ramp)
                secondary_task_blend = secondary_phase**3*(10-15*secondary_phase+6*secondary_phase**2)
                blend_secondary_task_metrics(fabric.dynamics_tasks.last_requests,
                                             priority_tasks, priority_rows, secondary_task_blend)
                priority_tasks, priority_rows = (), {}
            composition_snapshot = {}
            if recompose_tasks:
                from g1_priority_tracking import compose_tracking_policy, scale_secondary_tasks
                local_requests = scale_secondary_tasks(fabric.dynamics_tasks.last_requests,
                    priority_tasks, priority_rows, args.fabric_secondary_scale)
                # Keep numeric diagnostic values after the local leaves are
                # discarded. Re-evaluating the legacy attractor would lose
                # feedforward and all per-axis feedback/weight overrides.
                for request in local_requests:
                    if request.name == 'center_of_mass':
                        fabric.dynamics_tasks.last_snapshot['com_composed_desired'] = request.desired.detach().clone()
                        fabric.dynamics_tasks.last_snapshot['com_composed_metric'] = request.metric.detach().clone()
                if any(abs(control_time-stamp)<FABRIC_TIMESTEP/4 for stamp in args.qp_snapshot_times):
                    # Diagnostic arrays only: do not pass task objects to the QP.
                    # Capture the weighted leaves before composition clears them.
                    for request in local_requests:
                        for key in ('A', 'bias', 'desired', 'metric', 'active'):
                            composition_snapshot['composed_task_'+request.name+'_'+key] = (
                                _to_numpy(getattr(request, key)).copy())
                qdd_fabric, root_metric, qdd_dc, priority_diagnostics = compose_tracking_policy(
                    local_requests, mass, contact_jacobian, jdot_qdot,
                    output.active_row_mask, args.fabric_priority_mode,
                    fabric.dynamics_tasks.config['regularization'])
                fabric.dynamics_tasks.last_snapshot.update(priority_diagnostics)
                fabric.dynamics_tasks.last_snapshot['fabric_acc_metric'] = root_metric
                # Leaves are local policy inputs. Only (acceleration, metric)
                # crosses into inverse dynamics; no task-level QP stage.
                fabric.dynamics_tasks.last_requests = None
                qp_reference = qdd_fabric
                from dataclasses import replace as replace_projection_output
                active_contact_jac = contact_jacobian*output.active_row_mask[..., None]
                output = replace_projection_output(output, acc_dc=qdd_dc,
                    contact_residual_before=(active_contact_jac@qdd_fabric[..., None]).squeeze(-1)+jdot_qdot,
                    contact_residual_after=(active_contact_jac@qdd_dc[..., None]).squeeze(-1)+jdot_qdot)
            current_effort_limits=qp_effort_limits
            if elapsed_motion_time>=args.model_effort_start:
                effort_phase=min(1.,max(0.,(elapsed_motion_time-args.model_effort_start)/args.model_effort_ramp))
                effort_blend=effort_phase**3*(10-15*effort_phase+6*effort_phase**2)
                current_effort_limits=.8*(legacy_effort_limits+effort_blend*(effort_limits-legacy_effort_limits))
            if elapsed_motion_time>=args.synchronous_effort_budget_start:
                from synchronous_effort_budget import effort_fraction
                current_effort_limits=effort_limits*effort_fraction(elapsed_motion_time,args.synchronous_effort_budget_start,
                    args.synchronous_effort_budget_fraction,FABRIC_TIMESTEP,PD_TIMESTEP,
                    direct_wbc=args.direct_wbc,tracking_servo=args.tracking_servo)
            previous_validated_torque = qp_output.joint_torque.detach().clone()
            torque_continuity_weight = 0.
            torque_continuity_mask = None
            if torque_continuity_solver is not None:
                torque_phase = min(1., max(0., (elapsed_motion_time-args.torque_continuity_start)
                                           / args.torque_continuity_ramp))
                torque_continuity_weight = args.torque_continuity_weight*torque_phase**3*(10-15*torque_phase+6*torque_phase**2)
                if elapsed_motion_time > args.torque_continuity_end:
                    end_phase = min(1., (elapsed_motion_time-args.torque_continuity_end)/args.torque_continuity_end_ramp)
                    end_blend = end_phase**3*(10-15*end_phase+6*end_phase**2)
                    torque_continuity_weight += end_blend*(args.torque_continuity_end_weight-torque_continuity_weight)
                if args.torque_continuity_stance_only:
                    from g1_torque_continuity import stance_torque_weights
                    torque_continuity_mask=stance_torque_weights(output.contact_mode,previous_validated_torque)
                torque_continuity_solver.update(previous_validated_torque, torque_continuity_weight,
                                                torque_continuity_mask)
            if acceleration_only:
                acceleration_qp.solver.single_proximal = (args.qp_single_proximal
                    if control_time >= args.qp_single_proximal_start else 0.)
                acceleration_qp.solver.equilibration_cap = (args.qp_equilibration_cap
                    if control_time >= args.qp_equilibration_start else 1.)
                if control_time >= args.qp_late_equilibration_start:
                    acceleration_qp.solver.equilibration_cap = args.qp_late_equilibration_cap
                if control_time >= args.qp_final_equilibration_start:
                    acceleration_qp.solver.equilibration_cap = args.qp_final_equilibration_cap
                if control_time >= args.qp_late_iteration_start:
                    acceleration_qp.solver.max_iter = args.qp_late_iterations
                if control_time >= args.qp_cold_start_time:
                    acceleration_qp.solver.reset()
            qp_wall_start = time.perf_counter()
            if args.qp_base_linear_acceleration_weight is not None:
                qp_acceleration_weight[:3] = linear_acceleration_weight(
                    args.qp_base_acceleration_weight, args.qp_base_linear_acceleration_weight,
                    control_time, start=args.qp_base_linear_weight_start,
                    ramp=args.qp_base_linear_weight_ramp)
                acceleration_qp.acc_weight = qp_acceleration_weight
            if args.com_acceleration_preservation_weight:
                acceleration_qp.acc_weight = preserve_com_acceleration(qp_acceleration_weight,
                    fabric.dynamics_tasks.last_snapshot['com_jac'], args.com_acceleration_preservation_weight)
            else:
                acceleration_qp.acc_weight = qp_acceleration_weight
            if args.qp_fabric_metric:
                acceleration_qp.acc_weight = fabric.dynamics_tasks.last_snapshot['fabric_acc_metric']
                qp_reference = qdd_fabric
            live_com_feedback_scale = com_feedback_scale(args.qp_com_feedback_scale, control_time,
                start=args.qp_com_feedback_start, ramp=args.qp_com_feedback_ramp)
            if live_com_feedback_scale:
                acceleration_qp.acc_weight, qp_reference = track_com_feedback(
                    acceleration_qp.acc_weight, qp_reference, fabric.dynamics_tasks.last_snapshot,
                    live_com_feedback_scale, axes=args.qp_com_feedback_axes)
            command_acc_correction = (torch.zeros_like(qdd_fabric) if acceleration_only
                                      else qp_reference - qdd_dc)
            qp_solve=base_qp_solve
            acceleration_qp.tube_recovery_used=False
            acceleration_qp.orientation_tube_recovery_bandwidth=float('nan')
            if args.relax_joint_tube and (elapsed_motion_time>=args.tracking_tube_start and elapsed_motion_time<args.tracking_tube_end):
                from tube_recovery import solve_with_joint_tube_recovery
                qp_solve=partial(solve_with_joint_tube_recovery,acceleration_qp,tube_C=tube_C,tube_d=tube_d,recovery_metric=args.tube_recovery_metric)
                if args.recover_orientation_tube:
                    from tube_recovery import solve_with_orientation_tube_recovery
                    qp_solve=partial(solve_with_orientation_tube_recovery,acceleration_qp,tube_C=tube_C,tube_d=tube_d,
                        tube_diagnostics=tube_diagnostics,orientation_bandwidth=(orientation_tube_rate or tube_rate),
                        recovery_metric=args.tube_recovery_metric)
            if elapsed_motion_time>=args.full_body_preview_start:
                if not args.direct_wbc or not args.tracking_servo or str(args.device)!='cpu' or args.contact_pause_end is not None or args.extra_contact_pause or abs(FABRIC_TIMESTEP-.002)>1e-12:
                    raise ValueError('Full-body preview requires synchronous 500 Hz direct CPU tracking without a pause clock')
                if 'full_body_preview' not in locals():
                    from preview_process import ProcessPreview
                    from online_full_body_preview import OnlineFullBodyPreview
                    from fabrics_sim.reference.manager import contact_sequence_modes
                    preview_modes=contact_sequence_modes(_to_numpy(motion_contact_sequence),len(motion_trajectory))
                    preview_geometry=[tuple(_to_numpy(x) for x in row) for row in foot_landing_geometry]
                    if (args.full_body_preview_state_feedback or args.full_body_preview_release_points) and not args.full_body_preview_fine_control:
                        raise ValueError('State feedback and contact release require fine preview control')
                    if args.full_body_preview_fine_control and not args.full_body_preview_collision_barrier:
                        raise ValueError('Fine preview control requires future collision barriers')
                    if args.full_body_preview_exact_rotation and not (args.full_body_preview_fine_control and args.full_body_preview_release_points):
                        raise ValueError('Exact preview rotation requires fine control and contact release')
                    if args.full_body_preview_recover_joint_reserve and not args.full_body_preview_exact_rotation:
                        raise ValueError('Joint reserve recovery requires exact preview rotation')
                    if args.full_body_preview_damp_first_control and not args.full_body_preview_recover_joint_reserve:
                        raise ValueError('First-control damping requires the joint-reserve recovery predictor')
                    if args.full_body_preview_surface_kinematics and not args.full_body_preview_separating_points:
                        raise ValueError('Surface kinematics requires separation-aware fine preview')
                    if args.full_body_preview_separating_points and not args.full_body_preview_damp_first_control:
                        raise ValueError('Separating-point prediction requires first-control damping')
                    preview_configuration=dict(urdf=fabric.urdf_path,names=G1_JOINT_NAMES,
                        reference_path=str(args.output_path.with_suffix('.control_reference.npz')),
                        modes=preview_modes,geometry=preview_geometry,
                        limits=_to_numpy(joint_position_limits)[0],effort=_to_numpy(current_effort_limits).ravel(),
                        velocity=_to_numpy(joint_velocity_limits).ravel(),horizon=args.full_body_preview_horizon,
                        guarded=args.full_body_preview_guarded,warm_start=args.full_body_preview_warm_start,fine_control=args.full_body_preview_fine_control,release_points=args.full_body_preview_release_points,
                        native_contact_timing=args.full_body_preview_native_contact_timing,
                        exact_rotation=args.full_body_preview_exact_rotation,
                        recover_joint_reserve=args.full_body_preview_recover_joint_reserve,
                        damp_first_control=args.full_body_preview_damp_first_control,
                        separating_points=args.full_body_preview_separating_points,
                        surface_kinematics=args.full_body_preview_surface_kinematics,
                        contact_timing_start_frame=int(math.ceil(args.full_body_preview_start*motion_fps)),
                        contact_schedule_path=str(args.output_path.with_suffix('.preview_contact_schedule.npz')))
                    if args.full_body_preview_collision_barrier:
                        tasks=fabric.dynamics_tasks
                        if not tasks.wrist_hip_barrier or tasks.convex_wrist_geometry is not None:
                            raise ValueError('Preview collision barrier requires the enabled runtime sphere covers')
                        sphere_names=tasks.sphere_names
                        centers=_to_numpy(tasks.sphere_centers);radii=_to_numpy(tasks.sphere_radii)
                        spheres={name:dict(link=tasks.model.kinematics.link_names[tasks.sphere_indices[i]],center=centers[i].tolist(),radius=float(radii[i])) for i,name in enumerate(sphere_names)}
                        pairs=[[sphere_names[int(a)],sphere_names[int(b)]] for a,b in _to_numpy(tasks.pairs)[tasks.original_collision_pairs:tasks.barrier_pair_count]]
                        preview_configuration['collision_geometry']=dict(spheres=spheres,pairs=pairs,safe_distance=tasks.config['collision']['safe_distance'])
                    preview_planner=ProcessPreview(preview_configuration,args.output_path.with_suffix('.planner_worker.log'))
                    full_body_preview=OnlineFullBodyPreview(preview_planner,args.output_path.with_suffix('.full_body_preview.json'),period=args.full_body_preview_period,track_wrench=args.full_body_preview_track_wrench,reuse_verified=args.full_body_preview_reuse_verified,follow_plan=args.full_body_preview_fine_control,allow_projection_correction=args.full_body_preview_state_feedback)
                from observed_preview_patch import observed_native_patch_indices
                try:
                    initial_preview_patch=observed_native_patch_indices(_to_numpy(live_feet),preview_geometry,tuple(map(_to_numpy,contact_data)),int(output.contact_mode.item()))
                except ValueError as error:
                    full_body_preview.observation_failure(control_time,frame_position,error)
                    mismatch_path=args.output_path.with_suffix('.initial_patch_mismatch.npz')
                    if not mismatch_path.exists():
                        np.savez_compressed(mismatch_path,time=control_time,frame=frame_position,q=_to_numpy(q),v=_to_numpy(qd),feet=_to_numpy(live_feet),mode=int(output.contact_mode.item()),
                            **{key:_to_numpy(value) for key,value in zip(('forces','points','normals','distances','counts','starts'),contact_data)})
                else:
                    qp_solve=partial(full_body_preview.solve,qp_solve,acceleration_qp,q,qd,control_time,frame_position,frame_rate,initial_patch_indices=initial_preview_patch)
            if shoulder_guard is not None:
                qp_solve = partial(shoulder_guard.solve, qp_solve, acceleration_qp,
                    joint_names=G1_JOINT_NAMES, multiplier=shoulder_weight, side=args.shoulder_roll_weight_side)
            saved_objective_weights = acceleration_qp.wrench_weight, acceleration_qp.torque_weight
            joint_stop_directions = None
            if elapsed_motion_time >= args.observed_joint_stop_start:
                from fabrics_sim.solver.joint_stop import observed_stop_directions
                from g1_joint_stop_mode import loaded_stop_directions
                joint_stop_directions = observed_stop_directions(
                    joint_position, joint_velocity, joint_position_limits[:, :, 0], joint_position_limits[:, :, 1])
                joint_stop_directions = loaded_stop_directions(joint_stop_directions,
                    qp_output.predicted_acc[:, 6:], qp_output.joint_stop_reaction)
                selected_stops = [G1_JOINT_NAMES.index(name) for name in args.observed_joint_stop_joints]
                stop_mask = torch.zeros_like(joint_stop_directions)
                stop_mask[:, selected_stops] = 1
                joint_stop_directions *= stop_mask
            if args.qp_task_only_objective:
                acceleration_qp.torque_weight = 0.
                if not acceleration_qp.priority_wrench_reference:
                    acceleration_qp.wrench_weight = 0.
            qp_output = qp_solve(
                acc_reference=qp_reference,
                mass_matrix=mass,
                bias_force=bias_force,
                contact_jacobian=contact_jacobian,
                jdot_qdot=jdot_qdot,
                contact_mode=output.contact_mode,
                acc_lower=command_acc_lower,
                acc_upper=command_acc_upper,
                effort_limits=current_effort_limits,
                known_joint_torque=-kd*joint_velocity if args.direct_wbc and not args.tracking_servo else None,
                servo_bias=None if args.fixed_joint_test or args.direct_wbc else servo_bias,
                servo_acc_gain=None if args.fixed_joint_test or args.direct_wbc else servo_acc_gain,
                command_acc_correction=command_acc_correction,
                contact_wrench_reference=wrench_reference,
                contact_wrench_weights=wrench_weights,
                track_predicted_base=args.qp_track_predicted_base and not args.fixed_joint_test,
                friction_G=friction_G,
                friction_b=friction_b,
                task_requests=(None if acceleration_only else [composed_policy] if args.g1_md else
                    None if fabric.dynamics_tasks is None else fabric.dynamics_tasks.last_requests),
                task_tracks_predicted=not args.task_track_nominal,
                priority_tasks=('center_of_mass',) if args.g1_com_priority else priority_tasks,
                priority_requests=[r for r in fabric.dynamics_tasks.last_requests
                    if r.name == 'center_of_mass'] if args.g1_com_priority and not acceleration_only else None,
                priority_levels=priority_levels,
                priority_rows=priority_rows,
                priority_projections={"center_of_mass": com_priority_projection}
                    if com_priority_projection is not None else None,
                acceleration_constraints=task_acceleration_constraints,
                contact_row_mask=contact_row_mask,
                # Command bounds alone do not bound acceleration after the
                # contact-consistent response to explicit PD feedback.
                predicted_acc_limit=acceleration_limits if args.limit_predicted_acceleration else None,
                predicted_acc_lower=physical_lower if physical_braking_active else None,
                predicted_acc_upper=physical_upper if physical_braking_active else None,
                joint_stop_directions=joint_stop_directions,
                joint_stop_reaction_weight=args.observed_joint_stop_weight,
            )
            acceleration_qp.wrench_weight, acceleration_qp.torque_weight = saved_objective_weights
            for snapshot_time in args.qp_snapshot_times:
                if abs(control_time-snapshot_time)<FABRIC_TIMESTEP/4:
                    snapshot_arrays={'q':_to_numpy(q),'v':_to_numpy(qd),'time':control_time,
                        'reference_q':_to_numpy(reference_q),'reference_v':_to_numpy(reference_v),
                        'predicted_acc':_to_numpy(qp_output.predicted_acc),'feet':_to_numpy(live_feet),
                        'contact_mode':_to_numpy(output.contact_mode),'forces':_to_numpy(foot_normal_forces)}
                    snapshot_problem = (acceleration_qp.last_priority_problems[0]
                        if acceleration_qp.last_priority_problems else acceleration_qp.last_problem)
                    for key,value in vars(snapshot_problem).items():
                        if isinstance(value,torch.Tensor):snapshot_arrays['problem_'+key]=_to_numpy(value)
                    snapshot_arrays['problem_solution'] = _to_numpy(acceleration_qp.last_full_solution)
                    snapshot_arrays['joint_stop_reaction'] = _to_numpy(qp_output.joint_stop_reaction)
                    snapshot_arrays['joint_stop_directions'] = _to_numpy(
                        torch.zeros_like(joint_position) if joint_stop_directions is None else joint_stop_directions)
                    if getattr(acceleration_qp.solver, 'last_augmented_problem', None) is not None:
                        for key, value in vars(acceleration_qp.solver.last_augmented_problem).items():
                            if isinstance(value, torch.Tensor):snapshot_arrays['soft_problem_'+key]=_to_numpy(value)
                        snapshot_arrays['soft_problem_solution'] = _to_numpy(acceleration_qp.solver.last_augmented_solution)
                    for request in (() if acceleration_only else fabric.dynamics_tasks.last_requests):
                        for key in ('A','bias','desired','metric','active'):
                            snapshot_arrays['task_'+request.name+'_'+key]=_to_numpy(getattr(request,key))
                    snapshot_arrays.update(composition_snapshot)
                    np.savez_compressed(args.output_path.with_suffix(f'.snapshot_{snapshot_time:.3f}.npz'),**snapshot_arrays)
            if acceleration_qp.tube_recovery_used:
                saved_C,saved_d=task_acceleration_constraints
                task_acceleration_constraints=(torch.cat((saved_C[:,:-3],saved_C[:,-2:]),1),torch.cat((saved_d[:,:-3],saved_d[:,-2:]),1))
                print(f'Joint tracking tube softened at {control_time:.3f}s; physical QP validated={bool(qp_output.validated.all())}',flush=True)
            if args.g1_md and fabric.dynamics_tasks is not None:
                # Diagnostics remain task-space quantities even though the QP
                # now tracks one metric-preserving root policy, not leaf tasks.
                qp_output.task_residuals = {
                    request.name: request.residual(qp_output.predicted_acc)
                    for request in fabric.dynamics_tasks.last_requests
                }
            # Reading validation synchronizes GPU work before measuring solve
            # latency. A 2 ms simulation period does not certify 2 ms wall time.
            qp_validated = bool(qp_output.validated.all())
            qp_task_residuals = qp_output.task_residuals or {}
            com_diagnostic = {"qp_wall_seconds": time.perf_counter()-qp_wall_start,
                              "contact_sample_dt": contact_sample_dt}
            if math.isfinite(args.root_xy_tube_start):
                xy_active=(task_feedback_enabled and elapsed_motion_time>=max(args.root_xy_tube_start,args.tracking_tube_start)
                           and elapsed_motion_time<args.tracking_tube_end)
                com_diagnostic['root_xy_tube_diagnostics']=(_to_numpy(xy_tube_diagnostics).copy()
                    if xy_active else np.full((len(q),3),np.nan))
            if math.isfinite(args.soft_joint_braking_start):
                brake_slack=acceleration_qp.solver.last_braking_slack
                com_diagnostic['soft_joint_braking_weight']=soft_joint_braking_weight
                com_diagnostic['soft_joint_braking_slack']=(_to_numpy(brake_slack).copy()
                    if brake_slack is not None else np.zeros((1,2*len(args.soft_joint_braking_joints))))
            if args.late_tube_soft_weight is not None:
                com_diagnostic['tracking_tube_soft_weight']=float(tracking_tube_solver.weight)
            if args.late_foot_linear_feedback_limit is not None:
                com_diagnostic['foot_linear_feedback_blend']=linear_foot_blend
                com_diagnostic['foot_linear_feedback_limit']=args.late_foot_linear_feedback_limit
            if args.late_foot_angular_feedback_limit is not None:
                com_diagnostic['foot_angular_feedback_blend']=angular_blend
                com_diagnostic['foot_angular_feedback_limit']=args.late_foot_angular_feedback_limit
            if args.tracking_profile == 'amp-full':
                com_diagnostic['qp_native_solve_count'] = np.asarray(
                    acceleration_qp.solver.last_solve_counts, dtype=np.int64)
                com_diagnostic['fabric_task_requests_cached'] = (
                    fabric.dynamics_tasks.last_requests is not None)
            if args.shoulder_roll_weight_scale > 1:
                com_diagnostic['shoulder_roll_weight_multiplier'] = (
                    shoulder_weight if shoulder_guard is None else shoulder_guard.diagnostic['shoulder_guard_multiplier'])
            if shoulder_guard is not None:
                com_diagnostic.update(shoulder_guard.diagnostic)
            if math.isfinite(args.weighted_tasks_start):
                com_diagnostic['secondary_task_blend'] = secondary_task_blend
            if math.isfinite(args.support_flatten_start):
                com_diagnostic['support_flatten_weight'] = _to_numpy(support_flatten_weights).copy()
            if math.isfinite(args.root_position_task_start):
                com_diagnostic['root_position_task_weight'] = root_task_weight
                if args.root_position_task_late_weight is not None:
                    com_diagnostic['root_position_task_axis_weights'] = _to_numpy(root_task_axis_weights).copy()
            if math.isfinite(args.stance_posture_metric_start):
                com_diagnostic['stance_posture_metric_scale'] = stance_posture_scale
            if math.isfinite(args.rolling_contact_kinematics_start):
                com_diagnostic.update(rolling_bias_correction=_to_numpy(rolling_bias_correction).copy(),
                                      rolling_bias_matched=_to_numpy(rolling_bias_matched).copy())
            com_diagnostic.update(torque_continuity_weight=torque_continuity_weight,
                torque_step_change=_to_numpy(qp_output.joint_torque-previous_validated_torque).copy())
            if torque_continuity_mask is not None:
                com_diagnostic['torque_continuity_mask']=_to_numpy(torque_continuity_mask).copy()
            if mocap_transition is not None:
                com_diagnostic.update(mocap_transition_phase=mocap_transition.phase,
                    mocap_transition_unload_weight=mocap_transition.unload_weight(control_time))
            predicted_joint_velocity = joint_velocity + FABRIC_TIMESTEP*qp_output.predicted_acc[:,6:]
            velocity_violation = (predicted_joint_velocity.abs()-joint_velocity_limits).clamp_min(0.)
            if args.joint_velocity_limits:
                qp_validated = qp_validated and bool((velocity_violation <= 1e-4).all())
            com_diagnostic['predicted_joint_velocity'] = _to_numpy(predicted_joint_velocity).copy()
            com_diagnostic['joint_velocity_limits'] = _to_numpy(joint_velocity_limits).copy()
            com_diagnostic['qp_base_linear_acceleration_weight'] = float(qp_acceleration_weight[0])
            com_diagnostic['qp_com_feedback_scale'] = live_com_feedback_scale
            if args.qp_com_feedback_scale:
                com_diagnostic['qp_com_feedback_axes'] = np.asarray(args.qp_com_feedback_axes)
            com_diagnostic['qp_fabric_metric'] = args.qp_fabric_metric
            com_diagnostic['joint_velocity_violation'] = _to_numpy(velocity_violation).copy()
            com_diagnostic['measured_joint_velocity_violation'] = _to_numpy((joint_velocity.abs()-joint_velocity_limits).clamp_min(0.)).copy()
            com_diagnostic['joint_velocity_limits_enabled'] = args.joint_velocity_limits
            # Simulator joint stops are always physical, but the predictive
            # braking bounds are optional. Record which constraints the QP
            # actually saw; a configured bandwidth alone does not enable them.
            com_diagnostic['physical_braking_active'] = physical_braking_active
            com_diagnostic['physical_joint_acceleration_lower'] = _to_numpy(physical_lower[:, 6:]).copy()
            com_diagnostic['physical_joint_acceleration_upper'] = _to_numpy(physical_upper[:, 6:]).copy()
            com_diagnostic['observed_joint_stop_directions'] = _to_numpy(
                torch.zeros_like(joint_position) if joint_stop_directions is None else joint_stop_directions).copy()
            com_diagnostic['joint_stop_reaction'] = _to_numpy(qp_output.joint_stop_reaction).copy()
            com_diagnostic['joint_stop_constraint_error'] = _to_numpy(acceleration_qp.last_joint_stop_constraint_error).copy()
            com_diagnostic.update(sole_barrier_diagnostic)
            com_diagnostic.update(knee_guard_diagnostic)
            com_diagnostic.update(landing_diagnostic)
            com_diagnostic.update(momentum_bound_diagnostic)
            com_diagnostic.update(com_replan_diagnostic)
            if reference_manager is not None:
                com_diagnostic.update(reference_settle_time=reference_manager.settle_time,
                    reference_load_confirmation=reference_manager.confirmed*FABRIC_TIMESTEP,
                    sole_min_height=_to_numpy(observed_corners[...,2].amin(-1)),
                    sole_max_height=_to_numpy(observed_corners[...,2].amax(-1)))
            if args.observed_contact_patches:
                patch_desired=np.full((2,3),np.nan);patch_residual=np.full((2,3),np.nan)
                patch_flatten=np.zeros(2,dtype=bool)
                for foot,task in contact_patch_tasks.items():
                    width=task.desired.shape[-1]
                    patch_desired[foot,:width]=_to_numpy(task.desired[0])
                    patch_value = (task.residual(qp_output.predicted_acc) if recompose_tasks
                                   else qp_task_residuals[task.name])
                    patch_residual[foot,:width]=_to_numpy(patch_value[0])
                    patch_flatten[foot]=contact_patches[foot].flatten
                com_diagnostic.update(contact_orientation_desired=patch_desired,
                                      contact_orientation_residual=patch_residual,
                                      contact_flatten=patch_flatten)
            if args.wrist_hip_barrier:
                com_diagnostic['collision_acceleration_constraint_error'] = _to_numpy(
                    acceleration_qp.last_acceleration_constraint_error).copy()
            if args.qp_solver in ('osqp','daqp'):
                com_diagnostic["qp_solver_status"] = np.asarray(acceleration_qp.solver.last_status)
                com_diagnostic["qp_workspace_reused"] = np.asarray(acceleration_qp.solver.last_workspace_reused)
                com_diagnostic["qp_priority_solver_status"] = np.asarray(
                    acceleration_qp.last_priority_solver_status or ['inactive'])
            com_diagnostic['source_swing_position_priority_active'] = np.asarray(source_priority is not None and args.source_swing_position_priority)
            com_diagnostic['source_swing_pose_priority_active'] = np.asarray(source_priority is not None and args.source_swing_pose_priority)
            if args.transfer_balance_priority:
                # Keep the history schema constant before and after the first
                # transfer, so delayed activation does not drop these fields.
                com_diagnostic.update(
                    balance_priority_active=np.asarray(acceleration_qp.last_priority_errors is not None),
                    swing_height_priority_active=np.asarray(height_priority),
                    swing_com_priority_active=np.asarray(swing_com_priority),
                    priority_stage_constraint_error=np.zeros(1, dtype=np.float32),
                    priority_lock_error=np.zeros(1, dtype=np.float32))
            com_diagnostic.update(priority_error_diagnostics(acceleration_qp.last_priority_errors))
            if args.diagnose_contacts:
                # Retain the instantaneous coordinates actually constrained by
                # the QP, including observed point/edge contact frames. A small
                # acceleration residual does not imply zero contact velocity.
                com_diagnostic['contact_velocity'] = _to_numpy(
                    (contact_jacobian @ qd[..., None]).squeeze(-1)).copy()
                com_diagnostic['contact_active_rows'] = _to_numpy(output.active_row_mask).copy()
                com_diagnostic['contact_constraint_bias'] = _to_numpy(jdot_qdot).copy()
                com_diagnostic['ground_contact_forces'] = _to_numpy(ground_contact_forces).copy()
                # Native reports are needed to diagnose a failure, not to
                # retain large JSON strings for every 2 ms of a whole clip.
                com_diagnostic['foot_contact_report'] = np.asarray(
                    json.dumps(foot_contact_report) if not qp_validated else '')
                for name, value in zip(("contact_normal_force", "contact_point", "contact_normal",
                                        "contact_distance", "contact_count", "contact_start"), contact_data):
                    com_diagnostic[name] = _to_numpy(value).copy()
            if args.diagnose_com:
                com_diagnostic.update(com_acceleration_snapshot(fabric, q, qd, {
                    "fabric": qdd_fabric, "projected": qdd_dc,
                    "qp_reference": qp_reference,
                    "qp_nominal": qp_output.nominal_acc, "qp_predicted": qp_output.predicted_acc,
                }))
                complete_observed_interval(tracking_history, control_updates * FABRIC_TIMESTEP,
                                           com_diagnostic["com_velocity"])
                com_diagnostic.update(com_acc_observed=np.full_like(com_diagnostic["com_velocity"], np.nan),
                                      com_observed_dt=np.nan, com_observed_valid=False)
            if fabric.dynamics_tasks is not None:
                snapshot = fabric.dynamics_tasks.last_snapshot
                # Rolling support tasks appear/disappear as the observed sole
                # patch changes. Keep both fields in every history row; NaN
                # means inactive, not a fictitious zero tracking residual.
                for foot in range(2):
                    name = f'support_orientation_{foot}'
                    com_diagnostic['task_residual_' + name] = np.full((1,3), np.nan, dtype=np.float32)
                    com_diagnostic['task_active_' + name] = name in qp_task_residuals
                com_diagnostic.update({"task_residual_" + name: residual.detach().cpu().numpy().copy()
                                       for name, residual in qp_task_residuals.items()})
                if args.convex_wrist_collision:
                    com_diagnostic['collision_geometry_convex']=fabric.dynamics_tasks.convex_wrist_geometry is not None
                    for collision_key in ('task_residual_collision','collision_distance'):
                        if collision_key=='collision_distance' and 'collision_distance' in snapshot:
                            com_diagnostic[collision_key]=snapshot['collision_distance'].cpu().numpy().copy()
                        if collision_key not in com_diagnostic:continue
                        array=com_diagnostic[collision_key]
                        com_diagnostic[collision_key+'_count']=array.shape[-1]
                        width=len(fabric.dynamics_tasks.pairs)
                        if array.shape[-1]>width:raise ValueError('Unexpected collision diagnostic width')
                        com_diagnostic[collision_key]=np.pad(array,[(0,0),(0,width-array.shape[-1])],constant_values=np.nan)
                com_diagnostic["centroidal_momentum"] = snapshot["momentum"].cpu().numpy().copy()
                if "collision_distance" in snapshot and not args.convex_wrist_collision:
                    com_diagnostic["collision_distance"] = snapshot["collision_distance"].cpu().numpy().copy()
            if any(control_updates == round(t / FABRIC_TIMESTEP) for t in args.diagnose_qp_time):
                # Only serialize: no solver/controller input is changed.
                _, diagnostic_com_jac = fabric.center_of_mass_taskmap._position_and_jacobian(q)
                snapshot_path = args.output_path.with_suffix(f".qp_{control_updates:04d}.npz")
                snapshot_path.parent.mkdir(parents=True, exist_ok=True)
                tensors = dict(mass=mass, contact_jacobian=contact_jacobian,
                               contact_mode=output.contact_mode, bias_force=bias_force,
                               com_jacobian=diagnostic_com_jac, servo_bias=servo_bias,
                               servo_acc_gain=servo_acc_gain, correction=command_acc_correction,
                               q=q, qd=qd, nominal_acc=qp_output.nominal_acc,
                               reference_q=audit_reference_q,
                               predicted_acc=qp_output.predicted_acc, nominal_torque=qp_output.joint_torque,
                               physical_acc_lower=physical_lower, physical_acc_upper=physical_upper,
                               total_torque=qp_output.total_joint_torque,
                               fabric_to_isaac_joint=fabric_to_isaac_joint)
                np.savez_compressed(snapshot_path,
                    time=np.asarray(control_updates * FABRIC_TIMESTEP),
                    motion_frame_position=np.asarray(frame_position),
                    source_frame_count=np.asarray(getattr(args, 'source_frame_count', 0)),
                    source_start_frame=np.asarray(motion_start_frame),
                    **{"history_" + key: np.asarray([s[key] for s in tracking_history])
                       for key in (tracking_history[0] if tracking_history else {})},
                    **{"servo_" + key: np.asarray([s[key] for s in servo_history])
                       for key in (servo_history[0] if servo_history else {})},
                    controller_options=np.asarray(json.dumps(vars(args), default=str, sort_keys=True)),
                    **{k: v.detach().cpu().numpy() for k, v in tensors.items()},
                    **{"diagnostic_" + k: v for k, v in com_diagnostic.items()},
                    **{"problem_" + k: v.detach().cpu().numpy()
                       for k, v in vars(acceleration_qp.last_problem).items() if isinstance(v, torch.Tensor)},
                    **{"priority_problem_" + k: v.detach().cpu().numpy()
                       for k, v in (vars(acceleration_qp.last_priority_problem).items()
                                    if acceleration_qp.last_priority_problem is not None else ())
                       if isinstance(v, torch.Tensor)},
                    **{f"priority_level_{level}_" + k: v.detach().cpu().numpy()
                       for level, problem in enumerate(acceleration_qp.last_priority_problems)
                       for k, v in vars(problem).items() if isinstance(v, torch.Tensor)},
                    **{"task_" + request.name + "_" + field: getattr(request,field).detach().cpu().numpy()
                       for request in (fabric.dynamics_tasks.last_requests
                                       if fabric.dynamics_tasks is not None and not acceleration_only else ())
                       for field in ("A", "bias", "desired", "metric", "active")},
                    problem_solution=_to_numpy(acceleration_qp.last_full_solution))
                print(f"QP diagnostic snapshot: {snapshot_path}")
            if not qp_validated:
                failure_path = args.output_path.with_suffix(".failure.npz")
                failure_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    failure_path, q=q.detach().cpu().numpy(), qd=qd.detach().cpu().numpy(),
                    source_frame_count=np.asarray(getattr(args,'source_frame_count',0)),
                    source_start_frame=np.asarray(motion_start_frame),
                    controller_options=np.asarray(json.dumps(vars(args), default=str, sort_keys=True)),
                    control_time=np.asarray(control_updates * FABRIC_TIMESTEP),
                    joint_position_limits=joint_position_limits.cpu().numpy(),
                    joint_names=np.asarray(G1_JOINT_NAMES),
                    motion_ground_shift=np.asarray(motion_ground_shift),
                    initial_reference_q=(motion_trajectory[motion_start_frame:motion_start_frame + 1].cpu().numpy() if motion_trajectory is not None else np.empty((0,36))),
                    fabric_robot_name=np.asarray(fabric.g1_robot_name),
                    left_shoulder_roll_offset=np.asarray(args.left_shoulder_roll_offset),
                    acc_lower=command_acc_lower.cpu().numpy(), acc_upper=command_acc_upper.cpu().numpy(),
                    physical_acc_lower=physical_lower.cpu().numpy(),
                    physical_acc_upper=physical_upper.cpu().numpy(),
                    qp_acc=qp_output.acc.detach().cpu().numpy(), qp_validated=qp_output.validated.cpu().numpy(),
                    qp_predicted_acc=qp_output.predicted_acc.cpu().numpy(),
                    qp_nominal_acc=qp_output.nominal_acc.cpu().numpy(),
                    **{"diagnostic_" + key: value for key, value in com_diagnostic.items()},
                    **{"problem_" + key: value.detach().cpu().numpy()
                       for key, value in vars(acceleration_qp.last_problem).items()
                       if isinstance(value, torch.Tensor)},
                    **({"soft_problem_" + key: _to_numpy(value)
                        for key, value in vars(acceleration_qp.solver.last_augmented_problem).items()
                        if isinstance(value, torch.Tensor)}
                       if getattr(acceleration_qp.solver, 'last_augmented_problem', None) is not None else {}),
                    **({"soft_problem_solution": _to_numpy(acceleration_qp.solver.last_augmented_solution)}
                       if getattr(acceleration_qp.solver, 'last_augmented_solution', None) is not None else {}),
                    **{"priority_problem_" + key: value.detach().cpu().numpy()
                       for key, value in (vars(acceleration_qp.last_priority_problem).items()
                                          if acceleration_qp.last_priority_problem is not None else ())
                       if isinstance(value, torch.Tensor)},
                    **{f"priority_level_{level}_" + key: value.detach().cpu().numpy()
                       for level, problem in enumerate(acceleration_qp.last_priority_problems)
                       for key, value in vars(problem).items() if isinstance(value, torch.Tensor)},
                    **({"task_acceleration_C": _to_numpy(task_acceleration_constraints[0]),
                        "task_acceleration_d": _to_numpy(task_acceleration_constraints[1])}
                       if task_acceleration_constraints is not None else {}),
                    problem_solution=_to_numpy(acceleration_qp.last_full_solution),
                    self_contact_events=np.asarray(self_contact_events, dtype=str).reshape(-1, 4),
                    require_support_load=np.asarray(args.require_support_load),
                    phase_hold_duration=np.asarray(phase_hold_updates * FABRIC_TIMESTEP),
                    motion_frame_position=np.asarray(frame_position if motion_trajectory is not None else np.nan),
                    command_acc_correction=command_acc_correction.detach().cpu().numpy(),
                    qp_tracks_predicted_base=np.asarray(args.qp_track_predicted_base and not args.fixed_joint_test),
                    qp_total_torque=qp_output.total_joint_torque.cpu().numpy(),
                    servo_joint_order=np.asarray(isaac_joint_names),
                    **{"servo_" + key: np.asarray([s[key] for s in servo_history])
                       for key in (servo_history[0] if servo_history else {})},
                    **{"history_" + key: np.asarray([s[key] for s in tracking_history])
                       for key in (tracking_history[0] if tracking_history else {})},
                )
                print(f"[fail] QP validation; saved: {failure_path}", flush=True)
                if measured_steps is not None:
                    failed_steps = measured_steps.report()
                    failed_steps.update(controller_valid=False, steps_tracking_passed=False,
                        stable_final_support=False, stopped_at_tracking_steps=False,
                        stop_reason='QP validation failed', failed_time_s=float(control_time))
                    if len(tracking_history) >= 2:
                        from analyze_bounded_steps import summarize as summarize_steps
                        with np.load(failure_path) as recording:
                            failed_steps.update(summarize_steps(recording, json.loads(
                                args.output_path.with_suffix('.steps_plan.json').read_text())))
                        failed_steps['controller_valid'] = False
                    args.output_path.with_suffix('.steps.json').write_text(
                        json.dumps(failed_steps, indent=2)+'\n')
                    print(f"[fail] verified source landings={failed_steps['measured_steps']}/"
                          f"{failed_steps['expected_steps']}; "
                          f"PKL fidelity={failed_steps.get('pkl_fidelity_passed', False)}; "
                          f"report: {args.output_path.with_suffix('.steps.json')}", flush=True)
                if args.track_full_motion and len(tracking_history) >= 2:
                    from analyze_motion_tracking import summarize_full_motion
                    with np.load(failure_path) as recording:
                        failed_tracking = summarize_full_motion(recording, json.loads(
                            args.output_path.with_suffix('.steps_plan.json').read_text()))
                    failed_tracking.update(stop_reason='QP validation failed',
                        failed_time_s=float(control_time), failed_source_frame=float(frame_position))
                    if fidelity_stop is not None:
                        failed_tracking['first_fidelity_exceeded']=fidelity_stop
                        failed_tracking['diagnostic_continuation_after_fidelity_loss']=not args.stop_at_fidelity_loss
                    args.output_path.with_suffix('.tracking.json').write_text(
                        json.dumps(failed_tracking, indent=2)+'\n')
                joint_bound_error = torch.maximum(
                    physical_lower[:, 6:] - qp_output.predicted_acc[:, 6:],
                    qp_output.predicted_acc[:, 6:] - physical_upper[:, 6:]).clamp_min(0.)
                worst_joint = int(joint_bound_error[0].argmax())
                if float(joint_bound_error[0, worst_joint]) > args.qp_validation_tolerance:
                    print("[fail] measured joint acceleration bound: " + json.dumps({
                        "joint": G1_JOINT_NAMES[worst_joint],
                        "position": float(q[0, 7 + worst_joint]),
                        "velocity": float(qd[0, 6 + worst_joint]),
                        "predicted_acceleration": float(qp_output.predicted_acc[0, 6 + worst_joint]),
                        "minimum_acceleration": float(physical_lower[0, 6 + worst_joint]),
                        "maximum_acceleration": float(physical_upper[0, 6 + worst_joint]),
                        "violation": float(joint_bound_error[0, worst_joint]),
                    }), flush=True)
                import omni.kit.app
                omni.kit.app.get_app().post_quit(1)
                raise RuntimeError(
                    f"{args.control_frequency} Hz acceleration QP failed validation: "
                    f"control_time={control_updates * FABRIC_TIMESTEP:.3f}s, "
                    f"mode={output.contact_mode.tolist()}, "
                    f"normal_forces={foot_normal_forces.tolist()}, "
                    f"base_position={q[:, :3].tolist()}, "
                    f"foot_height={live_feet[..., 2].tolist()}, "
                    f"pose_error_max={pose_error.abs().amax().item():.6f}, "
                    f"velocity_norm={torch.linalg.vector_norm(qd).item():.6f}, "
                    f"projected_acceleration_norm="
                    f"{torch.linalg.vector_norm(qdd_dc).item():.6f}, "
                    f"converged={qp_output.converged.tolist()}, "
                    f"priority_errors={None if acceleration_qp.last_priority_errors is None else [v.tolist() for v in acceleration_qp.last_priority_errors]}, "
                    f"base_error={qp_output.base_dynamics_error.tolist()}, "
                    "joint_dynamics_error="
                    f"{qp_output.joint_dynamics_error.tolist()}, "
                    "contact_error="
                    f"{qp_output.contact_acceleration_error.tolist()}, "
                    "friction_violation="
                    f"{qp_output.friction_violation.tolist()}, "
                    "effort_violation="
                    f"{qp_output.effort_violation.tolist()}, "
                    f"bound_violation={qp_output.bound_violation.tolist()}, "
                    f"collision_acceleration_error={acceleration_qp.last_acceleration_constraint_error.tolist()}."
                )
            if (
                not bool(qp_output.converged.all())
                and not reported_nonconvergence
            ):
                print(
                    f"warning: {args.qp_solver.upper()} reached a physically validated solution "
                    "without satisfying its stricter numerical convergence "
                    f"flag after {qp_output.iters} iterations."
                )
                reported_nonconvergence = True
            qdd_dynamics_feasible = qp_output.acc
            joint_torque_feedforward = qp_output.joint_torque

            # Same command path with or without a PKL, including warmup.
            # Fabric/QP use live q, qd; desired states retain integrator
            # history instead of being overwritten by the motion reference.
            joint_velocity_desired = (
                joint_velocity_desired
                + FABRIC_TIMESTEP * qdd_dynamics_feasible[:, 6:35]
            )
            joint_position_desired = (
                joint_position_desired
                + FABRIC_TIMESTEP * joint_velocity_desired
            )
            if args.fixed_joint_test:
                joint_position_desired = integration_initial_position.clone()
                joint_velocity_desired = torch.zeros_like(integration_initial_velocity)
                joint_torque_feedforward = initial_joint_feedforward.clone()
            position_desired_isaac = joint_position_desired[
                :, fabric_to_isaac_joint
            ]
            velocity_desired_isaac = joint_velocity_desired[
                :, fabric_to_isaac_joint
            ]
            torque_feedforward_isaac = joint_torque_feedforward[
                :, fabric_to_isaac_joint
            ]
            joint_torque_pd = (
                torch.zeros_like(joint_velocity) if args.tracking_servo else -kd*joint_velocity if args.direct_wbc else
                kp * (joint_position_desired - joint_position)
                + kd * (joint_velocity_desired - joint_velocity)
            )
            joint_torque_unclipped = (
                joint_torque_feedforward + joint_torque_pd
            )
            joint_torque = torch.clamp(
                joint_torque_unclipped,
                min=-effort_limits,
                max=effort_limits,
            )
            if args.mocap_unload_test:
                world_wrench = qp_output.contact_wrench.reshape(1, 2, 6).clone()
                for foot, patch in contact_patches.items():
                    world_wrench[:, foot] = patch.wrench_to_world_ankle(
                        world_wrench[:, foot], _quat_xyzw_to_matrix(live_feet[:, foot, 3:]))
                com_diagnostic.update(
                    unload_support=-1 if transfer_reference is None else transfer_reference.support,
                    unload_elapsed=-1. if transfer_reference is None else control_time-transfer_reference.start_time,
                    contact_wrench_world_ankle=_to_numpy(world_wrench).copy(),
                    unload_com_velocity=_to_numpy(actual_com_velocity).copy(),
                    unload_effort_ratio=_to_numpy((qp_output.total_joint_torque.abs()/effort_limits).amax(-1)).copy(),
                    unload_qp_validated=_to_numpy(qp_output.validated).copy(),
                    unload_base_error=_to_numpy(qp_output.base_dynamics_error).copy(),
                    unload_contact_error=_to_numpy(qp_output.contact_acceleration_error).copy(),
                    unload_friction_error=_to_numpy(qp_output.friction_violation).copy(),
                    unload_bound_error=_to_numpy(qp_output.bound_violation).copy())
            if motion_trajectory is not None:
                source_feet_reference = _to_numpy(reference_feet_transforms(fabric, audit_reference_q)[0,:,:3]).copy()
                tracking_history.append({
                    **com_diagnostic,
                    "time": control_time,
                    "reference_mode": args.reference_mode,
                    "source_step_number": source_step_number,
                    "source_tail": getattr(source_step_rollout,"is_tail",False),
                    "source_recovery": source_step_rollout is not None and source_step_rollout.fixed_source_frame is not None,
                    "source_recovery_target_feet": (np.full((2,7),np.nan) if source_step_rollout is None
                        or source_step_rollout.recovery_target_feet is None else _to_numpy(source_step_rollout.recovery_target_feet)),
                    "source_recovery_root": (np.full(3,np.nan) if source_step_rollout is None
                        or source_step_rollout.recovery_source_root is None else _to_numpy(source_step_rollout.recovery_source_root)),
                    "source_step_end_frame": -1 if source_step_rollout is None else source_step_rollout.end_frame,
                    "source_steps_finished": source_steps_finished,
                    "source_step_elapsed": -1. if source_step_start is None else control_time-source_step_start,
                    "source_step_goal": (np.full(7, np.nan) if source_step_rollout is None
                                         else source_step_rollout.goal[0].cpu().numpy().copy()),
                    "source_step_source_goal": (np.full(7, np.nan) if source_step_rollout is None
                                                else source_step_rollout.source_goal[0].cpu().numpy().copy()),
                    "source_step_touchdown": source_step_rollout is not None and source_step_rollout.touchdown,
                    "support_switch_phase": "INACTIVE" if support_switch is None else support_switch.transfer.phase,
                    "support_switch_release_time": -1. if support_switch is None or support_switch.transfer.release_time is None else support_switch.transfer.release_time,
                    "support_switch_target": -1 if support_switch is None else support_switch.support,
                    "transfer_phase": "INACTIVE" if transfer_reference is None else transfer_reference.phase,
                    **({"whole_body_reference_q": whole_body_reference.q.cpu().numpy().copy(),
                        "whole_body_reference_v": whole_body_reference.v.cpu().numpy().copy(),
                        "whole_body_reference_a": whole_body_reference.a.cpu().numpy().copy()}
                       if unified_reference else {}),
                    "reference_phase": ("SOURCE_CLOCK" if args.motion_start_in_motion else "legacy") if managed_clock is None else managed_clock.phase,
                    "com_reference_policy": "unified_qva",
                    "reference_rate": args.motion_speed*clock_rate if managed_clock is None else managed_clock.rate,
                    "managed_feet_position": fabric.feet_pose_taskmap.position_target.detach().cpu().numpy().copy(),
                    "managed_joint_posture": fabric.joint_posture_taskmap.position_target.detach().cpu().numpy().copy(),
                    "reference_wait": 0. if managed_clock is None else managed_clock.waiting_seconds,
                    "preview_release_ready": preview_release_ready,
                    "preview_release_reason": preview_release_reason,
                    "landing_goal": (np.full(7,np.nan) if landing_transition is None
                                     else landing_transition.goal.copy()),
                    "landing_foot": -1 if landing_key is None else landing_key[1],
                    "pkl_joint_rmse": torch.mean((q[:,7:]-audit_reference_q[:,7:]).square(),dim=1).sqrt().detach().cpu().numpy(),
                    "pkl_root_position_error": torch.linalg.vector_norm(q[:,:3]-audit_reference_q[:,:3],dim=1).detach().cpu().numpy(),
                    "frame": frame_position,
                    "phase_held": phase_held if motion_trajectory is not None else False,
                    "single_support_ready_count": single_ready_count,
                    "single_support_released": single_release_time is not None,
                    "contact_cycle_phase": "disabled" if contact_cycle is None else contact_cycle.phase,
                    "q": q.detach().cpu().numpy().copy(),
                    "qd": qd.detach().cpu().numpy().copy(),
                    "reference_q": audit_reference_q.detach().cpu().numpy().copy(),
                    "source_feet_reference": source_feet_reference,
                    "control_reference_q": reference_q.detach().cpu().numpy().copy(),
                    "com": fabric.compute_center_of_mass(q).detach().cpu().numpy().copy(),
                    "com_target": com_balance_target.detach().cpu().numpy().copy(),
                    "com_target_velocity": com_target_velocity.detach().cpu().numpy().copy(),
                    "com_target_acceleration": com_target_acceleration.detach().cpu().numpy().copy(),
                    "planned_mode": planned_contact_mode.detach().cpu().numpy().copy(),
                    "commanded_mode": commanded_contact_mode.detach().cpu().numpy().copy(),
                    "controller_contact_plan": controller_contact_plan.detach().cpu().numpy().copy(),
                    "measured_mode": measured_mode.detach().cpu().numpy().copy(),
                    "foot_forces": ground_foot_forces.detach().cpu().numpy().copy(),
                    "ground_foot_forces": ground_foot_forces.detach().cpu().numpy().copy(),
                    "net_foot_forces": net_foot_forces.detach().cpu().numpy().copy(),
                    "support_force_source": "ground_filtered",
                    "feet": live_feet.detach().cpu().numpy().copy(),
                    "qdd_qp": qp_output.acc.detach().cpu().numpy().copy(),
                    "qdd_fabric": qdd_fabric.detach().cpu().numpy().copy(),
                    "qdd_projected": qdd_dc.detach().cpu().numpy().copy(),
                    "qdd_qp_reference": qp_reference.detach().cpu().numpy().copy(),
                    "qdd_predicted": qp_output.predicted_acc.detach().cpu().numpy().copy(),
                    "contact_wrench": qp_output.contact_wrench.detach().cpu().numpy().copy(),
                    "wrench_reference": (torch.zeros_like(qp_output.contact_wrench) if wrench_reference is None else wrench_reference).detach().cpu().numpy().copy(),
                    "wrench_objective_weights": (torch.full_like(qp_output.contact_wrench, acceleration_qp.wrench_weight)
                        if wrench_weights is None else acceleration_qp.wrench_weight*wrench_weights).detach().cpu().numpy().copy(),
                    "wrench_target_active": wrench_reference is not None,
                    "pelvis_target": fabric.pelvis_pose_taskmap.position_target.detach().cpu().numpy().copy(),
                    "qdd_nominal": qp_output.nominal_acc.detach().cpu().numpy().copy(),
                    "command_acc_correction": command_acc_correction.detach().cpu().numpy().copy(),
                    "joint_position_desired": joint_position_desired.detach().cpu().numpy().copy(),
                    "joint_velocity_desired": joint_velocity_desired.detach().cpu().numpy().copy(),
                    "integration_seed_position": integration_seed_position.detach().cpu().numpy().copy(),
                    "integration_seed_velocity": integration_seed_velocity.detach().cpu().numpy().copy(),
                    "torque_feedforward": joint_torque_feedforward.detach().cpu().numpy().copy(),
                    "torque_qp_total": qp_output.total_joint_torque.detach().cpu().numpy().copy(),
                    "torque_pd_estimate": joint_torque_pd.detach().cpu().numpy().copy(),
                    "unload_intended_mode": unload_intended_mode,
                    "unload_ramp": unload_ramp if dynamic_unloading else 0.,
                    "unload_intended_reason": unload_intended_reason,
                    "effective_qp_effort_limits": _to_numpy(current_effort_limits).copy(),
                    "joint_tube_recovery_used": acceleration_qp.tube_recovery_used,
                    "fabric_priority_mode": args.fabric_priority_mode,
                    "fabric_policy_recomposed": recompose_tasks,
                    "fabric_tracking_feedback": args.fabric_tracking_feedback,
                    "fabric_secondary_scale": args.fabric_secondary_scale,
                    "fabric_root_regularization": fabric.dynamics_tasks.config['regularization'] if fabric.dynamics_tasks is not None else float('nan'),
                    "qp_task_only_objective": args.qp_task_only_objective,
                    "qp_single_proximal": getattr(acceleration_qp.solver, 'single_proximal', 0.),
                    "qp_cold_start": bool(acceleration_only and control_time >= args.qp_cold_start_time),
                    "qp_equilibration_cap": getattr(acceleration_qp.solver, 'equilibration_cap', 1.),
                    "qp_iteration_budget": getattr(acceleration_qp.solver, 'max_iter', 0),
                    "fabric_priority_contact_error": _to_numpy((getattr(fabric.dynamics_tasks, 'last_snapshot', None) or {}).get(
                        'fabric_priority_contact_error', q.new_zeros(q.shape[0]))).copy(),
                    "fabric_priority_nullspace_error": _to_numpy((getattr(fabric.dynamics_tasks, 'last_snapshot', None) or {}).get(
                        'fabric_priority_nullspace_error', q.new_zeros(q.shape[0]))).copy(),
                    "fabric_priority_idempotence_error": _to_numpy((getattr(fabric.dynamics_tasks, 'last_snapshot', None) or {}).get(
                        'fabric_priority_idempotence_error', q.new_zeros(q.shape[0]))).copy(),
                    "tracking_tube_soft_slack": (_to_numpy(acceleration_qp.solver.last_slack).copy()
                        if getattr(acceleration_qp.solver, 'last_slack', None) is not None else np.zeros((1,3))),
                    "orientation_tube_recovery_bandwidth": acceleration_qp.orientation_tube_recovery_bandwidth,
                    "tracking_tube_diagnostics": (_to_numpy(tube_diagnostics).copy() if (elapsed_motion_time>=args.tracking_tube_start and elapsed_motion_time<args.tracking_tube_end) else np.full((1,3,3),np.nan)),
                    "tracking_tube_bound": (_to_numpy(tube_d).copy() if (elapsed_motion_time>=args.tracking_tube_start and elapsed_motion_time<args.tracking_tube_end) else np.full((1,3),np.nan)),
                    "tracking_tube_residual": (_to_numpy((tube_C@qp_output.predicted_acc[...,None]).squeeze(-1)-tube_d).copy() if (elapsed_motion_time>=args.tracking_tube_start and elapsed_motion_time<args.tracking_tube_end) else np.full((1,3),np.nan)),
                })
            if fidelity_stop is None and motion_trajectory is not None:
                from analyze_motion_tracking import summarize_tracking
                sample_audit=summarize_tracking(dict(history_frame=np.asarray([frame_position]),history_q=_to_numpy(q),history_reference_q=_to_numpy(audit_reference_q),source_frame_count=len(motion_trajectory),source_start_frame=0))
                if not sample_audit['pkl_fidelity_passed']:
                    fidelity_stop=dict(time_s=float(control_time),source_frame=float(frame_position),joint_rmse_rad=sample_audit['peak_joint_rmse'],root_position_m=sample_audit['peak_root_position_error'],root_orientation_rad=sample_audit['peak_root_orientation_error'])
                    print('Original-source fidelity limit exceeded: '+json.dumps(fidelity_stop),flush=True)
                    if args.stop_at_fidelity_loss:
                        break
            if (args.mocap_first_step_report is not None and transfer_reference is not None
                    and source_step_rollout is None and control_time >= transfer_reference.start_time+3.5-FABRIC_TIMESTEP):
                try:
                    from analyze_contact_transfer import summarize_unloading
                    prefix = {'history_'+key: np.asarray([row[key] for row in tracking_history])
                              for key in tracking_history[0]}
                    # Pre-release controller gate uses recorded physical QP and speed checks;
                    # final full-controller and landing validation still runs at shutdown.
                    prefix['motion_tracking_valid'] = bool(
                        np.all(prefix['history_unload_qp_validated'])
                        and np.max(prefix['history_joint_velocity_violation'])<=1e-4
                        and np.isfinite(prefix['history_q']).all())
                    probe_unload_report = summarize_unloading(prefix)
                    probe_unload_report['controller_gate_scope'] = 'pre-release recorded QP, speed and finite-state checks'
                    args.output_path.with_suffix('.probe_unloading.json').write_text(json.dumps(probe_unload_report,indent=2)+"\n")
                    if not probe_unload_report['unloading_passed']:
                        raise RuntimeError('First-step probe requires verified measured unloading')
                    from fabrics_sim.reference.optimized_source_rollout import OptimizedSourceRollout
                    from g1_armature import unitree_armatures
                    source_step_rollout = OptimizedSourceRollout(fabric,
                        WholeBodyReference(q.clone(),qd.clone(),qp_output.predicted_acc.clone()),
                        G1_JOINT_NAMES,control_motion_trajectory,FABRIC_TIMESTEP,
                        q.new_tensor(unitree_armatures(G1_JOINT_NAMES)),current_effort_limits,
                        args.mocap_first_step_report,args.output_path.with_suffix('.probe_plan.json'),
                        project_nominal_torque=args.mocap_first_step_project_torque)
                    source_step_start = control_time
                    source_step_number = 1
                    transfer_reference.release_time = control_time
                    reference_manager.phase = 'SWING'
                    reference_manager.pending_takeoff = None
                    fabric.dynamics_tasks.preserve_reference_acceleration = True
                    print(f'Live-boundary first-step probe started: time={control_time:.3f}s, target_frame=129',flush=True)
                except (ValueError,RuntimeError) as exc:
                    save_reference_failure(args,exc,control_time,q,qd,foot_normal_forces,tracking_history,servo_history)
                    raise
            if args.motion_start_in_motion and measured_steps is not None:
                from analyze_contact_transfer import sole_clearances
                feet_observed = _to_numpy(live_feet[0])
                measured_steps.update(frame_position, control_time, int(measured_mode.item()),
                    _to_numpy(foot_normal_forces[0]), sole_clearances(feet_observed,
                        np.stack([_to_numpy(g[0]) for g in foot_landing_geometry]),
                        np.stack([_to_numpy(g[1]) for g in foot_landing_geometry])),
                    feet_observed[:,:3], FABRIC_TIMESTEP, reference_feet=source_feet_reference)
                if measured_steps.complete and not args.track_full_motion:
                    stopped_at_tracking_steps = True
                    print(f'Measured {args.tracking_steps} source landings at {control_time:.3f}s; '
                          'continuous-walking endpoint, not a stationary settle', flush=True)
                    break
            if args.track_full_motion and frame_position >= len(motion_trajectory)-1:
                stopped_at_motion_end = True
                print(f'Full source endpoint reached: frame={frame_position:g}, time={control_time:.3f}s; '
                      'continuous endpoint, no stationary settling claim', flush=True)
                break
            control_updates += 1
            if args.contact_aware_servo:
                if args.nominal_implicit_servo:
                    servo_kp_bank = nominal_kp[None].expand(4,-1)
                    servo_kd_bank = nominal_kd[None].expand(4,-1)
                else:
                    servo_kp_bank, servo_kd_bank = contact_servo_gain_bank(
                        mass, contact_jacobian, contact_row_mask, nominal_kp, nominal_kd,
                        PD_TIMESTEP, args.servo_bandwidth)
                servo_response_bank = implicit_servo_response(mass,servo_kp_bank,servo_kd_bank,PD_TIMESTEP)
                servo_response_bank = servo_response_bank[:,fabric_to_isaac_joint][:,:,fabric_to_isaac_joint]
                servo_kp_bank = servo_kp_bank[:,fabric_to_isaac_joint]
                servo_kd_bank = servo_kd_bank[:,fabric_to_isaac_joint]

        actual_position_isaac = torch.as_tensor(
            _to_numpy(robot.get_joint_positions()),
            device=device,
            dtype=q.dtype,
        )
        actual_velocity_isaac = torch.as_tensor(
            _to_numpy(robot.get_joint_velocities()),
            device=device,
            dtype=q.dtype,
        )
        if args.tracking_servo:
            if args.contact_aware_servo:
                fast_forces = ground_support_forces(_to_numpy(
                    foot_contact_view.get_contact_force_matrix(dt=contact_sample_dt)))[:,2]
                loaded_mode = int(fast_forces[0] >= 20.) + 2*int(fast_forces[1] >= 20.)
                servo_contact_mode = loaded_mode & int(output.contact_mode.item())
                kp_isaac, kd_isaac = servo_kp_bank[servo_contact_mode], servo_kd_bank[servo_contact_mode]
            elapsed=(step % fabric_steps_per_update)*PD_TIMESTEP
            seed_position=integration_seed_position[:,fabric_to_isaac_joint]
            seed_velocity=integration_seed_velocity[:,fabric_to_isaac_joint]
            seed_acceleration=qp_output.predicted_acc[:,6:][:,fabric_to_isaac_joint]
            servo_position=seed_position+seed_velocity*elapsed+.5*seed_acceleration*elapsed**2
            servo_velocity=seed_velocity+seed_acceleration*elapsed
            joint_torque_pd_isaac=(kp_isaac*(servo_position-actual_position_isaac)
                                   +kd_isaac*(servo_velocity-actual_velocity_isaac))
            if args.contact_aware_servo:
                rhs = joint_torque_pd_isaac + PD_TIMESTEP*kp_isaac*(servo_velocity-actual_velocity_isaac)
                joint_torque_pd_isaac = (servo_response_bank[servo_contact_mode] @ rhs[...,None]).squeeze(-1)
        else:
            joint_torque_pd_isaac = (
                -kd_isaac*actual_velocity_isaac if args.direct_wbc else
                kp_isaac * (position_desired_isaac - actual_position_isaac)
                + kd_isaac * (velocity_desired_isaac - actual_velocity_isaac)
            )
        joint_torque_isaac_tensor = (
            torque_feedforward_isaac + joint_torque_pd_isaac
        )
        joint_torque_isaac_tensor = torch.clamp(
            joint_torque_isaac_tensor,
            min=-effort_limits_isaac,
            max=effort_limits_isaac,
        )
        # Send the limited total effort, including explicit feedback.
        robot.set_joint_efforts(
            joint_torque_isaac_tensor.detach().cpu().numpy()
        )
        physics_substeps = (args.late_physics_substeps
                            if step*PD_TIMESTEP >= args.late_physics_substeps_start else 1)
        servo_history.append({
            "time": step * PD_TIMESTEP,
            "physics_time": world.current_time - initial_physics_time,
            "qp_update_index": control_updates - 1,
            "physics_substeps": physics_substeps,
            "native_physics_dt": PD_TIMESTEP / physics_substeps,
            "kp": kp_isaac.detach().cpu().numpy().copy(),
            "kd": kd_isaac.detach().cpu().numpy().copy(),
            "velocity": actual_velocity_isaac.detach().cpu().numpy().copy(),
            "torque": joint_torque_isaac_tensor.detach().cpu().numpy().copy(),
            "unclipped": (torque_feedforward_isaac + joint_torque_pd_isaac).detach().cpu().numpy().copy(),
        })
        # Torque/control clocks remain unchanged when the native integrator is
        # subdivided. Contact impulses on the next tick use the last native dt.
        held_effort = joint_torque_isaac_tensor.detach().cpu().numpy()
        contact_sample_dt = step_physics_interval(
            world, PD_TIMESTEP, physics_substeps,
            apply_effort=(lambda: robot.set_joint_efforts(held_effort)) if physics_substeps > 1 else None)
        if args.show_isaac_window and (step + 1) % render_steps_per_update == 0:
            world.render()
        if step % fabric_steps_per_update == 0 and any(
                step // fabric_steps_per_update == round(t / FABRIC_TIMESTEP) for t in args.diagnose_qp_time):
            after_q, after_v, _, _, after_left_j, after_right_j = read_live_isaac_snapshot(
                robot, isaac_joint_names, device, q.dtype)
            after_com, after_com_j = fabric.center_of_mass_taskmap._position_and_jacobian(after_q)
            after_com_v = torch.bmm(after_com_j, after_v.unsqueeze(-1)).squeeze(-1)
            response_path = args.output_path.with_suffix(f".response_{step // fabric_steps_per_update:04d}.npz")
            np.savez_compressed(response_path, dt=np.asarray(PD_TIMESTEP),
                q=after_q.detach().cpu().numpy(), qd=after_v.detach().cpu().numpy(),
                com_velocity=after_com_v.detach().cpu().numpy(),
                contact_jacobian=torch.cat((after_left_j, after_right_j), dim=1).detach().cpu().numpy(),
                applied_torque=joint_torque_isaac_tensor[:, torch.argsort(fabric_to_isaac_joint)].detach().cpu().numpy())
            # Restore COM kinematic scratch buffers, not the physical state.
            fabric.center_of_mass_taskmap._position_and_jacobian(q)
        executed_steps = step + 1
    joint_torque_isaac = (
        joint_torque_isaac_tensor.detach().cpu().numpy()
    )
    joint_position_after_control_isaac = torch.as_tensor(
        _to_numpy(robot.get_joint_positions()),
        device=device,
        dtype=q.dtype,
    )
    joint_velocity_after_control = torch.as_tensor(
        _to_numpy(robot.get_joint_velocities()),
        device=device,
        dtype=q.dtype,
    )
    isaac_to_fabric_joint = torch.argsort(fabric_to_isaac_joint)
    joint_position_after_control = joint_position_after_control_isaac[
        :, isaac_to_fabric_joint
    ]
    joint_velocity_after_control = joint_velocity_after_control[
        :, isaac_to_fabric_joint
    ]
    final_root_position, final_root_orientation_wxyz = robot.get_world_poses()
    final_root_position = torch.as_tensor(
        _to_numpy(final_root_position), device=device, dtype=q.dtype
    )
    final_root_orientation_wxyz = torch.as_tensor(
        _to_numpy(final_root_orientation_wxyz), device=device, dtype=q.dtype
    )
    final_q = torch.cat(
        (
            final_root_position,
            final_root_orientation_wxyz[:, 1:4],
            final_root_orientation_wxyz[:, 0:1],
            joint_position_after_control,
        ),
        dim=1,
    )
    actual_center_of_mass = fabric.compute_center_of_mass(final_q).detach()
    center_of_mass_target = (
        fabric.center_of_mass_taskmap.position_target.detach().clone()
    )
    center_of_mass_error = actual_center_of_mass - center_of_mass_target
    center_of_mass_error_norm = torch.linalg.vector_norm(
        center_of_mass_error, dim=1
    )
    joint_position_rmse = torch.sqrt(
        torch.mean(
            (joint_position_after_control - joint_position_desired).square(),
            dim=1,
        )
    )
    minimum_allowed_com_height = (
        args.minimum_com_height_ratio * initial_center_of_mass[:, 2]
    )
    motion_tracking_valid = (
        (center_of_mass_error_norm <= args.maximum_com_tracking_error)
        & (actual_center_of_mass[:, 2] >= minimum_allowed_com_height)
        & (joint_position_rmse <= args.maximum_joint_position_rmse)
    )
    servo_valid, servo_oscillating, servo_saturation_fraction = servo_health(
        torch.from_numpy(np.asarray([s["velocity"] for s in servo_history])),
        torch.from_numpy(np.asarray([s["torque"] for s in servo_history])),
        torch.from_numpy(np.asarray([s["unclipped"] for s in servo_history])),
    )
    peak_com_error = center_of_mass_error_norm.max().item()
    peak_servo_rmse = joint_position_rmse.max().item()
    minimum_com_height = actual_center_of_mass[:, 2].min().item()
    for sample in tracking_history:
        peak_com_error = max(peak_com_error, float(np.linalg.norm(sample["com"] - sample["com_target"], axis=-1).max()))
        peak_servo_rmse = max(peak_servo_rmse, float(np.sqrt(np.mean((sample["q"][:, 7:] - sample["joint_position_desired"])**2, axis=-1)).max()))
        minimum_com_height = min(minimum_com_height, float(sample["com"][:, 2].min()))
    motion_tracking_valid &= (
        bool(servo_valid) and peak_com_error <= args.maximum_com_tracking_error
        and peak_servo_rmse <= args.maximum_joint_position_rmse
        and minimum_com_height >= minimum_allowed_com_height.max().item()
        and (executed_steps == physics_steps or stopped_at_motion_end or stopped_at_tracking_steps or stopped_at_source_step_end or stopped_at_unload_end)
        and not args.disable_self_collisions
    )
    peak_pkl_joint_rmse = max((float(np.max(row["pkl_joint_rmse"])) for row in tracking_history),default=float("nan"))
    peak_pkl_root_error = max((float(np.max(row["pkl_root_position_error"])) for row in tracking_history),default=float("nan"))
    if tracking_history:
        print(f"PKL fidelity (separate from servo validation): peak joint RMSE={peak_pkl_joint_rmse:.6f} rad, "
              f"peak root position error={peak_pkl_root_error:.6f} m, final source frame={frame_position:.2f}")
    reference_status = "disabled"
    if managed_clock is not None:
        reference_status = (
            "source_finished" if managed_clock.frame >= len(motion_trajectory)-1 else
            "waiting_for_contact" if managed_clock.phase in ("PREPARE_STEP", "LANDING", "SETTLE") and managed_clock.waiting_seconds > 0 else
            "initializing" if managed_clock.phase == "INITIALIZE" else "advancing"
        )
        if source_step_rollout is not None:
            reference_status = "diagnostic_" + managed_clock.phase.lower()
        elif args.unified_transfer_test and transfer_reference is not None:
            reference_status = "diagnostic_" + transfer_reference.phase.lower()
        print(f"ReferenceManager: status={reference_status}, phase={managed_clock.phase}, "
              f"source frame={managed_clock.frame:.2f}, rate={managed_clock.rate:.3f}, "
              f"waiting={managed_clock.waiting_seconds:.2f}s; controller validation alone does not certify gait success")
    if args.unified_transfer_test and not args.unified_source_step_test:
        recent = tracking_history[-int(round(1./FABRIC_TIMESTEP)):]
        transfer_valid = (transfer_reference is not None and transfer_reference.release_time is not None
                          and (executed_steps-1)*PD_TIMESTEP >= transfer_reference.release_time+3.
                          and len(recent) == int(round(1./FABRIC_TIMESTEP)))
        if transfer_valid:
            from analyze_contact_transfer import sole_clearances
            support = transfer_reference.support
            for row in recent:
                feet = row["feet"].reshape(2, 7)
                transfer_valid &= (np.all(row["commanded_mode"] == support+1)
                    and np.linalg.norm(row["com"]-row["com_target"]) <= .03
                    and sole_clearances(feet)[1-support] >= .02
                    and row["foot_forces"][1-support, 2] < 10.
                    and row["foot_forces"][support, 2] >= 100.)
        motion_tracking_valid &= bool(transfer_valid)
        print(f"Whole-body transfer validation: valid={bool(transfer_valid)}, release_time={None if transfer_reference is None else transfer_reference.release_time}; requires 3s after release and 1s stable single support")
    if args.unified_source_step_test or args.mocap_first_step_report is not None:
        last = tracking_history[-1]
        swing = 1-transfer_reference.support if transfer_reference is not None else 1
        landing_error = (float(np.linalg.norm(last["feet"].reshape(2,7)[swing,:3]-source_step_rollout.goal[0,:3].cpu().numpy()))
                         if source_step_rollout is not None else float('inf'))
        step_valid = bool(stopped_at_source_step_end and source_step_rollout is not None and source_step_rollout.touchdown and landing_error <= .04
                          and np.linalg.norm(last["com"]-last["com_target"]) <= .05)
        motion_tracking_valid &= step_valid
        print(f"First source-step validation: valid={step_valid}, landing_error={landing_error:.6f} m; no continuous-walking certification")
        if args.mocap_first_step_report is not None:
            args.output_path.with_suffix('.probe_result.json').write_text(json.dumps(dict(
                scope='bounded_first_source_step_probe', full_four_step_tracking_passed=False,
                nominal_torque_projection_requested=args.mocap_first_step_project_torque,
                step_valid=step_valid, landing_error_m=landing_error,
                measured_touchdown=bool(source_step_rollout is not None and source_step_rollout.touchdown),
                controller_valid=bool(motion_tracking_valid.all()),
                unloading=probe_unload_report),indent=2)+"\n")
    final_right_load_fraction = float("nan")
    if args.single_support_test and contact_cycle is None:
        recent = tracking_history[-50:]
        support_index = 0 if args.single_support_test == "left" else 1
        hold_valid = (single_release_time is not None and
                      (executed_steps - 1) * PD_TIMESTEP >= single_release_time + 3. and len(recent) == 50)
        for sample in recent:
            feet = sample["feet"].reshape(2, 7)
            hold_valid &= (bool(np.all(sample["commanded_mode"] == support_index + 1))
                           and np.linalg.norm(sample["com"] - sample["com_target"]) <= .03
                           and feet[1-support_index, 2] - single_initial_feet[0, 1-support_index, 2].item() >= .025
                           and sample["foot_forces"][1-support_index, 2] < 10.)
        # numpy comparisons above yield np.bool_; torch in-place boolean AND
        # must receive a Python bool rather than numpy scalar conversion.
        hold_valid = bool(hold_valid)
        motion_tracking_valid &= hold_valid
        print(f"single support validation: valid={hold_valid}, release_time={single_release_time}; requires 3s post-release and final 1s stable clearance/COM/contact")
    if contact_cycle is not None:
        cycle_valid = contact_cycle.phase == "done" and contact_cycle.ever_clear
        recent = tracking_history[-50:]
        cycle_valid &= len(recent) == 50 and all(
            np.all(s["commanded_mode"] == 3) and np.all(s["foot_forces"][:, 2] >= 20.)
            and np.linalg.norm(s["com"] - s["com_target"]) <= .03 for s in recent)
        motion_tracking_valid &= bool(cycle_valid)
        print(f"contact cycle validation: valid={bool(cycle_valid)}, phase={contact_cycle.phase}")
    if args.weight_transfer_test:
        recent_forces = np.asarray([sample["foot_forces"][:, 2]
                                    for sample in tracking_history[-10:]])
        mean_load = np.maximum(recent_forces, 0.).mean(axis=0)
        final_right_load_fraction = float(mean_load[1] / max(mean_load.sum(), 1e-8))
        transfer_valid = (
            final_right_load_fraction >= .8
            and center_of_mass_error_norm.max().item() <= .03
            and (executed_steps - 1) * PD_TIMESTEP >= args.motion_warmup_duration + 3.
            and all(np.all(sample["commanded_mode"] == 3) for sample in tracking_history)
        )
        motion_tracking_valid &= transfer_valid
        print(f"weight transfer: right load fraction={final_right_load_fraction:.3f}, "
              f"valid={transfer_valid} (requires >=0.8, COM error <=0.03 m, completed ramp, double support)")
    print(f"trajectory checks: peak COM error={peak_com_error:.6f} m, "
          f"peak servo RMSE={peak_servo_rmse:.6f} rad, "
          f"sustained oscillation={bool(servo_oscillating)}, "
          f"max per-joint saturation fraction={servo_saturation_fraction.item():.4f}")

    active = output.active_row_mask.bool()
    before_max = (
        output.contact_residual_before[active].abs().max().item()
        if active.any()
        else 0.0
    )
    after_max = (
        output.contact_residual_after[active].abs().max().item()
        if active.any()
        else 0.0
    )
    values = {
        "qdd_fabric": (qdd_fabric, (1, 35)),
        "mass_matrix": (mass, (1, 35, 35)),
        "bias_force": (bias_force, (1, 35)),
        "left_foot_jacobian": (left_j, (1, 6, 35)),
        "right_foot_jacobian": (right_j, (1, 6, 35)),
        "qdd_contact_consistent": (qdd_dc, (1, 35)),
        "qdd_dynamics_feasible": (
            qdd_dynamics_feasible,
            (1, 35),
        ),
        "joint_velocity_desired": (
            joint_velocity_desired,
            (1, 29),
        ),
        "joint_torque_feedforward": (
            joint_torque_feedforward,
            (1, 29),
        ),
        "joint_torque_pd": (joint_torque_pd, (1, 29)),
        "joint_position_desired": (
            joint_position_desired,
            (1, 29),
        ),
        "joint_torque": (joint_torque, (1, 29)),
        "joint_position_after_control": (
            joint_position_after_control,
            (1, 29),
        ),
        "joint_velocity_after_control": (
            joint_velocity_after_control,
            (1, 29),
        ),
        "actual_center_of_mass": (actual_center_of_mass, (1, 3)),
        "center_of_mass_target": (center_of_mass_target, (1, 3)),
        "center_of_mass_error": (center_of_mass_error, (1, 3)),
        "center_of_mass_error_norm": (center_of_mass_error_norm, (1,)),
        "joint_position_rmse": (joint_position_rmse, (1,)),
    }
    for name, (value, shape) in values.items():
        assert tuple(value.shape) == shape
        assert torch.isfinite(value).all(), f"{name} contains NaN/Inf"
        print(f"{name:27s} shape={tuple(value.shape)}")

    print(f"device={device}")
    print(
        "contact_mode="
        f"{int(output.contact_mode.item())} "
        f"(source={'PKL reference-frame' if motion_trajectory is not None else 'PhysX force'}, "
        f"left_fz={foot_normal_forces[0, 0].item():.3f} N, "
        f"right_fz={foot_normal_forces[0, 1].item():.3f} N)"
    )
    print(
        f"PD simulation={executed_steps}/{physics_steps} steps, "
        f"{args.control_frequency} Hz control updates={control_updates}, "
        f"requested duration={args.duration:.3f}s, "
        f"dt={PD_TIMESTEP:.4f}s"
    )
    print(f"|qdd_fabric|={torch.linalg.vector_norm(qdd_fabric).item():.6f}")
    print(f"|qdd_dc|={torch.linalg.vector_norm(qdd_dc).item():.6f}")
    print(
        "|qdd_dynamics_feasible|="
        f"{torch.linalg.vector_norm(qdd_dynamics_feasible).item():.6f}"
    )
    print(
        "Acceleration QP: "
        f"iters={qp_output.iters}, "
        f"base_error={qp_output.base_dynamics_error.max().item():.6e}, "
        "contact_error="
        f"{qp_output.contact_acceleration_error.max().item():.6e}, "
        "joint_dynamics_error="
        f"{qp_output.joint_dynamics_error.max().item():.6e}, "
        "friction_violation="
        f"{qp_output.friction_violation.max().item():.6e}, "
        "effort_violation="
        f"{qp_output.effort_violation.max().item():.6e}"
    )
    print(
        "|joint_torque_feedforward|="
        f"{torch.linalg.vector_norm(joint_torque_feedforward).item():.6f} Nm"
    )
    print(
        "|joint_torque_pd|="
        f"{torch.linalg.vector_norm(joint_torque_pd).item():.6f} Nm"
    )
    print(
        "|joint_torque|="
        f"{torch.linalg.vector_norm(joint_torque).item():.6f} Nm"
    )
    print(
        "max |joint_torque|="
        f"{joint_torque.abs().max().item():.6f} Nm"
    )
    print(f"max active residual before={before_max:.6e}")
    print(f"max active residual after ={after_max:.6e}")
    print(
        "COM actual/target/error="
        f"{actual_center_of_mass[0].cpu().tolist()} / "
        f"{center_of_mass_target[0].cpu().tolist()} / "
        f"{center_of_mass_error[0].cpu().tolist()}"
    )
    if motion_trajectory is not None:
        print(
            "tracking metrics: "
            f"COM error={center_of_mass_error_norm.item():.6f} m "
            f"(limit={args.maximum_com_tracking_error:.6f}), "
            f"COM height={actual_center_of_mass[0, 2].item():.6f} m "
            f"(minimum={minimum_allowed_com_height.item():.6f}), "
            f"joint RMSE={joint_position_rmse.item():.6f} rad "
            f"(limit={args.maximum_joint_position_rmse:.6f})"
        )
    print(f"Isaac DOF order={isaac_joint_names}")

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    validation_scope = (
        "first_source_step_probe" if args.mocap_first_step_report is not None else
        "double_support_unloading_only" if args.mocap_unload_test else
        "unified_first_source_step" if args.unified_source_step_test else
        "unified_first_transfer" if args.unified_transfer_test else
        "single_support" if args.single_support_test else
        "weight_transfer" if args.weight_transfer_test else
        "fixed_joint_servo" if args.fixed_joint_test else
        "static_reference" if motion_trajectory is None or motion_frame == motion_start_frame else
        "motion_segment"
    )
    if 'stance_heading_reference' in locals():
        args.output_path.with_suffix('.stance_heading.json').write_text(json.dumps(dict(
            events=stance_heading_reference.events,
            peak_position_residual=stance_heading_reference.peak_position_residual,
            peak_orientation_residual=stance_heading_reference.peak_orientation_residual,
            peak_heading_correction=stance_heading_reference.peak_heading_correction,
            peak_joint_correction=stance_heading_reference.peak_joint_correction), indent=2)+'\n')
    if 'measured_stance_reference' in locals():
        args.output_path.with_suffix('.stance_reference.json').write_text(json.dumps(dict(
            events=measured_stance_reference.events,
            peak_position_residual=measured_stance_reference.peak_position_residual,
            peak_orientation_residual=measured_stance_reference.peak_orientation_residual,
            peak_joint_correction=measured_stance_reference.peak_joint_correction),indent=2)+'\n')
    np.savez_compressed(
        args.output_path,
        controller_options=np.asarray(json.dumps(vars(args), default=str, sort_keys=True)),
        q=q.detach().cpu().numpy(),
        q_final=final_q.detach().cpu().numpy(),
        qd=qd.detach().cpu().numpy(),
        qdd_fabric=qdd_fabric.detach().cpu().numpy(),
        qdd_contact_consistent=qdd_dc.detach().cpu().numpy(),
        qdd_dynamics_feasible=(
            qdd_dynamics_feasible.detach().cpu().numpy()
        ),
        qdd_predicted=qp_output.predicted_acc.cpu().numpy(),
        task_profile=np.asarray(args.task_profile),
        motion_ground_shift=np.asarray(motion_ground_shift),
        **({} if qp_output.task_residuals is None else {
            "task_residual_" + name: residual.detach().cpu().numpy()
            for name, residual in qp_output.task_residuals.items()
        }),
        qdd_nominal=qp_output.nominal_acc.cpu().numpy(),
        qp_tracks_predicted_base=np.asarray(args.qp_track_predicted_base and not args.fixed_joint_test),
        self_collisions_disabled=np.asarray(args.disable_self_collisions),
        servo_bandwidth=np.asarray(args.servo_bandwidth),
        left_shoulder_roll_offset=np.asarray(args.left_shoulder_roll_offset),
        fabric_robot_name=np.asarray(fabric.g1_robot_name),
        nominal_contact_wrench=qp_output.nominal_contact_wrench.cpu().numpy(),
        contact_wrench=qp_output.contact_wrench.detach().cpu().numpy(),
        qp_converged=qp_output.converged.detach().cpu().numpy(),
        qp_validated=qp_output.validated.detach().cpu().numpy(),
        qp_objective=qp_output.objective.detach().cpu().numpy(),
        qp_base_dynamics_error=(
            qp_output.base_dynamics_error.detach().cpu().numpy()
        ),
        qp_joint_dynamics_error=(
            qp_output.joint_dynamics_error.detach().cpu().numpy()
        ),
        qp_contact_acceleration_error=(
            qp_output.contact_acceleration_error.detach().cpu().numpy()
        ),
        qp_friction_violation=(
            qp_output.friction_violation.detach().cpu().numpy()
        ),
        qp_effort_violation=(
            qp_output.effort_violation.detach().cpu().numpy()
        ),
        joint_position_desired=(
            joint_position_desired.detach().cpu().numpy()
        ),
        joint_velocity_desired=(
            joint_velocity_desired.detach().cpu().numpy()
        ),
        joint_torque_unclipped=(
            joint_torque_unclipped.detach().cpu().numpy()
        ),
        joint_torque_feedforward=(
            joint_torque_feedforward.detach().cpu().numpy()
        ),
        joint_torque_pd=joint_torque_pd.detach().cpu().numpy(),
        joint_torque=joint_torque.detach().cpu().numpy(),
        kp=kp.detach().cpu().numpy(),
        kd=kd.detach().cpu().numpy(),
        effort_limits=effort_limits.detach().cpu().numpy(),
        fabric_dt=np.asarray(FABRIC_TIMESTEP, dtype=np.float32),
        pd_dt=np.asarray(PD_TIMESTEP, dtype=np.float32),
        joint_torque_isaac=joint_torque_isaac,
        physx_drive_kp=_to_numpy(drive_kp),
        physx_drive_kd=_to_numpy(drive_kd),
        actuator_control_mode=np.asarray("explicit_pd_plus_qp_feedforward"),
        fixed_joint_test=np.asarray(args.fixed_joint_test),
        servo_joint_order=np.asarray(isaac_joint_names),
        **{
            "servo_" + key: np.asarray([sample[key] for sample in servo_history])
            for key in (servo_history[0] if servo_history else {})
        },
        joint_position_after_control=(
            joint_position_after_control.detach().cpu().numpy()
        ),
        joint_velocity_after_control=(
            joint_velocity_after_control.detach().cpu().numpy()
        ),
        mass_matrix=mass.detach().cpu().numpy(),
        bias_force=bias_force.detach().cpu().numpy(),
        left_foot_jacobian=left_j.detach().cpu().numpy(),
        right_foot_jacobian=right_j.detach().cpu().numpy(),
        active_row_mask=output.active_row_mask.detach().cpu().numpy(),
        contact_mode=output.contact_mode.detach().cpu().numpy(),
        foot_normal_forces=foot_normal_forces.detach().cpu().numpy(),
        actual_center_of_mass=actual_center_of_mass.cpu().numpy(),
        center_of_mass_target=center_of_mass_target.cpu().numpy(),
        center_of_mass_error=center_of_mass_error.cpu().numpy(),
        center_of_mass_error_norm=center_of_mass_error_norm.cpu().numpy(),
        joint_position_rmse=joint_position_rmse.cpu().numpy(),
        integration_initial_position=integration_initial_position.cpu().numpy(),
        integration_initial_velocity=integration_initial_velocity.cpu().numpy(),
        joint_position_limits=joint_position_limits.cpu().numpy(),
        joint_command_speed_limit=joint_command_speed_limit.cpu().numpy(),
        motion_tracking_valid=motion_tracking_valid.cpu().numpy(),
        validation_scope=np.asarray(validation_scope),
        reference_status=np.asarray(reference_status),
        self_contact_events=np.asarray(self_contact_events, dtype=str).reshape(-1, 4),
        require_support_load=np.asarray(args.require_support_load),
        phase_hold_duration=np.asarray(phase_hold_updates * FABRIC_TIMESTEP),
        weight_transfer_test=np.asarray(args.weight_transfer_test),
        align_balance_pelvis=np.asarray(args.align_balance_pelvis),
        jdot_filter_alpha=np.asarray(args.jdot_filter_alpha),
        final_right_load_fraction=np.asarray(final_right_load_fraction),
        servo_valid=servo_valid.numpy(),
        servo_oscillating=servo_oscillating.numpy(),
        servo_saturation_fraction=servo_saturation_fraction.numpy(),
        peak_com_error=np.asarray(peak_com_error),
        peak_servo_rmse=np.asarray(peak_servo_rmse),
        peak_pkl_joint_rmse=np.asarray(peak_pkl_joint_rmse),
        peak_pkl_root_position_error=np.asarray(peak_pkl_root_error),
        motion_tracking_evaluated=np.asarray(motion_trajectory is not None),
        simulation_completed=np.asarray(executed_steps == physics_steps or stopped_at_motion_end or stopped_at_tracking_steps or stopped_at_source_step_end or stopped_at_unload_end),
        physics_elapsed_seconds=np.asarray(world.current_time - initial_physics_time),
        physics_step_seconds=np.asarray(PD_TIMESTEP),
        torque_update_seconds=np.asarray(PD_TIMESTEP),
        final_native_physics_step_seconds=np.asarray(contact_sample_dt),
        rendering_step_seconds=np.asarray(world.get_rendering_dt()),
        source_frame_count=np.asarray(0 if motion_trajectory is None else len(motion_trajectory)),
        original_source_frame_count=np.asarray(getattr(args,"original_source_frame_count",0 if motion_trajectory is None else len(motion_trajectory))),
        requested_tracking_steps=np.asarray(args.tracking_steps),
        full_motion_requested=np.asarray(args.track_full_motion),
        source_start_frame=np.asarray(motion_start_frame),
        source_completed=np.asarray(motion_completion is not None and motion_completion.complete),
        stopped_at_motion_end=np.asarray(stopped_at_motion_end),
        stopped_at_tracking_steps=np.asarray(stopped_at_tracking_steps),
        stopped_at_unload_end=np.asarray(stopped_at_unload_end),
        stopped_at_source_step_end=np.asarray(stopped_at_source_step_end),
        final_joint_order=np.asarray(G1_JOINT_NAMES),
        **{
            "history_" + key: np.asarray([sample[key] for sample in tracking_history])
            for key in (tracking_history[0] if tracking_history else {})
        },
    )
    print(f"saved: {args.output_path}")
    print(
        "[pass] numerical pipeline: pelvis + COM + feet + hand Fabric -> "
        + ("metric summation -> constrained dynamics -> persistent MIT integration" if args.g1_md else
         "contact projection -> dynamics acceleration QP -> integration")
    )
    tracking_failed = (
        motion_trajectory is not None
        and (not bool(motion_tracking_valid.all())
             or ((args.until_motion_end or args.track_full_motion) and not stopped_at_motion_end))
    )
    if args.mocap_unload_test and args.mocap_first_step_report is None:
        from analyze_contact_transfer import summarize_unloading
        with np.load(args.output_path) as recording:
            unload_report = summarize_unloading(recording)
        args.output_path.with_suffix(".unloading.json").write_text(json.dumps(unload_report, indent=2)+"\n")
        tracking_failed |= not unload_report["unloading_passed"]
        print("Double-support unloading: "+json.dumps(unload_report), flush=True)
    if args.unified_transfer_test and not args.unified_source_step_test:
        from analyze_contact_transfer import summarize_transfer
        with np.load(args.output_path) as recording:
            transfer_report = summarize_transfer(recording)
        args.output_path.with_suffix(".transfer.json").write_text(json.dumps(transfer_report, indent=2)+"\n")
        tracking_failed |= not transfer_report["transfer_passed"]
    if measured_steps is not None:
        from analyze_bounded_steps import summarize as summarize_steps
        with np.load(args.output_path) as recording:
            steps_report = summarize_steps(recording, json.loads(
                args.output_path.with_suffix('.steps_plan.json').read_text()))
        args.output_path.with_suffix('.steps.json').write_text(json.dumps(steps_report, indent=2)+'\n')
        print(f"[{'pass' if steps_report['steps_tracking_passed'] else 'fail'}] "
              f"source landings={steps_report['measured_steps']}/{steps_report['expected_steps']}, "
              f"PKL fidelity={steps_report['pkl_fidelity_passed']}; "
              f"report: {args.output_path.with_suffix('.steps.json')}", flush=True)
        if not args.mocap_unload_test:
            tracking_failed |= not steps_report['steps_tracking_passed']
    if (args.until_motion_end and not args.tracking_steps) or args.track_full_motion:
        from analyze_motion_tracking import summarize_tracking, summarize_full_motion
        with np.load(args.output_path) as recording:
            full_tracking = (summarize_full_motion(recording, json.loads(
                args.output_path.with_suffix('.steps_plan.json').read_text()))
                if args.track_full_motion else summarize_tracking(recording))
        if fidelity_stop is not None:
            full_tracking['first_fidelity_exceeded']=fidelity_stop
            if args.stop_at_fidelity_loss:
                full_tracking.update(stop_reason='Original-source fidelity limit exceeded',fidelity_stop=fidelity_stop)
            else:
                full_tracking['diagnostic_continuation_after_fidelity_loss']=True
        full_report_path = args.output_path.with_suffix('.tracking.json')
        full_report_path.write_text(json.dumps(full_tracking,indent=2)+'\n')
        passed = full_tracking['full_motion_tracking_passed' if args.track_full_motion else 'full_tracking_passed']
        tracking_failed |= not passed
        print(f"[{'pass' if passed else 'fail'}] full source coverage and PKL fidelity: {full_report_path}")
    if motion_trajectory is not None:
        print(
            f"[{'pass' if bool(motion_tracking_valid.all()) else 'fail'}] {validation_scope} controller validation (not PKL fidelity)"
        )
        if args.until_motion_end and not stopped_at_motion_end:
            print(f"[fail] simulation budget exhausted before stable source completion: frame {frame_position:.2f}/{len(motion_trajectory)-1}")
    sys.stdout.flush()
    sys.stderr.flush()
    if args.show_isaac_window and args.keep_isaac_window:
        world.pause()
        print('Tracking finished; Isaac scene paused. Close the window to exit.', flush=True)
        while simulation_app.is_running():
            world.render()
            time.sleep(1./60.)
    if tracking_failed and simulation_app.is_running():
        # Set Kit's exit status before fast shutdown terminates Python.
        # An exception after close() alone is never reached in that mode.
        import omni.kit.app

        omni.kit.app.get_app().post_quit(1)
    simulation_app.close()
    if tracking_failed:
        raise RuntimeError(
            "Motion tracking failed controller checks or required source completion; "
            f"diagnostics were saved to {args.output_path}."
        )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        if _active_simulation_app is not None and _active_simulation_app.is_running():
            import omni.kit.app
            omni.kit.app.get_app().post_quit(1)
            _active_simulation_app.close()
        raise
