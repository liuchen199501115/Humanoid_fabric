"""AMP full-motion tracking with synchronous 500 Hz direct QP effort.

Run from the repository root:
    PYTHONPATH=src:examples:.runtime/qp_solvers python examples/test_g1.py \
        --tracking-profile amp-full

Live measured q/v and dynamics -> feedback task accelerations -> constrained
whole-body QP (acceleration, contact wrench, joint torque) -> limited QP torque
-> Isaac physics -> next measured state. PhysX joint drives are disabled.

One-step joint predictions are diagnostics, not motor position/velocity targets.
The inter-tick tracking servo is inactive at the retained 500 Hz / 500 Hz rates.
Archived modes and experimental branches below are commented for reference;
validate_direct_wbc_flow rejects options that would require them. Shared parser
and helper APIs are retained for other repository analysis entry points.
Physical limits, collision/contact handling and original-source validation stay
active. Historical 12/29 landings do not establish full-motion success.
"""

# """Track motion PKLs using Fabric tasks and a constrained dynamics QP.

# Full AMP attempt from the repository root:
    # PYTHONPATH=src:examples:.runtime/qp_solvers python examples/test_g1.py \
        # --tracking-profile amp-full
# The selected baseline verifies 12/29 landings, not complete tracking. This
# entry retains the complete source, native contacts and original error limits.
# Historical experiment helpers under acc_data are required by this profile.

# With --g1-md, use natural-form task summation as the dynamics objective,
# persistent joint integration and MIT effort output at 1 kHz / 2 kHz PhysX.
# Contact feasibility remains checked; see docs/g1_md_isaac.md.

# Default data flow:
    # qdd_fabric [1, 35]
        # -> Isaac Sim M [1, 35, 35] and foot J [1, 12, 35]
        # -> dynamically consistent projection
        # -> qdd_contact_consistent [1, 35]
        # -> floating-base dynamics acceleration QP
        # -> qdd_dynamics_feasible [1, 35]
        # -> actuated-joint semi-implicit integration [1, 29]

# For a clip that begins mid-motion, --motion-start-in-motion initializes
# source q/v and near-ground soles, primes fresh PhysX contacts, and follows
# the source clock without a stationary warmup. It does not certify tracking.

# With a motion PKL, the closed-loop simulation reconciles planned contacts
# with measured foot forces. PKL poses provide task references; joint commands
# come from integrating the QP command acceleration without hard resets.
# Its bounded synchronization correction is excluded from physical feedforward
# acceleration, but remains inside the QP's command and total-effort constraints.
# The QP also predicts the contact-consistent response to explicit feedback.
# Explicit 500 Hz joint PD
# and QP feedforward are combined and torque-limited before being sent to
# PhysX, whose internal joint drives remain disabled. Numerical feasibility does not establish
# motion tracking success; the final state and sampled history are saved for
# separate tracking validation.
# """

from __future__ import annotations

import argparse
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
from fabrics_sim.solver.contact_observation import ground_support_forces
from fabrics_sim.solver.AccelerationQP import split_foot_wrench_constraints
from fabrics_sim.solver.com_diagnostics import com_acceleration_snapshot, complete_observed_interval
# from fabrics_sim.solver.swing_clearance import clearance_active, sole_clearance_constraints
from fabrics_sim.solver.rigid_body_frames import (
    origin_to_com_map, dynamics_at_origin, jacobians_at_origins,
)
from fabrics_sim.solver.AccelerationQP import (
    G1DynamicsAccelerationQP,
    build_single_support_wrench_constraints,
)
# from fabrics_sim.solver.joint_control import (
    # explicit_pd_gains, integration_acceleration_bounds, integrated_servo_model, servo_health,
    # synchronize_command_acceleration,
    # bounded_reference_step,
    # measured_joint_acceleration_bounds,
    # joint_velocity_constraints,
    # joint_limit_avoidance_request,
    # contact_servo_gain_bank,
    # implicit_servo_response,
# )
from fabrics_sim.solver.joint_control import (
    servo_health,
    bounded_reference_step,
    measured_joint_acceleration_bounds,
    joint_velocity_constraints,
    joint_limit_avoidance_request,
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

# from test_g1_two_tasks import (
    # FABRIC_TIMESTEP,
    # G1_JOINT_NAMES,
    # G1_USD_PATH,
    # PD_TIMESTEP,
    # load_motion_trajectory,
    # load_motion_contact_sequence,
    # unitree_g1_effort_limits,
    # unitree_g1_pd_gains,
# )
from test_g1_two_tasks import (
    FABRIC_TIMESTEP,
    G1_JOINT_NAMES,
    G1_USD_PATH,
    PD_TIMESTEP,
    load_motion_trajectory,
    load_motion_contact_sequence,
    unitree_g1_effort_limits,
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
                          metric, stiffness=100., damping=20., feedback_limit=30.):
    """Track the moving root origin in world coordinates through the physical QP.

    This is a soft acceleration objective. It cannot apply forces to the
    unactuated base; the existing dynamics and measured contacts still govern
    what joint torque can realize. Feedforward is not feedback-clipped.
    """
    from fabrics_sim.fabrics.task_request import TaskRequest
    if (any(not math.isfinite(x) or x <= 0 for x in
            (metric, stiffness, damping, feedback_limit))):
        raise ValueError('Root task weights and gains must be finite and positive')
    feedback = (stiffness*(reference_q[:, :3]-q[:, :3])
                + damping*(reference_v[:, :3]-velocity[:, :3]))
    desired = reference_a[:, :3]+feedback.clamp(-feedback_limit, feedback_limit)
    A = q.new_zeros(q.shape[0], 3, 35)
    A[:, :, :3] = torch.eye(3, device=q.device, dtype=q.dtype)
    return TaskRequest('root_position', A, torch.zeros_like(desired), desired,
                       metric*torch.eye(3, device=q.device, dtype=q.dtype)[None].expand(q.shape[0], -1, -1),
                       torch.ones_like(desired, dtype=torch.bool))


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


def configure_full_tracking_profile(parser):
    """Reproduce the measured AMP baseline through this controller entry.

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
     '--full-swing-pose-start', '1.4', '--relax-joint-tube', '--recover-orientation-tube',
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
    preset += ['--output-path', 'acc_data/g1_full_tracking_20261106/full_motion.npz']
    args = parser.parse_args(preset + sys.argv[1:])
    if (args.tracking_steps or not args.track_full_motion
            or not args.stop_at_fidelity_loss or not args.direct_wbc or args.g1_md):
        parser.error('amp-full requires direct WBC, the whole source and original fidelity checks')
    if args.full_body_preview_start != float('inf') or args.centroidal_preview_start != float('inf'):
        parser.error('amp-full uses the validated nonpredictive WBC baseline')
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


# def inverse_dynamics_joint_feedforward(
    # mass_matrix: torch.Tensor,
    # bias_force: torch.Tensor,
    # contact_jacobian: torch.Tensor,
    # acceleration: torch.Tensor,
    # contact_wrench: torch.Tensor,
# ) -> torch.Tensor:
    # """Map QP acceleration and contact wrench to 29-D joint torque.

    # From M*qdd + h = S^T*tau + Jc^T*lambda, the actuated rows give
    # tau = Mj*qdd + hj - Jcj^T*lambda.
    # """
    # inertial_and_bias = torch.bmm(
        # mass_matrix[:, 6:35, :], acceleration.unsqueeze(-1)
    # ).squeeze(-1) + bias_force[:, 6:35]
    # contact_torque = torch.bmm(
        # contact_jacobian[:, :, 6:35].transpose(1, 2),
        # contact_wrench.unsqueeze(-1),
    # ).squeeze(-1)
    # return inertial_and_bias - contact_torque


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
            "Run amp-full at 500 Hz with direct QP effort. Legacy options remain "
            "parseable for analysis compatibility; archived execution modes are rejected."
        )
    )
    parser.add_argument("--g1-md", action="store_true",
                        help="Use g1.md Fabric composition -> constrained dynamics -> persistent MIT integration in Isaac Sim.")
    parser.add_argument("--motion-pkl", type=Path, default=None)
    parser.add_argument("--motion-start-in-motion", action="store_true",
                        help="Initialize from source q/v and touching soles, then track the source clock without rest warmup or a managed transfer.")
    parser.add_argument("--dynamic-unload-weight", type=float, default=0.,
                        help="Optional outgoing-wrench penalty during measured double support before a planned dynamic foot release.")
    parser.add_argument("--preserve-reference-acceleration", action="store_true",
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
    parser.add_argument("--joint-position-limits", action=argparse.BooleanOptionalAction, default=True,
                        help="Enforce simulator joint angle bounds with a 0.01 rad margin and measured-state braking in the QP.")
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
    parser.add_argument("--joint-limit-avoidance-weight", type=float, default=None,
                        help="Soft upper/lower joint-limit task weight (default: 20 for dynamics-aware, 0 for legacy); 0 disables this task.")
    parser.add_argument("--joint-limit-time-constant", type=float, default=None,
                        help="Optional measured-joint velocity damper time in seconds; ramps in over 0.5 s after warmup.")
    parser.add_argument("--joint-limit-barrier-bandwidth", type=float, default=None,
                        help="Optional joint-distance acceleration barrier bandwidth in rad/s; starts braking before a joint stop.")
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
    parser.add_argument('--clock-transition-start', type=float, default=1.34)
    parser.add_argument('--clock-transition-ramp', type=float, default=.2)
    parser.add_argument('--clock-final-speed', type=float, default=.7)
    parser.add_argument('--clock-second-start',type=float,default=100.)
    parser.add_argument('--clock-second-ramp',type=float,default=.3)
    parser.add_argument('--clock-second-speed',type=float,default=None)
    parser.add_argument('--clock-third-start',type=float,default=100.)
    parser.add_argument('--clock-third-ramp',type=float,default=.2)
    parser.add_argument('--clock-third-speed',type=float,default=None)
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
    parser.add_argument('--minimum-support-acceleration-start',type=float,default=100.)
    parser.add_argument('--minimum-support-acceleration',type=float,default=-3.)
    parser.add_argument('--minimum-support-acceleration-metric-scale',type=float,default=2.)
    parser.add_argument('--weighted-tasks-start', type=float, default=float('inf'),
                        help='Source seconds to blend secondary costs into one constrained task QP')
    parser.add_argument('--weighted-tasks-ramp', type=float, default=.1)
    parser.add_argument('--support-flatten-start', type=float, default=float('inf'),
                        help='Source seconds to blend loaded planned stance feet toward a level sole')
    parser.add_argument('--root-position-task-start', type=float, default=float('inf'),
                        help='Source seconds to enable direct root-position acceleration tracking')
    parser.add_argument('--root-position-task-weight', type=float, default=1000.)
    parser.add_argument('--root-position-task-ramp', type=float, default=.1)
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
    parser.add_argument('--late-pelvis-metric-scale',type=float,default=1.,help='Scale only the promoted pelvis task after its start, preserving the earlier controller.')
    parser.add_argument('--late-pelvis-yaw-only',action='store_true')
    parser.add_argument('--contact-friction-rays',type=int,choices=[8,16],default=8)
    parser.add_argument('--contact-friction-refine-start',type=float,default=float('inf'))
    parser.add_argument('--support-orientation-priority-start', type=float, default=100.)
    parser.add_argument('--support-orientation-metric-scale', type=float, default=1.)
    parser.add_argument('--support-orientation-roll-pitch-only',action='store_true',help='Promote only the free support-foot tilt axes, with a 60 ms smooth onset; contact constraints and source q/v/a are unchanged.')
    parser.add_argument('--upper-posture-metric-start', type=float, default=100.)
    parser.add_argument('--upper-posture-metric-scale', type=float, default=1.)
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
    parser.add_argument('--late-foot-metric-start', type=float, default=100.)
    parser.add_argument('--late-foot-metric-scale', type=float, default=1.)
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
    parser.add_argument('--late-contact-stabilization-start', type=float, default=100.)
    parser.add_argument('--late-contact-stabilization-kp', type=float, default=20.)
    parser.add_argument('--late-contact-stabilization-kd', type=float, default=8.)
    parser.add_argument('--tube-reference-method', choices=['pchip','bspline'], default='pchip')
    parser.add_argument('--tracking-tube-orientation-bandwidth', type=float, default=None)
    parser.add_argument('--tracking-tube-joint-bandwidth', type=float, default=None)
    parser.add_argument('--late-joint-tube-start', type=float, default=100.)
    parser.add_argument('--late-joint-tube-bandwidth', type=float, default=4.)
    parser.add_argument('--tracking-margin-start', type=float, default=float('inf'))
    parser.add_argument('--tracking-margin-reaction-time', type=float, default=.08)
    parser.add_argument('--tracking-margin-maximum', type=float, default=16.)
    parser.add_argument('--tracking-tube-start', type=float, default=1.34)
    parser.add_argument('--tracking-tube-end', type=float, default=100.)
    parser.add_argument('--contact-pause-start', type=float, default=4.2)
    parser.add_argument('--contact-pause-end', type=float, default=None)
    parser.add_argument('--contact-pause-landings', type=int, default=11)
    parser.add_argument('--contact-pause-resume', type=float, default=.3)
    parser.add_argument('--contact-pause-early-resume', action='store_true')
    parser.add_argument('--late-tracking-tube-start', type=float, default=100.)
    parser.add_argument('--late-tracking-tube-bandwidth', type=float, default=10.)
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
    parser.add_argument('--tracking-tube-bandwidth', type=float, default=10.)
    parser.add_argument('--late-leg-feedback-start', type=float, default=1.88)
    parser.add_argument('--late-leg-feedback-limit', type=float, default=150.)
    parser.add_argument('--late-leg-feedback-stance', action='store_true')
    parser.add_argument('--late-leg-feedback-side', choices=['right','both'], default='right')
    parser.add_argument('--waist-feedback-start', type=float, default=1.7)
    parser.add_argument('--waist-feedback-limit', type=float, default=100.)
    parser.add_argument('--waist-metric-scale', type=float, default=4.)
    parser.add_argument('--flight-feedback-joints', choices=['ankles','leg'], default='ankles')
    parser.add_argument('--flight-feedback-start', type=float, default=0.)
    parser.add_argument('--flight-feedback-ramp', type=float, default=.1)
    return configure_full_tracking_profile(parser)


# Fixed selectors for the only executable flow in this entry. Legacy argument
# parsing and shared helper APIs remain available to existing analysis tools.
_DIRECT_WBC_OPTIONS = {
    'late_leg_feedback_start': 100.,
    'waist_feedback_start': 100.,
    'com_feedback_start': 100.,
    'acceleration_and_braking_limits': False,
    'align_balance_pelvis': False,
    'align_takeoff_frame': False,
    'balance_body_priority': False,
    'balance_priority': True,
    'body_orientation_priority': False,
    'centroidal_balance_priority': False,
    'centroidal_preview_start': float('inf'),
    'clock_second_speed': None,
    'clock_second_start': 100.0,
    'clock_third_speed': None,
    'clock_third_start': 100.0,
    'collision_barrier_bandwidth_start': float('inf'),
    'configuration_tracking_start': float('inf'),
    'contact_aware_servo': True,
    'contact_cycle_test': None,
    'contact_friction_refine_start': float('inf'),
    'contact_mode': None,
    'contact_pause_end': None,
    'control_frequency': 500,
    'convex_wrist_collision': True,
    'coordinated_source_start': False,
    'cycle_crouch_depth': 0.0,
    'direct_wbc': True,
    'disable_collision_task': False,
    'disable_momentum_task': False,
    'disable_self_collisions': False,
    'early_touchdown': False,
    'fixed_joint_test': False,
    'flat_foot_spawn': False,
    'full_body_preview_start': float('inf'),
    'g1_com_priority': False,
    'g1_md': False,
    'gate_contact_transitions': True,
    'ground_motion': True,
    'horizontal_balance_priority': False,
    'joint_correction_overlay': None,
    'joint_limit_priority_start': 100.0,
    'joint_limit_time_constant': None,
    'joint_position_limits': True,
    'joint_velocity_limits': True,
    'landing_orientation_priority': False,
    'landing_speed_limit': None,
    'late_balance_hierarchy_start': 100.0,
    'late_contact_stabilization_start': 100.0,
    'late_foot_angular_metric_end': float('inf'),
    'late_foot_angular_metric_start': 100.0,
    'late_joint_tube_start': 100.0,
    'late_momentum_priority_start': 100.0,
    'late_momentum_zero_target': False,
    'late_pelvis_priority_start': 100.0,
    'late_physics_substeps': 1,
    'late_physics_substeps_start': float('inf'),
    'late_tracking_tube_start': 100.0,
    'left_shoulder_roll_offset': 0.0,
    'leg_posture_secondary_start': float('inf'),
    'limit_predicted_acceleration': False,
    'measured_stance_reference_start': 100.0,
    'minimum_support_acceleration_start': 100.0,
    'mocap_first_step_project_torque': False,
    'mocap_first_step_report': None,
    'mocap_pipeline': False,
    'mocap_unload_test': False,
    'mocap_unload_weight': 0.0,
    'motion_interpolation': 'pchip-analytic',
    'motion_start_in_motion': True,
    'native_contact_plan_start': 100.0,
    'nominal_implicit_servo': True,
    'observed_contact_patches': True,
    'orientation_tube_end': float('inf'),
    'orientation_tube_start': float('inf'),
    'pelvis_orientation_replace_start': float('inf'),
    'physical_braking_start': 100.0,
    'physical_braking_upper_only': False,
    'physical_braking_waist_only': False,
    'physics_frequency': None,
    'position_tube_end': float('inf'),
    'posture_last_priority': False,
    'posture_metric_scale': None,
    'prepared_contact_plan': None,
    'preserve_reference_acceleration': True,
    'recover_orientation_tube': True,
    'reference_manager': False,
    'reference_mode': 'unified',
    'relax_joint_tube': True,
    'require_support_load': False,
    'retain_planned_support_start': 100.0,
    'rolling_contact_kinematics_start': float('inf'),
    'root_position_task_start': float('inf'),
    'secondary_joint_correction_overlay': None,
    'separate_balance_priority': False,
    'shoulder_roll_guard': False,
    'shoulder_roll_weight_scale': 1.0,
    'shoulder_roll_weight_start': float('inf'),
    'single_support_test': None,
    'source_arm_roll_offset': 0.0,
    'source_bridge_duration': None,
    'source_contact_events': False,
    'source_momentum_reference': False,
    'source_recovery_after_step': 0,
    'source_recovery_step_count': 0,
    'source_step_landing_search_depth': 0.0,
    'source_step_profile': None,
    'source_step_stop_duration': 0.0,
    'source_support_switch': False,
    'source_swing_pose_priority': False,
    'source_swing_position_priority': False,
    'source_swing_reference': False,
    'stance_heading_start': float('inf'),
    'stance_posture_metric_start': float('inf'),
    'stance_posture_secondary_start': 100.0,
    'stop_at_fidelity_loss': True,
    'support_flatten_start': float('inf'),
    'support_knee_min_angle': None,
    'support_load_floor_start': float('inf'),
    'support_orientation_priority_start': 100.0,
    'swing_balance_priority_start': 100.0,
    'swing_clearance_priority': False,
    'swing_metric_scale': None,
    'swing_pose_priority': False,
    'swing_position_priority': True,
    'synchronous_effort_budget_start': 100.0,
    'task_profile': 'dynamics-aware',
    'torque_continuity_start': float('inf'),
    'track_full_motion': True,
    'tracking_profile': 'amp-full',
    'tracking_servo': True,
    'tracking_steps': 0,
    'tracking_tube_joint_bandwidth': None,
    'tracking_tube_orientation_bandwidth': None,
    'transfer_balance_priority': False,
    'transition_swing_priority_start': 100.0,
    'unified_source_step_test': False,
    'unified_transfer_test': False,
    'until_motion_end': False,
    'upper_body_priority': False,
    'use_sole_wrench_geometry': True,
    'weight_transfer_test': False,
    'weighted_tasks_start': float('inf'),
    'whole_body_posture_priority': True,
    'wrist_body_avoidance': True,
    'wrist_hip_avoidance': True,
    'wrist_hip_barrier': True,
}


def validate_direct_wbc_flow(args):
    """Reject archived execution modes before constructing Isaac Sim."""
    changed = [name for name, expected in _DIRECT_WBC_OPTIONS.items()
               if getattr(args, name) != expected]
    if changed:
        raise ValueError("This entry now runs only the amp-full 500 Hz direct-QP flow; "
                         "archived options differ: " + ", ".join(changed))
    # Dormant >=100 s experimental gates are commented out for this 18 s run.
    if not math.isfinite(args.duration) or not 0 < args.duration <= 18.:
        raise ValueError("The retained amp-full flow supports 0 < duration <= 18 s")


def main() -> None:
    global FABRIC_TIMESTEP, PD_TIMESTEP
    args = parse_args()
    validate_direct_wbc_flow(args)
    # from g1_shoulder_roll import shoulder_roll_multiplier, apply_shoulder_roll_weights
    # shoulder_roll_multiplier(0., args.shoulder_roll_weight_start,
                             # args.shoulder_roll_weight_ramp, args.shoulder_roll_weight_scale)
    # [archived branch] if args.shoulder_roll_weight_scale > 1 and (
            # [archived branch] not math.isfinite(args.shoulder_roll_weight_start)
            # [archived branch] or args.tracking_profile != 'amp-full' or not args.direct_wbc):
        # [archived branch] raise ValueError('Shoulder weight experiment requires amp-full direct WBC and a finite start')
    shoulder_guard = None
    # [archived branch] if args.shoulder_roll_guard:
        # [archived branch] from g1_guarded_shoulder import GuardedShoulderWeights
        # [archived branch] if args.shoulder_roll_weight_scale <= 1 or math.isfinite(args.weighted_tasks_start):
            # [archived branch] raise ValueError('Guard requires a shoulder scale > 1 and the original priority stages')
        # [archived branch] shoulder_guard = GuardedShoulderWeights(args.shoulder_roll_guard_relative, args.shoulder_roll_guard_slack)
    if not math.isfinite(args.joint_limit_priority_scale) or args.joint_limit_priority_scale <= 0:
        raise ValueError('Joint-limit priority scale must be finite and positive')
    # [archived branch] if args.support_knee_min_angle is not None:
        # [archived branch] if (not math.isfinite(args.support_knee_min_angle) or not 0 < args.support_knee_min_angle < 1
                # [archived branch] or not math.isfinite(args.support_knee_guard_bandwidth) or args.support_knee_guard_bandwidth <= 0
                # [archived branch] or math.isnan(args.support_knee_guard_start) or args.support_knee_guard_start < 0):
            # [archived branch] raise ValueError('Invalid optional support-knee guard parameters')
    # [archived branch] if args.physical_braking_waist_only and args.physical_braking_upper_only:
        # [archived branch] raise ValueError('Choose waist-only or upper-body optional braking bounds')
    if (math.isnan(args.torque_continuity_start) or args.torque_continuity_start < 0
            or not math.isfinite(args.torque_continuity_weight) or args.torque_continuity_weight < 0
            or not math.isfinite(args.torque_continuity_ramp) or args.torque_continuity_ramp <= 0):
        raise ValueError('Torque continuity needs a nonnegative start/weight and a positive ramp')
    # [archived branch] if math.isfinite(args.torque_continuity_start) and (args.tracking_profile != 'amp-full'
            # [archived branch] or not args.direct_wbc or not args.tracking_servo):
        # [archived branch] raise ValueError('Torque continuity requires amp-full direct effort tracking')
    if (math.isnan(args.stance_heading_start) or args.stance_heading_start < 0
            or not math.isfinite(args.stance_heading_limit) or not 0 < args.stance_heading_limit <= .5
            or not math.isfinite(args.stance_heading_ramp) or args.stance_heading_ramp <= 0):
        raise ValueError('Stance heading requires a nonnegative start, limit in (0,.5], and positive ramp')
    # [archived branch] if math.isfinite(args.stance_heading_start) and (
            # [archived branch] args.tracking_profile != 'amp-full' or args.measured_stance_reference_start < float('inf')):
        # [archived branch] raise ValueError('Stance heading requires amp-full and excludes simultaneous stance position adaptation')
    # [archived branch] if math.isnan(args.late_physics_substeps_start) or args.late_physics_substeps_start < 0:
        # [archived branch] raise ValueError('Late physics substeps require a nonnegative simulation start')
    # [archived branch] if (args.late_physics_substeps > 1 and math.isfinite(args.late_physics_substeps_start)
            # [archived branch] and (args.tracking_profile != 'amp-full' or not args.direct_wbc
                 # [archived branch] or args.control_frequency != (args.physics_frequency or 500))):
        # [archived branch] raise ValueError('Late substeps require synchronous amp-full direct WBC')
    if (not math.isfinite(args.orientation_tube_radius)
            or not 0 < args.orientation_tube_radius <= .34
            or math.isnan(args.orientation_tube_start) or args.orientation_tube_start < 0):
        raise ValueError('Orientation tube requires a positive radius at most .34 rad and nonnegative source start')
    # [archived branch] if args.motion_start_in_motion:
    if (args.motion_pkl is None or args.reference_manager or args.mocap_pipeline
            or args.flat_foot_spawn or args.fixed_joint_test or args.weight_transfer_test
            or args.single_support_test or args.require_support_load
            or (not args.g1_md and args.task_profile != 'dynamics-aware')
            or args.reference_mode != 'unified'
            or args.motion_interpolation not in ('pchip-analytic', 'bspline-analytic')):
        raise ValueError('--motion-start-in-motion requires an analytic unified PKL reference and excludes stationary/managed starts')
    args.motion_warmup_duration = args.motion_ramp_duration = 0.
    # [archived branch] if (not math.isfinite(args.mocap_unload_weight) or args.mocap_unload_weight < 0
            # [archived branch] or (args.mocap_unload_weight and not args.mocap_pipeline)):
        # [archived branch] raise ValueError('--mocap-unload-weight requires the mocap pipeline and a finite nonnegative weight')
    if args.track_full_motion and (not args.motion_start_in_motion or args.tracking_steps
                                   or args.motion_frame != 0 or args.until_motion_end):
        raise ValueError('--track-full-motion requires dynamic start at frame zero and excludes bounded steps or stationary completion')
    if (not math.isfinite(args.swing_midpoint_clearance) or args.swing_midpoint_clearance < 0
            or (args.swing_midpoint_clearance and not (args.motion_start_in_motion and (args.tracking_steps or args.track_full_motion)))):
        raise ValueError('--swing-midpoint-clearance requires dynamic bounded steps or full-motion tracking')
    if (not math.isfinite(args.dynamic_unload_weight) or args.dynamic_unload_weight < 0
            or (args.dynamic_unload_weight and not args.motion_start_in_motion)):
        raise ValueError('--dynamic-unload-weight requires dynamic start and a finite nonnegative weight')
    # [archived branch] if args.tracking_steps < 0:
        # [archived branch] raise ValueError('--tracking-steps must be nonnegative')
    # [archived branch] if args.tracking_steps:
        # [archived branch] if not (args.mocap_pipeline or args.motion_start_in_motion) or args.motion_frame != 0:
            # [archived branch] raise ValueError('--tracking-steps requires mocap pipeline or dynamic start from frame zero')
        # [archived branch] if not args.motion_start_in_motion:
            # [archived branch] args.until_motion_end = True
    # [archived branch] if args.mocap_first_step_project_torque and args.mocap_first_step_report is None:
        # [archived branch] raise ValueError('--mocap-first-step-project-torque requires a bounded first-step report')
    # [archived branch] if args.mocap_first_step_report is not None:
        # [archived branch] if not args.mocap_pipeline or args.acceleration_and_braking_limits or args.source_step_count != 1:
            # [archived branch] raise ValueError("First-step probe requires mocap pipeline, one source step, and acceleration/braking limits off")
        # [archived branch] args.mocap_unload_test = True
    # [archived branch] if args.mocap_unload_test:
        # [archived branch] if not args.mocap_pipeline:
            # [archived branch] raise ValueError("--mocap-unload-test requires --mocap-pipeline")
        # [archived branch] args.until_motion_end = False
    # [archived branch] if args.mocap_pipeline:
        # [archived branch] if args.motion_pkl is None:
            # [archived branch] raise ValueError('--mocap-pipeline requires --motion-pkl')
        # [archived branch] if args.source_arm_roll_offset or args.left_shoulder_roll_offset or args.coordinated_source_start or args.unified_transfer_test or args.unified_source_step_test:
            # [archived branch] raise ValueError('Mocap pipeline excludes experimental reference replacements')
        # [archived branch] if args.motion_warmup_duration <= 0 or args.motion_ramp_duration <= 0:
            # [archived branch] raise ValueError('Mocap pipeline requires positive warmup and clock ramp durations')
        # [archived branch] if (args.source_support_switch or args.source_contact_events
                # [archived branch] or args.source_step_profile is not None or args.source_step_stop_duration):
            # [archived branch] raise ValueError('Mocap pipeline excludes reduced-model source-step/stop planners')
        # [archived branch] args.reference_mode = 'unified'
        # [archived branch] args.task_profile = 'dynamics-aware'
        # [archived branch] args.reference_manager = True
        # [archived branch] args.reference_footsteps = 'source'
        # [archived branch] args.motion_interpolation = 'pchip-analytic' if args.tracking_steps else 'bspline-analytic'
        # [archived branch] args.direct_wbc = True
        # [archived branch] if args.tracking_steps:
            # [archived branch] if args.source_recovery_after_step or args.source_recovery_step_count:
                # [archived branch] raise ValueError('Strict steps cannot contain recovery steps')
            # Track the bounded PKL directly. SourceStepRollout introduces
            # reduced COM previews and a per-step stopping planner.
            # Contact sequencing remains owned by the measured-contact manager.
    # [archived branch] if args.g1_md:
        # [archived branch] from fabrics_sim.solver.fabric_mit import configure_fabric_mit
        # [archived branch] configure_fabric_mit(args)
        # [archived branch] PD_TIMESTEP = 0.0005  # Isaac PhysX: two substeps per 1 kHz control tick.
    # [archived branch] if args.unified_source_step_test:
        # [archived branch] args.unified_transfer_test = True
    from fabrics_sim.reference.whole_body import WholeBodyReference, validate_reference_options
    unified_reference = validate_reference_options(args)
    # [archived branch] if args.unified_transfer_test and (not unified_reference or not args.reference_manager or args.until_motion_end):
        # [archived branch] raise ValueError("--unified-transfer-test requires unified references/manager and cannot use --until-motion-end")
    # [archived branch] if args.unified_source_step_test and not (args.direct_wbc and args.motor_armature
            # [archived branch] and args.control_frequency == 500 and args.observed_contact_patches
            # [archived branch] and args.early_touchdown and args.reference_footsteps == "source"):
        # [archived branch] raise ValueError("--unified-source-step-test requires direct WBC, motor armature, 500 Hz, "
                         # [archived branch] "observed contact patches, early touchdown and source footsteps")
    if not np.isfinite(args.later_step_stop_duration) or args.later_step_stop_duration <= 0:
        raise ValueError("Later source steps require a positive finite stop duration")
    source_step_profile = {}
    # [archived branch] if args.source_step_profile is not None:
        # [archived branch] source_step_profile = json.loads(args.source_step_profile.read_text())
        # [archived branch] if (not isinstance(source_step_profile,dict) or set(source_step_profile)-{'first','following'}
                # [archived branch] or any(not isinstance(value,dict) for value in source_step_profile.values())):
            # [archived branch] raise ValueError('Source step profile requires first/following option objects')
    # [archived branch] if args.source_contact_events and (not args.source_support_switch or args.source_step_count<1):
        # [archived branch] raise ValueError("Source contact events require --source-support-switch")
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
    # [archived branch] if args.source_support_switch and (not args.coordinated_source_start or args.source_step_stop_duration<=0):
        # [archived branch] raise ValueError("--source-support-switch requires coordinated start and a landing stop")
    if (not np.isfinite(args.source_step_stop_duration) or args.source_step_stop_duration < 0
            or (args.source_step_stop_duration and not args.coordinated_source_start)):
        raise ValueError("Source-step stop requires nonnegative duration and coordinated start")
    if (not np.isfinite(args.source_step_landing_search_depth) or not 0 <= args.source_step_landing_search_depth <= .01
            or (args.source_step_landing_search_depth != 0 and not args.coordinated_source_start)):
        raise ValueError("Source-step landing search requires coordinated start and depth in [0,.01]")
    if (not np.isfinite(args.source_step_landing_fraction) or not 0 < args.source_step_landing_fraction <= 1
            or (args.source_step_landing_fraction != 1 and not args.coordinated_source_start)):
        raise ValueError("--source-step-landing-fraction requires (0,1] and coordinated source start for adaptation")
    # [archived branch] if args.coordinated_source_start:
        # [archived branch] if (args.unified_transfer_test or not unified_reference or not args.reference_manager
                # [archived branch] or not args.direct_wbc or not args.motor_armature or args.control_frequency != 500
                # [archived branch] or not args.observed_contact_patches or not args.early_touchdown
                # [archived branch] or args.reference_footsteps != "source"):
            # [archived branch] raise ValueError('--coordinated-source-start requires unified source references, manager, '
                             # [archived branch] '500 Hz direct WBC, armature and observed/early contacts; excludes transfer diagnostics')
    FABRIC_TIMESTEP, PD_TIMESTEP = control_timesteps(
        args.control_frequency, args.physics_frequency, g1_md=args.g1_md)
    # [archived branch] if args.g1_md:
        # [archived branch] print(f"[g1.md] Fabric composition -> constrained dynamics -> integrated MIT torque; "
              # [archived branch] f"{1/FABRIC_TIMESTEP:g} Hz control / {1/PD_TIMESTEP:g} Hz physics "
              # [archived branch] "(simulation time, not a real-time guarantee)")
    # [archived branch] if args.tracking_profile == 'amp-full':
        # Preserve confirmation time when refining the synchronous baseline.
    args.contact_switch_confirm_updates = max(1, round(
        args.contact_switch_confirm_updates * .002 / FABRIC_TIMESTEP))
    # [archived branch] if args.until_motion_end and not args.reference_manager:
        # [archived branch] raise ValueError('--until-motion-end requires --reference-manager')
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
    # [archived branch] if args.source_bridge_duration is not None and (not math.isfinite(args.source_bridge_duration)
            # [archived branch] or args.source_bridge_duration<=0 or not args.source_swing_reference):
        # [archived branch] raise ValueError('--source-bridge-duration requires a positive finite duration and source swing reference')
    # [archived branch] if args.landing_speed_limit is not None and (not (args.reference_manager or args.motion_start_in_motion)
            # [archived branch] or not math.isfinite(args.landing_speed_limit) or args.landing_speed_limit<=0):
        # [archived branch] raise ValueError('--landing-speed-limit requires a reference manager or dynamic start and positive finite speed')
    if (math.isnan(args.weighted_tasks_start) or args.weighted_tasks_start < 0
            or not math.isfinite(args.weighted_tasks_ramp) or args.weighted_tasks_ramp <= 0):
        raise ValueError('Weighted task transition requires nonnegative start and positive ramp')
    # [archived branch] if math.isfinite(args.weighted_tasks_start) and (
            # [archived branch] args.tracking_profile != 'amp-full' or args.g1_com_priority):
        # [archived branch] raise ValueError('Weighted task experiment requires amp-full without a separate COM priority')
    # [archived branch] if math.isnan(args.support_flatten_start) or args.support_flatten_start < 0:
        # [archived branch] raise ValueError('Support flatten start must be nonnegative or infinity')
    # [archived branch] if math.isfinite(args.support_flatten_start) and args.tracking_profile != 'amp-full':
        # [archived branch] raise ValueError('Support flatten experiment requires the amp-full profile')
    if (math.isnan(args.root_position_task_start) or args.root_position_task_start < 0
            or any(not math.isfinite(x) or x <= 0 for x in
                   (args.root_position_task_weight, args.root_position_task_ramp))):
        raise ValueError('Root tracking requires nonnegative start and positive finite weight/ramp')
    # [archived branch] if math.isfinite(args.root_position_task_start) and not (
            # [archived branch] args.tracking_profile == 'amp-full' and args.direct_wbc):
        # [archived branch] raise ValueError('Root position task currently requires the amp-full direct-WBC profile')
    if (math.isnan(args.stance_posture_metric_start) or args.stance_posture_metric_start < 0
            or not math.isfinite(args.stance_posture_metric_scale)
            or not 0 < args.stance_posture_metric_scale <= 1
            or not math.isfinite(args.stance_posture_metric_ramp) or args.stance_posture_metric_ramp <= 0):
        raise ValueError('Stance posture weighting requires nonnegative start, scale in (0,1], and positive ramp')
    # [archived branch] if math.isfinite(args.stance_posture_metric_start) and not (
            # [archived branch] args.motion_start_in_motion and args.task_profile == 'dynamics-aware'):
        # [archived branch] raise ValueError('Stance posture weighting requires dynamic source tracking and physical tasks')
    # [archived branch] if math.isfinite(args.rolling_contact_kinematics_start) and (
            # [archived branch] args.rolling_contact_kinematics_start < 0 or not args.observed_contact_patches
            # [archived branch] or not args.motion_start_in_motion or not args.track_full_motion):
        # [archived branch] raise ValueError('Rolling contact correction requires full dynamic tracking with native contact patches')
    # [archived branch] if math.isnan(args.rolling_contact_kinematics_start) or args.rolling_contact_kinematics_start == -float('inf'):
        # [archived branch] raise ValueError('Rolling contact correction start must be nonnegative or inf')
    if not math.isfinite(args.landing_speed_limit_start) or args.landing_speed_limit_start < 0:
        raise ValueError('--landing-speed-limit-start must be finite and nonnegative')
    # [archived branch] if args.late_momentum_zero_target and args.late_momentum_feedback_damping is None:
        # [archived branch] raise ValueError('Zero momentum regulation requires a positive feedback damping')
    # [archived branch] if args.balance_body_priority and (not args.separate_balance_priority
            # [archived branch] or args.source_pelvis_weight<=0 or args.task_profile!='dynamics-aware'):
        # [archived branch] raise ValueError('--balance-body-priority requires separate balance, physical tasks and a positive source pelvis weight')
    # [archived branch] if args.landing_orientation_priority and not (args.observed_contact_patches and args.transfer_balance_priority):
        # [archived branch] raise ValueError('--landing-orientation-priority requires observed contact patches and transfer balance priority')
    # [archived branch] if args.horizontal_balance_priority and (not args.separate_balance_priority):
        # [archived branch] raise ValueError('--horizontal-balance-priority requires separate balance priority and excludes swing height priority')
    # [archived branch] if args.tracking_servo and not args.direct_wbc:
        # [archived branch] raise ValueError('--tracking-servo requires --direct-wbc')
    # [archived branch] if args.contact_aware_servo and not args.tracking_servo:
        # [archived branch] raise ValueError('--contact-aware-servo requires --tracking-servo')
    # [archived branch] if args.nominal_implicit_servo and not args.contact_aware_servo:
        # [archived branch] raise ValueError('--nominal-implicit-servo requires --contact-aware-servo')
    # [archived branch] if args.balance_priority and args.task_profile != "dynamics-aware":
        # [archived branch] raise ValueError('--balance-priority requires --task-profile dynamics-aware')
    # [archived branch] if args.wrist_hip_avoidance and (args.task_profile != 'dynamics-aware' or args.disable_collision_task):
        # [archived branch] raise ValueError('--wrist-hip-avoidance requires enabled dynamics-aware collision tasks')
    if (not np.isfinite(args.source_arm_roll_offset) or args.source_arm_roll_offset<0
            or (args.source_arm_roll_offset and not (unified_reference and args.motion_pkl is not None))):
        raise ValueError('Arm adaptation requires a nonnegative offset and unified PKL tracking')
    if args.convex_wrist_collision and (not args.wrist_hip_avoidance or str(args.device) != 'cpu'):
        raise ValueError('Convex wrist collision requires wrist avoidance and CPU dynamics')
    # [archived branch] if args.wrist_hip_barrier and not args.wrist_hip_avoidance:
        # [archived branch] raise ValueError('--wrist-hip-barrier requires --wrist-hip-avoidance')
    # [archived branch] if args.wrist_body_avoidance and not args.wrist_hip_barrier:
        # [archived branch] raise ValueError('--wrist-body-avoidance requires --wrist-hip-barrier')
    # [archived branch] if args.transfer_balance_priority and (not (args.unified_transfer_test or args.coordinated_source_start) or args.balance_priority):
        # [archived branch] raise ValueError('--transfer-balance-priority requires dynamic references or a unified transfer, and excludes --balance-priority')
    # [archived branch] if (args.source_swing_position_priority or args.source_swing_pose_priority) and not (
            # [archived branch] unified_reference and (args.coordinated_source_start or args.mocap_first_step_report is not None)):
        # [archived branch] raise ValueError("Source swing priority requires a coordinated unified source reference or bounded first-step probe")
    # [archived branch] if args.source_swing_position_priority and args.source_swing_pose_priority:
        # [archived branch] raise ValueError("Select source swing position or pose priority, not both")
    # [archived branch] if (args.swing_clearance_priority or args.swing_position_priority or args.swing_pose_priority) and (not args.balance_priority):
        # [archived branch] raise ValueError('Swing-foot priority requires a COM priority option')
    if sum((args.swing_clearance_priority,args.swing_position_priority,args.swing_pose_priority))>1:
        raise ValueError('Select one swing-foot priority level')
    # [archived branch] if (args.upper_body_priority or args.whole_body_posture_priority) and (not (args.balance_priority or args.transfer_balance_priority)):
        # [archived branch] raise ValueError('Upper-body priority requires an active COM priority configuration')
    # [archived branch] if args.upper_body_priority and args.whole_body_posture_priority:
        # [archived branch] raise ValueError('Select upper-body-only or whole-body posture priority')
    # [archived branch] if args.centroidal_balance_priority and (not (args.balance_priority or args.transfer_balance_priority)):
        # [archived branch] raise ValueError('Centroidal balance requires an active COM priority')
    # [archived branch] if args.body_orientation_priority and (args.upper_body_priority or args.source_pelvis_weight <= 0 or (not (args.balance_priority or args.transfer_balance_priority))):
        # [archived branch] raise ValueError('Body orientation priority needs a COM priority and positive pelvis weight; excludes upper-body priority')
    if not math.isfinite(args.swing_touchdown_lead) or not 0 <= args.swing_touchdown_lead <= .3:
        raise ValueError('--swing-touchdown-lead must be between 0 and 0.3 seconds')
    if not math.isfinite(args.minimum_step_duration) or args.minimum_step_duration < 0:
        raise ValueError('--minimum-step-duration must be finite and nonnegative')
    # [archived branch] if args.observed_contact_patches:
    if not (unified_reference and (args.motion_start_in_motion or (args.reference_manager and args.early_touchdown))):
        raise ValueError('--observed-contact-patches requires unified references with managed early touchdown or a dynamic source start')
    args.diagnose_contacts = True
    if args.joint_limit_avoidance_weight is None:
        # [archived branch] args.joint_limit_avoidance_weight = 20. if args.task_profile == 'dynamics-aware' else 0.
        args.joint_limit_avoidance_weight = 20.0
    if (not math.isfinite(args.joint_limit_avoidance_weight) or args.joint_limit_avoidance_weight < 0
            or (args.joint_limit_avoidance_weight > 0 and args.task_profile != 'dynamics-aware')):
        raise ValueError('--joint-limit-avoidance-weight requires a finite nonnegative value and physical tasks')
    if args.qp_reuse_workspace and args.qp_solver != 'osqp':
        raise ValueError('--qp-reuse-workspace requires --qp-solver osqp')
    if args.qp_active_set_predictor and args.qp_solver != 'osqp':
        raise ValueError('--qp-active-set-predictor requires --qp-solver osqp')
    # [archived branch] if args.source_momentum_reference and (not args.reference_manager or args.reference_footsteps != 'source'):
        # [archived branch] raise ValueError('--source-momentum-reference requires managed source footsteps')
    # [archived branch] if args.align_takeoff_frame and (not args.reference_manager or args.reference_footsteps != 'source'):
        # [archived branch] raise ValueError('--align-takeoff-frame requires managed source footsteps')
    if not np.isfinite(args.takeoff_lead) or not 0. <= args.takeoff_lead <= .15 or (args.takeoff_lead and not args.align_takeoff_frame):
        raise ValueError('--takeoff-lead needs aligned takeoff and a value between 0 and .15')
    # [archived branch] if args.source_swing_reference and (not args.reference_manager or args.reference_footsteps != 'source'
                                       # [archived branch] or args.motion_interpolation != 'pchip-analytic'):
        # [archived branch] raise ValueError('--source-swing-reference requires managed source footsteps and pchip-analytic')
    # [archived branch] if args.joint_limit_time_constant is not None and (
            # [archived branch] not math.isfinite(args.joint_limit_time_constant) or args.joint_limit_time_constant <= 0
            # [archived branch] or args.motion_pkl is None):
        # [archived branch] raise ValueError('--joint-limit-time-constant requires a PKL and a finite positive value')
    # [archived branch] if args.flat_foot_spawn and (args.motion_pkl is None or args.task_profile != "dynamics-aware"
                                # [archived branch] or args.motion_warmup_duration <= 0):
        # [archived branch] raise ValueError('--flat-foot-spawn requires a PKL, dynamics-aware tasks and positive warmup')
    if args.joint_limit_barrier_bandwidth is not None and (
            args.motion_pkl is None or not math.isfinite(args.joint_limit_barrier_bandwidth)
            or args.joint_limit_barrier_bandwidth <= 0):
        raise ValueError('--joint-limit-barrier-bandwidth requires a PKL and finite positive bandwidth')
    # [archived branch] if args.early_touchdown and not args.reference_manager:
        # [archived branch] raise ValueError('--early-touchdown requires --reference-manager')
    # [archived branch] if args.direct_wbc and (args.task_profile != "dynamics-aware" or args.fixed_joint_test):
        # [archived branch] raise ValueError('--direct-wbc requires dynamics-aware tasks and excludes fixed-joint servo diagnostics')
    if args.contact_stabilization_kp is None:
        # [archived branch] args.contact_stabilization_kp = 20. if args.reference_manager else 0.
        args.contact_stabilization_kp = 0.0
    if args.contact_stabilization_kd is None:
        # [archived branch] args.contact_stabilization_kd = 8. if args.reference_manager else 0.
        args.contact_stabilization_kd = 0.0
    if args.stabilize_foot_rotation is None:
        args.stabilize_foot_rotation = args.reference_manager
    # [archived branch] if args.contact_cycle_test:
        # [archived branch] if args.single_support_test and args.single_support_test != args.contact_cycle_test:
            # [archived branch] raise ValueError("Contact cycle and single-support test must select the same support foot.")
        # [archived branch] args.single_support_test = args.contact_cycle_test
        # [archived branch] args.align_balance_pelvis = True
        # [archived branch] args.require_support_load = True
    # [archived branch] if args.g1_com_priority and not args.g1_md:
        # [archived branch] raise ValueError('--g1-com-priority requires --g1-md')
    if args.diagnose_com and args.motion_pkl is None:
        raise ValueError("--diagnose-com requires --motion-pkl for sampled history output.")
    if args.diagnose_contacts and args.motion_pkl is None:
        raise ValueError("--diagnose-contacts requires --motion-pkl for sampled history output.")
    if args.diagnose_qp_time and (not args.diagnose_com or args.fixed_joint_test or
            any(not math.isfinite(t) or t < 0 or t >= args.duration for t in args.diagnose_qp_time)):
        raise ValueError("QP snapshot times must be finite within duration, require --diagnose-com and exclude --fixed-joint-test.")
    if not 0. <= args.jdot_filter_alpha < 1.:
        raise ValueError("--jdot-filter-alpha must be in [0,1).")
    # [archived branch] if args.weight_transfer_test and (args.motion_pkl is None or args.fixed_joint_test):
        # [archived branch] raise ValueError("--weight-transfer-test requires motion PKL and cannot combine with --fixed-joint-test.")
    # [archived branch] if args.single_support_test and (args.motion_pkl is None or args.fixed_joint_test or args.weight_transfer_test
                                    # [archived branch] or not args.align_balance_pelvis):
        # [archived branch] raise ValueError("Single-support test requires PKL and --align-balance-pelvis; excludes other isolated tests.")
    # [archived branch] if not math.isfinite(args.left_shoulder_roll_offset):
        # [archived branch] raise ValueError("--left-shoulder-roll-offset must be finite.")
    if not math.isfinite(args.servo_bandwidth) or args.servo_bandwidth <= 0:
        raise ValueError("--servo-bandwidth must be finite and positive.")
    # [archived branch] if args.disable_self_collisions:
        # [archived branch] print("DIAGNOSTIC: self-collisions disabled; this run cannot certify motion tracking.")
    # [archived branch] if args.contact_mode is not None:
        # [archived branch] print(
            # [archived branch] "warning: --contact-mode is deprecated and ignored; "
            # [archived branch] "contact mode comes from PKL prediction when a motion is loaded."
        # [archived branch] )
    if args.regularization < 0.0:
        raise ValueError("--regularization must be nonnegative.")
    if args.duration <= 0.0:
        raise ValueError("--duration must be positive.")
    if args.motion_speed <= 0.0:
        raise ValueError("--motion-speed must be positive.")
    if not math.isfinite(args.motion_ramp_duration) or args.motion_ramp_duration < 0:
        raise ValueError("--motion-ramp-duration must be finite and nonnegative.")
    # [archived branch] if args.reference_manager and (args.motion_pkl is None or args.task_profile != "dynamics-aware"
            # [archived branch] or args.fixed_joint_test or args.weight_transfer_test or args.single_support_test
            # [archived branch] or args.allow_flight or args.motion_ramp_duration <= 0):
        # [archived branch] raise ValueError("Reference manager requires a PKL, dynamics-aware profile, positive ramp, and normal non-flight tracking.")
    if args.motion_warmup_duration < 0.0:
        raise ValueError("--motion-warmup-duration must be nonnegative.")
    if not math.isfinite(args.support_transfer_duration) or args.support_transfer_duration <= 0:
        raise ValueError("--support-transfer-duration must be finite and positive.")
    if not math.isfinite(args.cycle_crouch_depth) or not 0 <= args.cycle_crouch_depth <= .08:
        raise ValueError("--cycle-crouch-depth must be between 0 and .08 m.")
    # [archived branch] if args.cycle_crouch_depth and not args.contact_cycle_test:
        # [archived branch] raise ValueError("--cycle-crouch-depth requires --contact-cycle-test.")
    # [archived branch] cycle_preparation = 3. if args.contact_cycle_test and args.cycle_crouch_depth else 0.
    cycle_preparation = 0.0
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
    if args.qp_base_acceleration_weight <= 0.0:
        raise ValueError("--qp-base-acceleration-weight must be positive.")
    if args.maximum_com_tracking_error <= 0.0:
        raise ValueError("--maximum-com-tracking-error must be positive.")
    if not 0.0 < args.minimum_com_height_ratio <= 1.0:
        raise ValueError(
            "--minimum-com-height-ratio must be in the interval (0, 1]."
        )
    if args.maximum_joint_position_rmse <= 0.0:
        raise ValueError("--maximum-joint-position-rmse must be positive.")
    # [archived branch] if args.swing_metric_scale is not None and (not (args.reference_manager or args.motion_start_in_motion)
            # [archived branch] or not math.isfinite(args.swing_metric_scale) or args.swing_metric_scale <= 0):
        # [archived branch] raise ValueError("--swing-metric-scale requires a reference manager or dynamic start and a finite positive value.")
    # [archived branch] if args.posture_metric_scale is not None and (not (args.reference_manager or args.motion_start_in_motion)
            # [archived branch] or not math.isfinite(args.posture_metric_scale) or args.posture_metric_scale <= 0):
        # [archived branch] raise ValueError("--posture-metric-scale requires a reference manager or dynamic start and a finite positive value.")
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
        joint_posture_enabled=True,
    )
    if not fabric.joint_posture_enabled:
        shoulder_guard = None
    # [archived branch] if args.motion_start_in_motion:
    # [archived branch] if args.swing_metric_scale is not None:
        # [archived branch] fabric.fabric_params['feet_pose_attractor']['metric'] = [
            # [archived branch] x*args.swing_metric_scale for x in fabric.fabric_params['feet_pose_attractor']['metric']]
    # [archived branch] if args.posture_metric_scale is not None:
        # [archived branch] fabric.fabric_params['joint_posture_attractor']['metric'] *= args.posture_metric_scale
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
    # [archived branch] if not args.acceleration_and_braking_limits:
    fabric.fabric_params['joint_limits']['active'] = False
    args.limit_predicted_acceleration = False
    if args.collision_obstacles is not None and fabric.dynamics_tasks is None:
        raise ValueError("--collision-obstacles requires --task-profile dynamics-aware.")
    if fabric.dynamics_tasks is not None:
        fabric.dynamics_tasks.set_hand_tracking(args.hand_tracking)
        fabric.dynamics_tasks.collision_enabled = not args.disable_collision_task
        fabric.dynamics_tasks.stable_constraint_rows = args.qp_reuse_workspace
        # [archived branch] if args.wrist_hip_avoidance:
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
        # [archived branch] if args.left_shoulder_roll_offset:
            # [archived branch] motion_trajectory_cpu = offset_left_shoulder_reference(
                # [archived branch] motion_trajectory_cpu, args.left_shoulder_roll_offset)
            # [archived branch] print(f"REFERENCE MODIFIED: left shoulder roll offset={args.left_shoulder_roll_offset:g} rad; PKL unchanged")
        contact_path = args.contacts_pkl or args.motion_pkl
        if args.contacts_pkl is not None:
            contact_q, contact_fps = load_motion_trajectory(contact_path)
            if contact_fps != motion_fps or contact_q.shape != motion_trajectory_cpu.shape or not torch.allclose(contact_q, motion_trajectory_cpu, atol=1e-6, rtol=0):
                raise ValueError('Contact companion poses/fps do not match motion')
        motion_contact_sequence_cpu = load_motion_contact_sequence(contact_path)
        if args.contacts_pkl is not None and motion_contact_sequence_cpu is None:
            raise ValueError('Explicit contact companion has no contact labels')
        # [archived branch] if args.tracking_steps or args.track_full_motion:
        from fabrics_sim.reference.bounded_steps import select_steps, select_dynamic_steps, select_all_dynamic_steps
        from fabrics_sim.reference.manager import contact_sequence_modes
        if motion_contact_sequence_cpu is None:
            raise ValueError('Strict bounded steps require contact labels; supply --contacts-pkl')
        # [archived branch] selector = select_dynamic_steps if args.motion_start_in_motion else select_steps
        selector = select_dynamic_steps
        modes = contact_sequence_modes(motion_contact_sequence_cpu.numpy(), len(motion_trajectory_cpu))
        # [archived branch] bounded_events = (select_all_dynamic_steps(modes) if args.track_full_motion
                          # [archived branch] else selector(modes, args.tracking_steps))
        bounded_events = select_all_dynamic_steps(modes)
        if not bounded_events:
            raise ValueError('The contact plan contains no complete source landings')
        args.original_source_frame_count = len(motion_trajectory_cpu)
        # [archived branch] if not args.motion_start_in_motion:
            # [archived branch] end = bounded_events[-1]['end_frame']
            # [archived branch] motion_trajectory_cpu = motion_trajectory_cpu[:end+1]
            # [archived branch] motion_contact_sequence_cpu = motion_contact_sequence_cpu[:end+1]
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
    # [archived branch] if args.source_arm_roll_offset:
        # [archived branch] import xml.etree.ElementTree as ET
        # [archived branch] from fabrics_sim.reference.arm_clearance import arm_clearance_reference
        # [archived branch] limits={node.attrib['name']:node.find('limit') for node in ET.parse(fabric.urdf_path).getroot().findall('joint')}
        # [archived branch] control_motion_trajectory = arm_clearance_reference(motion_trajectory,G1_JOINT_NAMES,
            # [archived branch] args.source_arm_roll_offset,
            # [archived branch] q.new_tensor([float(limits[name].attrib['lower']) for name in G1_JOINT_NAMES]),
            # [archived branch] q.new_tensor([float(limits[name].attrib['upper']) for name in G1_JOINT_NAMES]))
        # [archived branch] q = control_motion_trajectory[motion_start_frame:motion_start_frame+1].clone()
        # [archived branch] print(f'Explicit arm clearance adaptation: {args.source_arm_roll_offset:.4f} rad; original PKL retained for audit',flush=True)
    # [archived branch] if args.flat_foot_spawn:
        # [archived branch] from fabrics_sim.reference.initial_stance import flat_foot_spawn
        # [archived branch] source_spawn=q.clone()
        # [archived branch] q=flat_foot_spawn(fabric,q,G1_JOINT_NAMES)
        # [archived branch] print(f"Flat-foot spawn: root height change={float(q[0,2]-source_spawn[0,2]):.6f} m, "
              # [archived branch] f"max joint change={float((q[:,7:]-source_spawn[:,7:]).abs().max()):.6f} rad; reference unchanged",flush=True)
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
        # [archived branch] if args.mocap_pipeline:
            # [archived branch] from fabrics_sim.reference.preprocessing import prepare_reference
            # [archived branch] motion_sampler, preparation = prepare_reference(control_motion_trajectory, motion_fps,
                # [archived branch] G1_JOINT_NAMES, fabric.urdf_path,
                # [archived branch] args.reference_acceleration_cap if args.acceleration_and_braking_limits else None, args.motion_speed,
                # [archived branch] method="pchip" if args.tracking_steps else "bspline")
            # [archived branch] args.motion_speed = preparation['playback_speed']
            # [archived branch] preparation.update(motion=str(args.motion_pkl), contact_source=str(args.contacts_pkl or args.motion_pkl),
                               # [archived branch] contact_labels_present=motion_contact_sequence is not None,
                               # [archived branch] reference_policy='direct_full_body_pkl', reduced_model_preview=False,
                               # [archived branch] capture_point_gate=False)
            # [archived branch] args.output_path.parent.mkdir(parents=True, exist_ok=True)
            # [archived branch] args.output_path.with_suffix('.preparation.json').write_text(json.dumps(preparation, indent=2)+'\n')
            # [archived branch] print('Online reference prepared: '+json.dumps(preparation), flush=True)
        # [archived branch] else:
        # [archived branch] motion_sampler = MotionInterpolator(control_motion_trajectory,
            # [archived branch] method='bspline' if args.motion_interpolation == 'bspline-analytic' else 'pchip')
        motion_sampler = MotionInterpolator(control_motion_trajectory, method='pchip')

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
        # [archived branch] return (source_contact_step(pose) if args.source_swing_reference else source_grounded_step(pose))
        return source_grounded_step(pose)


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
    # [archived branch] if args.observed_contact_patches:
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
    # [archived branch] if args.native_contact_plan_start<100.:
        # [archived branch] if motion_contact_sequence is None or motion_planned_modes is None or not math.isfinite(args.native_contact_plan_start) or args.native_contact_plan_start<0:
            # [archived branch] raise ValueError('Native contact plan requires finite source time and an explicit contact sequence')
        # [archived branch] first_native_frame=min(len(motion_planned_modes),int(round(args.native_contact_plan_start*motion_fps)))
        # [archived branch] supplied_modes=motion_contact_sequence[:,0].long()+2*motion_contact_sequence[:,1].long()
        # [archived branch] previous_modes=motion_planned_modes.clone()
        # [archived branch] motion_planned_modes[first_native_frame:]=supplied_modes[first_native_frame:]
        # [archived branch] args.output_path.with_suffix('.native_contact_plan.json').write_text(json.dumps(dict(
            # [archived branch] first_source_frame=first_native_frame,
            # [archived branch] changed_source_frames=torch.nonzero(previous_modes!=motion_planned_modes).flatten().cpu().tolist(),
            # [archived branch] measured_support_gates_unchanged=True),indent=2)+'\n')
    # [archived branch] if args.reference_manager:
        # [archived branch] from fabrics_sim.reference.manager import ReferenceManager, infer_contact_plan, infer_corner_contact_plan, contact_sequence_modes
        # [archived branch] fabric.dynamics_tasks.managed_walking = True
        # [archived branch] fabric.dynamics_tasks.source_pelvis_metric_scale = args.source_pelvis_weight
        # [archived branch] if args.swing_metric_scale is not None:
            # [archived branch] fabric.dynamics_tasks.config['managed_walking']['swing_metric_scale'] = args.swing_metric_scale
        # [archived branch] if args.posture_metric_scale is not None:
            # [archived branch] fabric.dynamics_tasks.config['managed_walking']['posture_metric_scale'] = args.posture_metric_scale
        # [archived branch] with torch.no_grad():
            # [archived branch] reference_feet_all = torch.cat([reference_feet_transforms(fabric, row[None])
                                           # [archived branch] for row in motion_trajectory], dim=0)
            # [archived branch] corners = q.new_tensor([[-.05,-.025,-.035],[-.05,.025,-.035],[.12,-.025,-.035],[.12,.025,-.035]])
            # [archived branch] rotations = _quat_xyzw_to_matrix(reference_feet_all[...,3:7])
            # [archived branch] sole_points = reference_feet_all[...,:3].unsqueeze(-2)+(rotations.unsqueeze(-3)@corners[...,None]).squeeze(-1)
            # [archived branch] heights = sole_points[...,2].amin(-1).cpu().numpy()
            # [archived branch] centers = reference_feet_all[...,:3].cpu().numpy()
        # [archived branch] if args.source_swing_reference:
            # [archived branch] from fabrics_sim.reference.interpolation import MotionInterpolator
            # [archived branch] source_foot_samplers=[MotionInterpolator(reference_feet_all[:,i],pose_dimension=7) for i in range(2)]
            # [archived branch] source_xyz_coefficients=[None if sampler.coefficients is None else
                                     # [archived branch] sampler.coefficients[:,:,:3].detach().cpu().double().numpy()
                                     # [archived branch] for sampler in source_foot_samplers]
        # [archived branch] speeds = np.linalg.norm(np.gradient(centers,1./motion_fps,axis=0),axis=-1)
        # [archived branch] if args.reference_footsteps == "source":
            # [archived branch] candidate_modes,ambiguous,speeds=infer_corner_contact_plan(sole_points.cpu().numpy(),motion_fps)
            # [archived branch] contact_plan_method='sole_corner'
        # [archived branch] else:
            # [archived branch] candidate_modes, ambiguous = infer_contact_plan(heights,speeds,motion_fps)
            # [archived branch] contact_plan_method='ankle'
        # [archived branch] contact_geometry_method = contact_plan_method
        # [archived branch] if motion_contact_sequence is not None:
            # [archived branch] candidate_modes = contact_sequence_modes(
                # [archived branch] motion_contact_sequence.detach().cpu().numpy(), len(motion_trajectory))
            # [archived branch] contact_plan_method = 'pkl_contact_sequence'
        # [archived branch] if bounded_events is not None:
            # [archived branch] from fabrics_sim.reference.bounded_steps import MeasuredSteps
            # [archived branch] goals = np.stack([_to_numpy(reference_feet_all[e['end_frame'],1-e['support'],:3])
                              # [archived branch] for e in bounded_events])
            # [archived branch] measured_steps = MeasuredSteps(bounded_events, goals)
            # [archived branch] args.output_path.parent.mkdir(parents=True, exist_ok=True)
            # [archived branch] args.output_path.with_suffix('.steps_plan.json').write_text(json.dumps(dict(
                # [archived branch] events=bounded_events, landing_positions_world=goals.tolist(),
                # [archived branch] interpolation='PCHIP passes original poses; C1, not C2',
                # [archived branch] original_source_frame_count=args.original_source_frame_count,
                # [archived branch] source_positions_modified=False), indent=2)+'\n')
        # [archived branch] motion_planned_modes = torch.as_tensor(candidate_modes,device=device)
        # [archived branch] reference_manager = ReferenceManager(candidate_modes,motion_fps,motion_start_frame,
                                             # [archived branch] args.motion_speed,args.motion_ramp_duration,
                                             # [archived branch] settle_duration=0.3,
                                             # [archived branch] early_touchdown=args.early_touchdown,
                                             # [archived branch] settle_on_incoming=False,
                                             # [archived branch] align_takeoff_frame=args.align_takeoff_frame,
                                             # [archived branch] continuous_source_clock=False,
                                             # [archived branch] takeoff_lead=args.takeoff_lead)
        # [archived branch] from fabrics_sim.reference.completion import MotionCompletion
        # [archived branch] motion_completion = MotionCompletion(len(motion_trajectory)-1)
        # [archived branch] args.output_path.parent.mkdir(parents=True,exist_ok=True)
        # [archived branch] np.savez_compressed(args.output_path.with_suffix(".contact_plan.npz"), modes=candidate_modes,
                            # [archived branch] sole_height=heights, foot_speed=speeds, ambiguous=ambiguous, fps=motion_fps,
                            # [archived branch] method=contact_plan_method, geometry_method=contact_geometry_method)
        # [archived branch] print(f"Experimental contact plan: method={contact_plan_method}, "
              # [archived branch] f"modes={np.bincount(candidate_modes,minlength=4).tolist()}, "
              # [archived branch] f"geometry ambiguous frames={int(ambiguous.sum())}/{len(ambiguous)}",flush=True)
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
    # [archived branch] if args.motion_start_in_motion:
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

    # [archived branch] (
        # [archived branch] mass_np,
        # [archived branch] bias_np,
        # [archived branch] left_j_np,
        # [archived branch] right_j_np,
        # [archived branch] isaac_joint_names,
        # [archived branch] simulation_app,
        # [archived branch] world,
        # [archived branch] robot,
        # [archived branch] foot_contact_view,
    # [archived branch] ) = (
        # [archived branch] initialize_isaac_dynamics(
            # [archived branch] q,
            # [archived branch] qd,
            # [archived branch] headless=not args.show_isaac_window,
            # [archived branch] enable_self_collisions=False if args.disable_self_collisions else None,
            # [archived branch] report_contacts=args.report_self_contacts,
            # [archived branch] physics_iterations=args.physics_iterations,
            # [archived branch] diagnose_contacts=args.diagnose_contacts,
            # [archived branch] motor_armature=args.motor_armature,
            # [archived branch] prime_source_contacts=args.motion_start_in_motion,
        # [archived branch] )
    # [archived branch] )
    (mass_np, bias_np, left_j_np, right_j_np, isaac_joint_names, simulation_app, world, robot, foot_contact_view) = initialize_isaac_dynamics(
        q,
        qd,
        headless=not args.show_isaac_window,
        enable_self_collisions=None,
        report_contacts=args.report_self_contacts,
        physics_iterations=args.physics_iterations,
        diagnose_contacts=args.diagnose_contacts,
        motor_armature=args.motor_armature,
        prime_source_contacts=args.motion_start_in_motion,
    )
    # [archived branch] if args.motion_start_in_motion:
    q, qd, _, _, _, _ = read_live_isaac_snapshot(robot, isaac_joint_names, device, q.dtype)
    qdd_fabric = fabric.compute_acceleration(q, qd)
    mass = torch.as_tensor(mass_np, device=device, dtype=q.dtype)
    if fabric.dynamics_tasks is not None:
        armature_order = [isaac_joint_names.index(name) for name in G1_JOINT_NAMES]
        fabric.dynamics_tasks.model.validate_mass_matrix(
            q, mass, joint_armature=_to_numpy(robot.get_armatures())[:, armature_order])
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
        # [archived branch] if args.landing_speed_limit is not None:
            # [archived branch] print(f'Landing speed envelope: {args.landing_speed_limit:g} m/s; '
                  # [archived branch] f'{len(foot_landing_geometry[0][0])} collision points per foot',flush=True)
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
        # [archived branch] if args.track_full_motion:
        from fabrics_sim.reference.swing_clearance import feasible_swing_clearance_reference
        control_motion_trajectory, clearance_feasibility = feasible_swing_clearance_reference(
            fabric, control_motion_trajectory, G1_JOINT_NAMES, bounded_events,
            args.swing_midpoint_clearance, collision_geometry=geometry)
        # [archived branch] else:
            # [archived branch] control_motion_trajectory = swing_clearance_reference(
                # [archived branch] fabric, control_motion_trajectory, G1_JOINT_NAMES, bounded_events,
                # [archived branch] args.swing_midpoint_clearance, collision_geometry=geometry)
        # [archived branch] if args.tracking_profile is not None:
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
        # [archived branch] motion_sampler = MotionInterpolator(control_motion_trajectory,
            # [archived branch] method='bspline' if args.motion_interpolation == 'bspline-analytic' else 'pchip')
        motion_sampler = MotionInterpolator(control_motion_trajectory, method='pchip')
        if args.smooth_stride:
            from blended_spline import BlendedSplineReference
            motion_sampler=BlendedSplineReference(motion_sampler,control_motion_trajectory,stride=args.smooth_stride,
                start_frame=args.smooth_start*motion_fps,ramp_frames=args.smooth_ramp*motion_fps)
        if args.late_reference_smooth_stride:
            from late_tube_smoothing import transition_tube_spline
            motion_sampler=transition_tube_spline(motion_sampler,control_motion_trajectory,
                stride=args.late_reference_smooth_stride,start_frame=args.late_reference_smooth_start*motion_fps,
                ramp_frames=args.late_reference_smooth_ramp*motion_fps)
        # [archived branch] if args.joint_correction_overlay is not None:
            # [archived branch] from joint_correction_overlay import JointCorrectionOverlay
            # [archived branch] motion_sampler=JointCorrectionOverlay(motion_sampler,control_motion_trajectory,motion_trajectory,args.joint_correction_overlay)
        # [archived branch] if args.secondary_joint_correction_overlay is not None:
            # [archived branch] import hashlib
            # [archived branch] if args.joint_correction_overlay is None:raise ValueError('Secondary overlay requires its recorded parent')
            # [archived branch] with np.load(args.secondary_joint_correction_overlay) as correction:
                # [archived branch] if correction['parent_overlay_sha256'].item()!=hashlib.sha256(args.joint_correction_overlay.read_bytes()).hexdigest():
                    # [archived branch] raise ValueError('Secondary correction parent mismatch')
            # [archived branch] motion_sampler=JointCorrectionOverlay(motion_sampler,control_motion_trajectory,motion_trajectory,args.secondary_joint_correction_overlay)
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
    # [archived branch] if args.prepared_contact_plan is not None:
        # [archived branch] if not args.direct_wbc or not args.motion_start_in_motion or args.reference_pose_candidate is None or reference_manager is not None:
            # [archived branch] raise ValueError('Prepared contact intentions require direct source WBC with an explicit prepared pose reference')
        # [archived branch] from prepared_contact_plan import load_prepared_contact_plan
        # [archived branch] prepared_modes,contact_report=load_prepared_contact_plan(args.prepared_contact_plan,args.reference_pose_candidate,_to_numpy(motion_planned_modes))
        # [archived branch] motion_planned_modes=motion_planned_modes.new_tensor(prepared_modes)
        # [archived branch] args.output_path.with_suffix('.prepared_contacts.json').write_text(json.dumps(contact_report,indent=2)+'\n')
        # [archived branch] print('Prepared contact intentions: '+json.dumps(contact_report),flush=True)
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
    # [archived branch] measured_contact_detector = BatchedFootContactDetector(
        # [archived branch] num_envs=1,
        # [archived branch] device=device,
        # [archived branch] config=ContactDetectorConfig(
            # [archived branch] force_on=20.0,
            # [archived branch] force_off=10.0,
            # [archived branch] confirm_steps=max(1,round(args.contact_switch_confirm_updates*.02/FABRIC_TIMESTEP)) if args.reference_manager else args.contact_switch_confirm_updates,
            # [archived branch] impact_force_on=100. if args.motion_start_in_motion else None,
        # [archived branch] ),
    # [archived branch] )
    measured_contact_detector = BatchedFootContactDetector(num_envs=1, device=device, config=ContactDetectorConfig(force_on=20.0, force_off=10.0, confirm_steps=args.contact_switch_confirm_updates, impact_force_on=100.0))
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
    # [archived branch] if args.motion_start_in_motion:
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
    # [archived branch] contact_cycle = ContactCycle() if args.contact_cycle_test else None
    contact_cycle = None
    single_initial_feet = contact_anchors.clone()
    # The first finite-difference Jdot*v sample is intentionally zero because
    # there is no previous live Jacobian yet.
    zero_bias = torch.zeros((1, 6), device=device, dtype=q.dtype)
    output = projector.step(
        acc_fabric=qdd_fabric,
        mass_matrix=mass,
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
        qdd_fabric = fabric.compute_acceleration(q, qd)
    # [archived branch] qdd_dc = qdd_fabric if args.g1_md else output.acc_dc
    qdd_dc = output.acc_dc
    contact_jacobian = torch.cat((left_j, right_j), dim=1)
    jdot_qdot = torch.cat((zero_bias, zero_bias), dim=1)
    qp_acceleration_weight = torch.ones(
        35, device=device, dtype=q.dtype
    )
    qp_acceleration_weight[:6] = args.qp_base_acceleration_weight
    qp_type = G1DynamicsAccelerationQP
    # [archived branch] if args.tracking_profile is not None:
    from unload_qp import G1DynamicsAccelerationQP as qp_type
    # [archived branch] acceleration_qp = qp_type(
        # [archived branch] solver_backend=args.qp_solver,
        # [archived branch] heel_length=.05 if args.use_sole_wrench_geometry else .06,
        # [archived branch] half_width=.025 if args.use_sole_wrench_geometry else .04,
        # [archived branch] max_iter=2000,
        # [archived branch] rho=100.0,
        # A 1e-6 wrench weight leaves the acceleration/wrench Hessian
        # poorly conditioned (roughly six orders of magnitude apart) and
        # repeatedly stalls ADMM at its iteration limit during gait contact
        # transitions.  This remains small relative to the acceleration
        # tracking term, while providing enough curvature for a stable solve.
        # [archived branch] wrench_weight=1.0e-4,
        # [archived branch] acc_weight=qp_acceleration_weight,
        # [archived branch] validation_tolerance=args.qp_validation_tolerance,
        # [archived branch] friction_validation_tolerance=1.0,
        # [archived branch] effort_validation_tolerance=1.0,
        # [archived branch] box_validation_tolerance=1.0,
    # [archived branch] )
    acceleration_qp = qp_type(
        solver_backend=args.qp_solver,
        heel_length=0.05,
        half_width=0.025,
        max_iter=2000,
        rho=100.0,
        wrench_weight=0.0001,
        acc_weight=qp_acceleration_weight,
        validation_tolerance=args.qp_validation_tolerance,
        friction_validation_tolerance=1.0,
        effort_validation_tolerance=1.0,
        box_validation_tolerance=1.0,
    )
    if args.qp_reuse_workspace:
        acceleration_qp.solver.reuse_workspace = True
    if args.qp_active_set_predictor:
        acceleration_qp.solver.active_set_predictor = True
    acceleration_limits = (
        args.qp_acceleration_limit_scale
        * fabric.accel_limits.to(device=device, dtype=q.dtype)
    )
    # [archived branch] if not args.acceleration_and_braking_limits:
    acceleration_limits = torch.full_like(acceleration_limits, float('inf'))
    import xml.etree.ElementTree as ET
    urdf_joint_limits = {node.attrib['name']:node.find('limit')
                         for node in ET.parse(fabric.urdf_path).getroot().findall('joint')}
    joint_velocity_limits = q.new_tensor([float(urdf_joint_limits[name].attrib['velocity'])
                                         for name in G1_JOINT_NAMES])
    # nominal_kp, nominal_kd = unitree_g1_pd_gains(device)
    # kp_batch, kd_batch = explicit_pd_gains(
        # mass, nominal_kp, nominal_kd, PD_TIMESTEP, bandwidth=args.servo_bandwidth
    # )
    # kp, kd = kp_batch[0], kd_batch[0]
    # [archived branch] if args.direct_wbc and not args.tracking_servo:
        # [archived branch] kp = torch.zeros_like(kp)
    # No actuator-level PD in the synchronous direct-effort path.
    kp = torch.zeros(29, device=device, dtype=q.dtype)
    kd = torch.zeros_like(kp)
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
    # [archived branch] initial_friction_G, initial_friction_b = (
        # [archived branch] build_single_support_wrench_constraints(
            # [archived branch] contact_mode=output.contact_mode,
            # [archived branch] foot_rotation_world=_quat_xyzw_to_matrix(
                # [archived branch] initial_feet[..., 3:7]
            # [archived branch] ),
            # [archived branch] friction_coefficient=acceleration_qp.friction_coefficient,
            # [archived branch] yaw_friction_coefficient=(
                # [archived branch] acceleration_qp.yaw_friction_coefficient
            # [archived branch] ),
            # [archived branch] toe_length=acceleration_qp.toe_length,
            # [archived branch] heel_length=acceleration_qp.heel_length,
            # [archived branch] half_width=acceleration_qp.half_width,
            # [archived branch] sole_depth=.035 if args.use_sole_wrench_geometry else 0.,
            # [archived branch] coupled_yaw=args.reference_manager,
            # [archived branch] maximum_normal_force=(
                # [archived branch] acceleration_qp.maximum_normal_force
            # [archived branch] ),
        # [archived branch] )
    # [archived branch] )
    (initial_friction_G, initial_friction_b) = build_single_support_wrench_constraints(
        contact_mode=output.contact_mode,
        foot_rotation_world=_quat_xyzw_to_matrix(initial_feet[..., 3:7]),
        friction_coefficient=acceleration_qp.friction_coefficient,
        yaw_friction_coefficient=acceleration_qp.yaw_friction_coefficient,
        toe_length=acceleration_qp.toe_length,
        heel_length=acceleration_qp.heel_length,
        half_width=acceleration_qp.half_width,
        sole_depth=0.035,
        coupled_yaw=args.reference_manager,
        maximum_normal_force=acceleration_qp.maximum_normal_force,
    )
    # Simulator limits are in Isaac DOF order; all controller tasks use G1 order.
    limit_order = [isaac_joint_names.index(name) for name in G1_JOINT_NAMES]
    joint_position_limits = torch.as_tensor(
        _to_numpy(robot.get_dof_limits()), device=device, dtype=q.dtype
    )[:, limit_order]
    initial_physical_lower = torch.full_like(qd, -float('inf'))
    initial_physical_upper = torch.full_like(qd, float('inf'))
    # [archived branch] if args.joint_position_limits or args.acceleration_and_braking_limits:
    initial_physical_lower[:, 6:], initial_physical_upper[:, 6:] = measured_joint_acceleration_bounds(
        q[:, 7:], qd[:, 6:], joint_position_limits[:, :, 0], joint_position_limits[:, :, 1],
        FABRIC_TIMESTEP, barrier_bandwidth=args.joint_limit_barrier_bandwidth,
        impact_recovery=args.joint_limit_impact_recovery)
    if args.joint_limit_avoidance_weight > 0:
        fabric.dynamics_tasks.last_requests.append(joint_limit_avoidance_request(
            q[:, 7:], qd[:, 6:], joint_position_limits[:, :, 0], joint_position_limits[:, :, 1],
            args.joint_limit_avoidance_weight))
    composed_policy = None
    # [archived branch] if args.g1_md:
        # [archived branch] from fabrics_sim.solver.fabric_mit import compose_task_policy
        # [archived branch] qdd_dc, composed_policy = compose_task_policy(
            # [archived branch] fabric.dynamics_tasks.last_requests, qd,
            # [archived branch] fabric.dynamics_tasks.config["regularization"])
    # [archived branch] qp_output = acceleration_qp.solve(
        # [archived branch] acc_reference=qdd_dc,
        # [archived branch] mass_matrix=mass,
        # [archived branch] bias_force=bias_force,
        # [archived branch] contact_jacobian=contact_jacobian,
        # [archived branch] jdot_qdot=jdot_qdot,
        # [archived branch] contact_mode=output.contact_mode,
        # [archived branch] acc_lower=-acceleration_limits,
        # [archived branch] acc_upper=acceleration_limits,
        # [archived branch] predicted_acc_lower=initial_physical_lower,
        # [archived branch] predicted_acc_upper=initial_physical_upper,
        # [archived branch] effort_limits=qp_effort_limits,
        # [archived branch] known_joint_torque=None,
        # [archived branch] acceleration_constraints=joint_velocity_constraints(qd[:,6:],joint_velocity_limits,FABRIC_TIMESTEP)
            # [archived branch] if args.joint_velocity_limits else None,
        # [archived branch] friction_G=initial_friction_G,
        # [archived branch] friction_b=initial_friction_b,
        # [archived branch] task_requests=([composed_policy] if args.g1_md else
                    # [archived branch] None if fabric.dynamics_tasks is None else fabric.dynamics_tasks.last_requests),
        # [archived branch] task_tracks_predicted=not args.task_track_nominal,
        # [archived branch] priority_tasks=('center_of_mass',) if args.g1_com_priority else (),
        # [archived branch] priority_requests=[r for r in fabric.dynamics_tasks.last_requests
                           # [archived branch] if r.name == 'center_of_mass'] if args.g1_com_priority else None,
        # [archived branch] predicted_acc_limit=acceleration_limits if args.limit_predicted_acceleration else None,
    # [archived branch] )
    qp_output = acceleration_qp.solve(
        acc_reference=qdd_dc,
        mass_matrix=mass,
        bias_force=bias_force,
        contact_jacobian=contact_jacobian,
        jdot_qdot=jdot_qdot,
        contact_mode=output.contact_mode,
        acc_lower=-acceleration_limits,
        acc_upper=acceleration_limits,
        predicted_acc_lower=initial_physical_lower,
        predicted_acc_upper=initial_physical_upper,
        effort_limits=qp_effort_limits,
        known_joint_torque=None,
        acceleration_constraints=joint_velocity_constraints(qd[:, 6:], joint_velocity_limits, FABRIC_TIMESTEP),
        friction_G=initial_friction_G,
        friction_b=initial_friction_b,
        task_requests=None if fabric.dynamics_tasks is None else fabric.dynamics_tasks.last_requests,
        task_tracks_predicted=not args.task_track_nominal,
        priority_tasks=(),
        priority_requests=None,
        predicted_acc_limit=None,
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
    # initial_joint_feedforward = joint_torque_feedforward.clone()
    # Initial measured state for one-step prediction diagnostics.
    # The loop refreshes these seeds from measurements at every QP tick.
    joint_velocity = qd[:, 6:35]
    joint_velocity_prediction = joint_velocity.clone()
    joint_position = q[:, 7:36]
    joint_position_prediction = joint_position.clone()
    initial_joint_position = joint_position_prediction.clone()
    initial_joint_velocity = joint_velocity_prediction.clone()

    # Initial direct QP effort; zero PD fields retain the recording schema.
    # [archived branch] joint_torque_pd = (
        # [archived branch] torch.zeros_like(joint_velocity) if args.tracking_servo else -kd*joint_velocity if args.direct_wbc else
        # [archived branch] kp * (joint_position_desired - joint_position)
        # [archived branch] + kd * (joint_velocity_desired - joint_velocity)
    # [archived branch] )
    joint_torque_pd = torch.zeros_like(joint_velocity)
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
    # position_desired_isaac = joint_position_desired[
        # :, fabric_to_isaac_joint
    # ]
    # velocity_desired_isaac = joint_velocity_desired[
        # :, fabric_to_isaac_joint
    # ]
    kp_isaac = kp[fabric_to_isaac_joint]
    kd_isaac = kd[fabric_to_isaac_joint]
    effort_limits_isaac = effort_limits[fabric_to_isaac_joint]
    # [archived branch] if motion_trajectory is not None and args.left_shoulder_roll_offset:
        # [archived branch] shoulder = G1_JOINT_NAMES.index("left_shoulder_roll_joint")
        # [archived branch] values = motion_trajectory[:, 7 + shoulder]
        # [archived branch] if ((values < joint_position_limits[0, shoulder, 0]) |
                # [archived branch] (values > joint_position_limits[0, shoulder, 1])).any():
            # [archived branch] raise ValueError("Shoulder clearance offset exceeds simulator joint limits in the reference trajectory.")
    # Conservative command-speed ceiling for this tracking experiment.
    joint_command_speed_limit = torch.full_like(joint_position_prediction, 4.0)
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
    print(f"actuation: direct QP effort at {1./PD_TIMESTEP:g} Hz; actuator PD=0; PhysX drive gains=0")
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
    # [archived branch] mocap_transition = MocapContactTransition(fabric, G1_JOINT_NAMES) if args.g1_md and args.mocap_pipeline else None
    mocap_transition = None
    from fabrics_sim.reference.contact_transfer import TransferStartGate
    transfer_start_gate = TransferStartGate()
    source_step_rollout = None
    probe_unload_report = None
    source_step_start = None
    support_switch = None
    source_step_number = 0
    direct_unload_event = None
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
    # [archived branch] if math.isfinite(args.torque_continuity_start):
        # [archived branch] from g1_torque_continuity import TorqueContinuitySolver
        # [archived branch] torque_continuity_solver = TorqueContinuitySolver(acceleration_qp.solver)
        # [archived branch] acceleration_qp.solver = torque_continuity_solver
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

            # [archived branch] com_replan_diagnostic = (dict(centroidal_replan_attempted=False,
                # [archived branch] centroidal_replan_accepted=False,centroidal_replan_reason='')
                # [archived branch] if None is not None else {})
            com_replan_diagnostic = {}

            control_time = control_updates * FABRIC_TIMESTEP
            elapsed_motion_time = control_time
            if motion_trajectory is not None:
                warming_up = args.fixed_joint_test or args.weight_transfer_test or bool(args.single_support_test) or control_time < args.motion_warmup_duration
                wall_motion_time = max(0.0, control_time - args.motion_warmup_duration)
                elapsed_motion_time = ramped_motion_time(wall_motion_time, args.motion_ramp_duration)
                previous_motion_time = ramped_motion_time(wall_motion_time-FABRIC_TIMESTEP, args.motion_ramp_duration)
                next_motion_time = ramped_motion_time(wall_motion_time+FABRIC_TIMESTEP, args.motion_ramp_duration)
                clock_rate, clock_acceleration = 1., 0.
                # [archived branch] if args.tracking_profile is not None:
                from smooth_clock import smooth_clock
                # [archived branch] if args.contact_pause_end is not None:
                    # [archived branch] from landing_pause_clock import LandingPauseClock
                    # [archived branch] if args.contact_pause_early_resume:
                        # [archived branch] from early_resume_clock import EarlyResumeLandingClock as LandingPauseClock
                    # [archived branch] if 'source_contact_pause' not in locals():
                        # [archived branch] source_contact_pause=LandingPauseClock(args.contact_pause_start,args.contact_pause_end,args.contact_pause_resume,args.contact_pause_landings)
                    # [archived branch] speed_clock=smooth_clock
                    # [archived branch] input_phase=speed_clock(wall_motion_time,args.clock_transition_start,args.clock_transition_ramp,args.clock_final_speed,args.clock_second_start,args.clock_second_ramp,args.clock_second_speed,args.clock_third_start,args.clock_third_ramp,args.clock_third_speed)[0]
                    # [archived branch] source_contact_pause.update(input_phase,len(measured_steps.landings) if measured_steps is not None else 0)
                    # [archived branch] def smooth_clock(time,*parameters):
                        # [archived branch] phase,rate,acceleration=speed_clock(time,*parameters)
                        # [archived branch] out,phase_rate,phase_acceleration=source_contact_pause.sample(phase)
                        # [archived branch] return out,phase_rate*rate,phase_acceleration*rate**2+phase_rate*acceleration
                (elapsed_motion_time, clock_rate, clock_acceleration) = smooth_clock(
                    wall_motion_time,
                    args.clock_transition_start,
                    args.clock_transition_ramp,
                    args.clock_final_speed,
                    args.clock_second_start,
                    args.clock_second_ramp,
                    args.clock_second_speed,
                    args.clock_third_start,
                    args.clock_third_ramp,
                    args.clock_third_speed,
                )
                previous_motion_time=smooth_clock(wall_motion_time-FABRIC_TIMESTEP,args.clock_transition_start,args.clock_transition_ramp,args.clock_final_speed,args.clock_second_start,args.clock_second_ramp,args.clock_second_speed,args.clock_third_start,args.clock_third_ramp,args.clock_third_speed)[0]
                next_motion_time=smooth_clock(wall_motion_time+FABRIC_TIMESTEP,args.clock_transition_start,args.clock_transition_ramp,args.clock_final_speed,args.clock_second_start,args.clock_second_ramp,args.clock_second_speed,args.clock_third_start,args.clock_third_ramp,args.clock_third_speed)[0]
                # [archived branch] if args.fixed_joint_test or args.weight_transfer_test or args.single_support_test:
                    # [archived branch] elapsed_motion_time = 0.0
                phase_held = False
                pending_phase_mode = None
                preview_release_ready = True
                preview_release_reason = ""
                frame_position = min(
                    motion_start_frame
                    + elapsed_motion_time * motion_fps * args.motion_speed,
                    motion_trajectory.shape[0] - 1,
                )
                # [archived branch] if reference_manager is not None:
                    # [archived branch] from fabrics_sim.reference.balance import support_balance_ready
                    # [archived branch] observed_feet = reference_feet_transforms(fabric,q)
                    # [archived branch] observed_rotations = _quat_xyzw_to_matrix(observed_feet[...,3:])
                    # [archived branch] observed_corners = observed_feet[...,:3].unsqueeze(-2)+(observed_rotations.unsqueeze(-3)@corners[...,None]).squeeze(-1)
                    # [archived branch] sole_clearance = observed_corners[...,2].amin(-1)[0].detach().cpu().numpy()
                    # [archived branch] actual_com,actual_com_jac = fabric.center_of_mass_taskmap._position_and_jacobian(q)
                    # [archived branch] actual_com_velocity=(actual_com_jac@qd[...,None]).squeeze(-1)
                    # No capture-point/LIPM gate in direct mocap tracking.
                    # None omits only the optional balance heuristic; the manager
                    # still checks measured support and unloading before release.
                    # [archived branch] balance_ready = None if args.mocap_pipeline else support_balance_ready(actual_com[0].detach().cpu().numpy(), actual_com_velocity[0].detach().cpu().numpy(), observed_feet[0].detach().cpu().numpy(), max_speed=0.06)
                    # [archived branch] if mocap_transition is not None and reference_manager.mode == 3:
                        # [archived branch] balance_ready = [False, False]
                        # [archived branch] if mocap_transition.transfer is not None and mocap_transition.ready(
                                # [archived branch] control_time, actual_com, actual_com_velocity, q):
                            # [archived branch] balance_ready[mocap_transition.transfer.support] = True
                    # [archived branch] if args.mocap_unload_test:
                        # Keep both constraints: this experiment measures unloading only.
                        # [archived branch] balance_ready = [False, False]
                        # [archived branch] if (args.mocap_first_step_report is None and transfer_reference is not None
                                # [archived branch] and control_time >= transfer_reference.start_time+3.5):
                            # [archived branch] stopped_at_unload_end = True
                            # [archived branch] frame_position = managed_clock.frame
                            # [archived branch] print("Double-support unloading diagnostic completed", flush=True)
                            # [archived branch] break
                    # [archived branch] if (transfer_reference is not None and transfer_reference.release_time is None
                            # [archived branch] and not transfer_reference.ready_for_release(control_time,actual_com,actual_com_velocity,q)):
                        # [archived branch] balance_ready[transfer_reference.support] = False
                    # [archived branch] measured_left, measured_right, measured_mode = measured_contact_detector.update(foot_normal_forces)
                    # [archived branch] measured_contacts = torch.stack((measured_left,measured_right),dim=1)
                    # [archived branch] try:
                        # This diagnostic holds the first lift; source coverage remains at takeoff.
                        # Before release, the unchanged manager verifies measured load and balance.
                        # [archived branch] if transfer_reference is None or transfer_reference.release_time is None:
                            # [archived branch] managed_clock = reference_manager.step(foot_normal_forces[0].cpu().numpy(),
                                # [archived branch] measured_contacts[0].cpu().numpy(),FABRIC_TIMESTEP,warming_up,sole_clearance,balance_ready,
                                # [archived branch] sole_max_height=_to_numpy(observed_corners[...,2].amax(-1)[0]) if args.early_touchdown else None,
                                # [archived branch] touchdown_contact=loaded_sole_contacts(observed_feet,tuple(map(_to_numpy,contact_data)))
                                    # [archived branch] if patch_manager is not None else None)
                            # [archived branch] if (transfer_reference is not None and managed_clock.mode != 3
                                    # [archived branch] and (args.coordinated_source_start or args.unified_source_step_test)):
                                # [archived branch] from fabrics_sim.reference.source_sequence import hold_takeoff_clock
                                # [archived branch] managed_clock = hold_takeoff_clock(managed_clock,reference_manager.takeoff_frame)
                        # [archived branch] else:
                            # [archived branch] measured_support = (bool(measured_contacts[0, transfer_reference.support])
                                # [archived branch] if source_step_rollout is None else source_step_rollout.has_measured_support(
                                    # [archived branch] measured_contacts[0].cpu().numpy()))
                            # [archived branch] if not measured_support:
                                # [archived branch] raise RuntimeError("Whole-body transfer lost its measured support")
                            # [archived branch] from dataclasses import replace
                            # [archived branch] managed_clock = replace(managed_clock, previous_frame=managed_clock.frame,
                                                    # [archived branch] next_frame=managed_clock.frame, rate=0.)
                            # [archived branch] if support_switch is not None:
                                # [archived branch] mode = support_switch.update(control_time,_to_numpy(foot_normal_forces[0]),
                                    # [archived branch] _to_numpy(measured_contacts[0]),actual_com,actual_com_velocity,q)
                                # [archived branch] managed_clock = replace(managed_clock,mode=mode,support_mode=support_switch.support+1,phase="SUPPORT_SWITCH")
                            # [archived branch] elif source_step_rollout is not None:
                                # [archived branch] elapsed = control_time-source_step_start
                                # [archived branch] step_mode = source_step_rollout.contact_mode(_to_numpy(foot_normal_forces[0]),
                                    # [archived branch] sole_clearance, _to_numpy(observed_corners[...,2].amax(-1)[0]))
                                # [archived branch] if elapsed > source_step_rollout.duration+.5*FABRIC_TIMESTEP:
                                    # [archived branch] if not source_step_rollout.touchdown:
                                        # [archived branch] raise RuntimeError("Source step reached its endpoint without confirmed touchdown")
                                    # [archived branch] if getattr(source_step_rollout,"is_tail",False):
                                        # [archived branch] source_steps_finished = True
                                    # [archived branch] elif source_step_number >= args.source_step_count and (args.source_step_count > 1 or args.source_contact_events):
                                        # [archived branch] source_steps_finished = True
                                    # [archived branch] elif args.source_support_switch:
                                        # [archived branch] from fabrics_sim.reference.source_sequence import next_source_event
                                        # [archived branch] pending_source_event = None
                                        # [archived branch] if args.source_contact_events and not source_recovery_active(source_step_number,
                                                # [archived branch] args.source_recovery_after_step,args.source_recovery_step_count):
                                            # [archived branch] pending_source_event = next_source_event(_to_numpy(motion_planned_modes),
                                                # [archived branch] source_step_rollout.end_frame)
                                        # [archived branch] if pending_source_event is not None and pending_source_event['kind']=='tail':
                                            # [archived branch] from fabrics_sim.reference.source_tail import DoubleSupportSourceRollout
                                            # [archived branch] from g1_armature import unitree_armatures
                                            # [archived branch] source_step_rollout = DoubleSupportSourceRollout(fabric,
                                                # [archived branch] source_step_rollout.references[-1],G1_JOINT_NAMES,control_motion_trajectory,
                                                # [archived branch] pending_source_event['start_frame'],pending_source_event['end_frame'],
                                                # [archived branch] source_step_rollout.support,FABRIC_TIMESTEP,
                                                # [archived branch] q.new_tensor(unitree_armatures(G1_JOINT_NAMES)),
                                                # [archived branch] args.output_path.with_suffix('.source_tail_plan.json'),
                                                # [archived branch] dict(lower=joint_position_limits[:,:,0],upper=joint_position_limits[:,:,1],
                                                    # [archived branch] dt=FABRIC_TIMESTEP,bandwidth=args.joint_limit_barrier_bandwidth,
                                                    # [archived branch] velocity_damper_time=args.joint_limit_time_constant,
                                                    # [archived branch] impact_recovery=args.joint_limit_impact_recovery),
                                                # [archived branch] acceleration_limits,duration=args.source_tail_duration)
                                            # [archived branch] source_step_start=control_time
                                            # [archived branch] elapsed=0.
                                            # [archived branch] step_mode=3
                                            # [archived branch] print("Final source tail started: "+json.dumps(pending_source_event),flush=True)
                                        # [archived branch] else:
                                            # [archived branch] from fabrics_sim.reference.support_switch import SupportSwitchReference
                                            # [archived branch] next_support=(1-source_step_rollout.support if pending_source_event is None
                                                # [archived branch] else pending_source_event['support'])
                                            # [archived branch] support_switch = SupportSwitchReference(fabric,source_step_rollout.references[-1],
                                                # [archived branch] G1_JOINT_NAMES,observed_feet,next_support,control_time,FABRIC_TIMESTEP,
                                                # [archived branch] dict(lower=joint_position_limits[:,:,0],upper=joint_position_limits[:,:,1],
                                                    # [archived branch] dt=FABRIC_TIMESTEP,bandwidth=args.joint_limit_barrier_bandwidth,
                                                    # [archived branch] velocity_damper_time=args.joint_limit_time_constant,
                                                    # [archived branch] impact_recovery=args.joint_limit_impact_recovery),
                                                # [archived branch] com_height_offset=.01 if source_step_number==1 else args.later_transfer_height_offset,
                                                # [archived branch] lift_height=.04 if source_step_number==1 else args.later_transfer_lift_height,
                                                # [archived branch] pelvis_roll_offset=(args.recovery_left_transfer_roll_offset
                                                    # [archived branch] if source_step_rollout.fixed_source_frame is not None and next_support==0 else 0.))
                                            # [archived branch] switch_report=dict(support_switch.transfer.preflight,source_event=pending_source_event)
                                            # [archived branch] args.output_path.with_suffix(".switch_plan.json" if source_step_number==1
                                                # [archived branch] else f".switch_{source_step_number}_plan.json").write_text(json.dumps(switch_report,indent=2)+"\n")
                                            # [archived branch] print(f"Support switch started: time={control_time:.3f}, support={support_switch.support}",flush=True)
                                    # [archived branch] elif args.coordinated_source_start:
                                        # [archived branch] raise RuntimeError("Landing reference stopped but the next load-transfer/lift trajectory is not planned"
                                            # [archived branch] if source_step_rollout.stop_duration else
                                            # [archived branch] "First landing reached but the next load-transfer trajectory is not planned")
                                    # [archived branch] else:
                                        # [archived branch] stopped_at_source_step_end = True
                                        # [archived branch] print(f"First source step reached frame {source_step_rollout.end_frame} with measured touchdown",flush=True)
                                        # [archived branch] break
                                # [archived branch] source_frame = source_step_rollout.frame(elapsed)
                                # [archived branch] previous_frame = source_step_rollout.frame(max(0.,elapsed-FABRIC_TIMESTEP))
                                # [archived branch] next_frame = source_step_rollout.frame(elapsed+FABRIC_TIMESTEP)
                                # [archived branch] managed_clock = replace(managed_clock,frame=source_frame,previous_frame=previous_frame,
                                    # [archived branch] next_frame=next_frame,rate=(next_frame-previous_frame)/(2*FABRIC_TIMESTEP*motion_fps),
                                    # [archived branch] mode=step_mode,support_mode=source_step_rollout.support+1,phase=("SOURCE_STEPS_HOLD" if source_steps_finished else "LANDING_STOP" if elapsed > source_step_rollout.swing_duration
                                        # [archived branch] else "SOURCE_LANDING" if step_mode==3 else "SOURCE_SWING"))
                    # [archived branch] except (ValueError,RuntimeError) as exc:
                        # [archived branch] save_reference_failure(args,exc,control_time,q,qd,foot_normal_forces,
                                               # [archived branch] tracking_history,servo_history)
                        # [archived branch] raise
                    # [archived branch] frame_position = managed_clock.frame
                    # [archived branch] phase_held = managed_clock.rate < 1e-5 and not warming_up
                    # [archived branch] commanded_contact_mode.fill_(managed_clock.mode)
                    # [archived branch] phase_hold_updates += int(phase_held)
                    # [archived branch] source_complete = motion_completion.update(
                        # [archived branch] frame_position, managed_clock.mode,
                        # [archived branch] reference_manager.settling and not getattr(source_step_rollout,"is_tail",False),
                        # [archived branch] foot_normal_forces[0].detach().cpu().numpy(),
                        # [archived branch] actual_com_velocity[0].detach().cpu().numpy(), FABRIC_TIMESTEP)
                    # [archived branch] if measured_steps is not None:
                        # [archived branch] measured_steps.update(frame_position, control_time, int(measured_mode.item()),
                            # [archived branch] _to_numpy(foot_normal_forces[0]), sole_clearance,
                            # [archived branch] _to_numpy(observed_feet[0,:,:3]), FABRIC_TIMESTEP)
                        # [archived branch] source_complete = source_complete and measured_steps.complete
                    # [archived branch] if args.until_motion_end and source_complete:
                        # [archived branch] stopped_at_motion_end = True
                        # [archived branch] print(f"Source endpoint and stable final support confirmed at {control_time:.2f}s",flush=True)
                        # [archived branch] break
                # [archived branch] if reference_manager is None and args.require_support_load and not warming_up:
                    # [archived branch] phase_frame_position, pending_phase_mode, phase_held = advance_contact_phase(
                        # [archived branch] phase_frame_position,
                        # [archived branch] max(0., elapsed_motion_time - previous_wall_motion_time) * motion_fps * args.motion_speed,
                        # [archived branch] motion_planned_modes, int(commanded_contact_mode.item()),
                    # [archived branch] )
                    # [archived branch] frame_position = phase_frame_position
                    # [archived branch] phase_hold_updates += int(phase_held)
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
                # [archived branch] if managed_clock is not None:
                    # [archived branch] previous_reference_q = None if warming_up else sample_motion(managed_clock.previous_frame)
                    # [archived branch] next_reference_q = None if warming_up else sample_motion(managed_clock.next_frame)
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
                        # [archived branch] frame_rate = motion_fps * args.motion_speed * clock_rate if (
                            # [archived branch] args.track_full_motion or frame_position < motion_trajectory.shape[0]-1) else 0.
                        frame_rate = motion_fps * args.motion_speed * clock_rate
                        frame_acceleration = motion_fps * args.motion_speed * clock_acceleration
                    elif managed_clock is not None:
                        frame_rate = (managed_clock.next_frame-managed_clock.previous_frame)/(2*FABRIC_TIMESTEP)
                        frame_acceleration = (managed_clock.next_frame-2*frame_position+managed_clock.previous_frame)/FABRIC_TIMESTEP**2
                    else:
                        # [archived branch] frame_rate = 0. if phase_held else (previous_frame_stride+next_frame_stride)/(2*FABRIC_TIMESTEP)
                        frame_rate = (previous_frame_stride + next_frame_stride) / (2 * FABRIC_TIMESTEP)
                        # [archived branch] frame_acceleration = 0. if phase_held else (next_frame_stride-previous_frame_stride)/FABRIC_TIMESTEP**2
                        frame_acceleration = (next_frame_stride - previous_frame_stride) / FABRIC_TIMESTEP ** 2
                    _,reference_v,reference_a = motion_sampler.kinematics(frame_position,frame_rate,frame_acceleration)
                    reference_kinematics = (reference_v,reference_a)
                if unified_reference:
                    whole_body_reference = WholeBodyReference(reference_q, reference_v, reference_a)
                    stopped_for_transfer = (managed_clock is not None and managed_clock.phase == "PREPARE_STEP" and phase_held
                        and reference_v.abs().max() < 1e-4 and reference_a.abs().max() < 1e-3)
                    # [archived branch] measured_transfer_ready = (transfer_start_gate.update(stopped_for_transfer,
                        # [archived branch] actual_com_velocity, qd[:, 6:], foot_normal_forces, FABRIC_TIMESTEP)
                        # [archived branch] if args.mocap_unload_test and transfer_reference is None else True)
                    measured_transfer_ready = True
                    try:
                        pass  # No active code in this amp-full branch.
                        # [archived branch] if ((args.unified_transfer_test or args.coordinated_source_start or args.mocap_unload_test) and transfer_reference is None
                                # [archived branch] and measured_transfer_ready
                                # [archived branch] and managed_clock.phase == "PREPARE_STEP" and phase_held
                                # [archived branch] and reference_v.abs().max() < 1e-4 and reference_a.abs().max() < 1e-3):
                            # [archived branch] from fabrics_sim.reference.contact_transfer import ContactTransferReference
                            # [archived branch] support = int(reference_manager.modes[reference_manager.pending_takeoff])-1
                            # [archived branch] transfer_initial = (WholeBodyReference(q.clone(), torch.zeros_like(qd), torch.zeros_like(qd))
                                                # [archived branch] if args.mocap_unload_test else whole_body_reference)
                            # [archived branch] transfer_reference = ContactTransferReference(
                                # [archived branch] fabric, transfer_initial, G1_JOINT_NAMES, observed_feet, support, control_time,
                                # [archived branch] preflight_dt=FABRIC_TIMESTEP,
                                # [archived branch] joint_bounds=dict(lower=joint_position_limits[:, :, 0],
                                    # [archived branch] upper=joint_position_limits[:, :, 1],dt=FABRIC_TIMESTEP,
                                    # [archived branch] bandwidth=args.joint_limit_barrier_bandwidth,
                                    # [archived branch] velocity_damper_time=args.joint_limit_time_constant,
                                    # [archived branch] impact_recovery=args.joint_limit_impact_recovery) if args.acceleration_and_braking_limits else None)
                            # [archived branch] if args.mocap_unload_test:
                                # [archived branch] from fabrics_sim.reference.bridge import WholeBodyReferenceBridge
                                # [archived branch] unload_bridge = WholeBodyReferenceBridge(
                                    # [archived branch] WholeBodyReference(q.clone(), qd.clone(), torch.zeros_like(qd)),
                                    # [archived branch] transfer_reference.sample(control_time, 3), .5)
                                # [archived branch] transfer_reference.preflight.update(initial_state="measured q and v; initial acceleration assumed zero",
                                    # [archived branch] scope="double_support_unloading_only", bridge_dynamics_certified=False,
                                    # [archived branch] measured_settle_seconds=transfer_start_gate.elapsed)
                            # [archived branch] args.output_path.parent.mkdir(parents=True, exist_ok=True)
                            # [archived branch] args.output_path.with_suffix('.transfer_plan.json').write_text(
                                # [archived branch] json.dumps(transfer_reference.preflight, indent=2)+"\n")
                            # [archived branch] print("Transfer reference preflight: "+json.dumps(transfer_reference.preflight),flush=True)
                            # [archived branch] print(f"Whole-body transfer started: time={control_time:.3f}s, frame={frame_position:.3f}, support={support}", flush=True)
                        # [archived branch] if transfer_reference is not None:
                            # [archived branch] if source_step_rollout is None:
                                # [archived branch] whole_body_reference = transfer_reference.sample(control_time, managed_clock.mode)
                                # [archived branch] if unload_bridge is not None:
                                    # [archived branch] whole_body_reference = unload_bridge.sample(control_time-transfer_reference.start_time, whole_body_reference)
                                # [archived branch] if ((args.unified_source_step_test or args.coordinated_source_start) and transfer_reference.release_time is not None
                                        # [archived branch] and control_time >= transfer_reference.release_time+3.):
                                    # [archived branch] from fabrics_sim.reference.source_step_rollout import SourceStepRollout
                                    # [archived branch] from g1_armature import unitree_armatures
                                    # [archived branch] source_step_rollout = SourceStepRollout(fabric,whole_body_reference,G1_JOINT_NAMES,
                                        # [archived branch] control_motion_trajectory,reference_manager.takeoff_frame,reference_manager.landing_frame,
                                        # [archived branch] transfer_reference.support,FABRIC_TIMESTEP,
                                        # [archived branch] q.new_tensor(unitree_armatures(G1_JOINT_NAMES)),args.output_path.with_suffix('.step_plan.json'),
                                        # [archived branch] candidate_options=(dict(duration=1.5,
                                            # [archived branch] landing_height=.63 if args.source_step_landing_fraction < 1 else .61,
                                            # [archived branch] swing_phase_blend=.5 if args.source_step_landing_fraction < 1 else .7,
                                            # [archived branch] landing_com_fraction=.45, landing_fraction=args.source_step_landing_fraction,
                                            # [archived branch] landing_search_depth=args.source_step_landing_search_depth,
                                            # [archived branch] terminal_capture='double_support',
                                            # [archived branch] pelvis_pitch_offset=-.2 if args.source_step_landing_fraction < 1 else -.4)
                                            # [archived branch] | source_step_profile.get('first',{})
                                            # [archived branch] if args.coordinated_source_start else None),
                                        # [archived branch] strict_landing=bool(args.tracking_steps),
                                        # [archived branch] stop_duration=args.source_step_stop_duration,
                                        # [archived branch] acceleration_limits=acceleration_limits,
                                        # [archived branch] joint_bounds=dict(lower=joint_position_limits[:, :, 0],
                                            # [archived branch] upper=joint_position_limits[:, :, 1],dt=FABRIC_TIMESTEP,
                                            # [archived branch] bandwidth=args.joint_limit_barrier_bandwidth,
                                            # [archived branch] velocity_damper_time=args.joint_limit_time_constant,
                                            # [archived branch] impact_recovery=args.joint_limit_impact_recovery) if args.acceleration_and_braking_limits else None)
                                    # [archived branch] fabric.dynamics_tasks.preserve_reference_acceleration = True
                                    # [archived branch] source_step_start = control_time
                                    # [archived branch] source_step_number = 1
                                    # [archived branch] print(f"First source swing started: time={control_time:.3f}s",flush=True)
                            # [archived branch] if support_switch is not None:
                                # [archived branch] whole_body_reference = support_switch.sample(control_time)
                                # [archived branch] if (source_step_number < args.source_step_count
                                        # [archived branch] and support_switch.transfer.release_time is not None
                                        # [archived branch] and control_time >= support_switch.transfer.release_time+3.):
                                    # [archived branch] from fabrics_sim.reference.source_sequence import next_source_landing
                                    # [archived branch] from fabrics_sim.reference.source_step_rollout import SourceStepRollout
                                    # [archived branch] from g1_armature import unitree_armatures
                                    # [archived branch] start_frame = source_step_rollout.end_frame
                                    # [archived branch] recovering = source_recovery_active(source_step_number,
                                        # [archived branch] args.source_recovery_after_step,args.source_recovery_step_count)
                                    # [archived branch] takeoff, end_frame = ((start_frame,start_frame) if recovering else
                                        # [archived branch] (pending_source_event["takeoff_frame"],pending_source_event["end_frame"])
                                        # [archived branch] if pending_source_event is not None else next_source_landing(
                                            # [archived branch] _to_numpy(motion_planned_modes),start_frame,support_switch.support))
                                    # [archived branch] next_rollout = SourceStepRollout(fabric,whole_body_reference,G1_JOINT_NAMES,
                                        # [archived branch] control_motion_trajectory,start_frame,end_frame,support_switch.support,FABRIC_TIMESTEP,
                                        # [archived branch] q.new_tensor(unitree_armatures(G1_JOINT_NAMES)),
                                        # [archived branch] args.output_path.with_suffix(f'.step_{source_step_number+1}_plan.json'),
                                        # [archived branch] candidate_options=dict(duration=2.,landing_height=args.following_step_landing_height,
                                            # [archived branch] swing_phase_blend=.85,landing_com_fraction=(args.recovery_landing_com_fraction if recovering else .45),
                                            # [archived branch] clearance=.01,body_join="distributed",
                                            # [archived branch] landing_fraction=(1. if recovering else args.following_step_landing_fraction if source_step_number==1
                                                # [archived branch] else args.later_step_landing_fraction),
                                            # [archived branch] landing_search_depth=args.source_step_landing_search_depth,
                                            # [archived branch] terminal_capture='double_support',pelvis_pitch_offset=-.2)
                                            # [archived branch] | source_step_profile.get('following',{}),
                                        # [archived branch] stop_duration=(args.source_step_stop_duration if source_step_number==1
                                            # [archived branch] else args.later_step_stop_duration),acceleration_limits=acceleration_limits,
                                        # [archived branch] strict_landing=bool(args.tracking_steps),
                                        # [archived branch] fixed_source_frame=start_frame if recovering else None,
                                        # [archived branch] maximum_step_distance=args.recovery_maximum_step_distance if recovering else None,
                                        # [archived branch] joint_bounds=dict(lower=joint_position_limits[:,:,0],upper=joint_position_limits[:,:,1],
                                            # [archived branch] dt=FABRIC_TIMESTEP,bandwidth=args.joint_limit_barrier_bandwidth,
                                            # [archived branch] velocity_damper_time=args.joint_limit_time_constant,
                                            # [archived branch] impact_recovery=args.joint_limit_impact_recovery) if args.acceleration_and_braking_limits else None)
                                    # [archived branch] source_step_rollout = next_rollout
                                    # [archived branch] source_step_start = control_time
                                    # [archived branch] source_step_number += 1
                                    # [archived branch] support_switch = None
                                    # [archived branch] print(f"Source step {source_step_number} started: time={control_time:.3f}, frames={start_frame}-{end_frame}, takeoff={takeoff}",flush=True)
                            # [archived branch] elif source_step_rollout is not None:
                                # [archived branch] whole_body_reference = (source_step_rollout.references[-1] if source_steps_finished
                                    # [archived branch] else source_step_rollout.sample(control_time-source_step_start))
                            # [archived branch] reference_q, reference_v, reference_a = whole_body_reference.q, whole_body_reference.v, whole_body_reference.a
                            # [archived branch] reference_kinematics = (reference_v, reference_a)
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
                    # [archived branch] if elapsed_motion_time>=args.measured_stance_reference_start:
                        # [archived branch] from measured_stance_reference import MeasuredStanceReference
                        # [archived branch] if 'measured_stance_reference' not in locals():
                            # [archived branch] measured_stance_reference=MeasuredStanceReference(
                                # [archived branch] 'src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf',
                                # [archived branch] G1_JOINT_NAMES,_to_numpy(motion_planned_modes),motion_fps,
                                # [archived branch] ramp=args.measured_stance_reference_ramp,limit=args.measured_stance_reference_limit)
                        # [archived branch] measured_stance_reference.observe(frame_position,_to_numpy(q[0]),
                            # [archived branch] _to_numpy(foot_normal_forces[0]),tuple(map(_to_numpy,contact_data)))
                        # [archived branch] whole_body_reference=measured_stance_reference.project(frame_position,frame_rate,
                            # [archived branch] frame_acceleration,motion_sampler,whole_body_reference)
                        # [archived branch] reference_q,reference_v,reference_a=whole_body_reference.q,whole_body_reference.v,whole_body_reference.a
                        # [archived branch] reference_kinematics=(reference_v,reference_a)
                    # [archived branch] if elapsed_motion_time >= args.stance_heading_start:
                        # [archived branch] from g1_stance_heading import StanceHeadingReference
                        # [archived branch] try:
                            # [archived branch] if 'stance_heading_reference' not in locals():
                                # [archived branch] stance_heading_reference = StanceHeadingReference(
                                    # [archived branch] 'src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf',
                                    # [archived branch] G1_JOINT_NAMES, _to_numpy(motion_planned_modes), motion_fps,
                                    # [archived branch] ramp=args.stance_heading_ramp, limit=args.stance_heading_limit)
                            # [archived branch] stance_heading_reference.observe(frame_position, _to_numpy(q[0]),
                                # [archived branch] _to_numpy(foot_normal_forces[0]), tuple(map(_to_numpy, contact_data)))
                            # [archived branch] whole_body_reference = stance_heading_reference.project(
                                # [archived branch] frame_position, frame_rate, frame_acceleration, motion_sampler, whole_body_reference)
                            # [archived branch] reference_q, reference_v, reference_a = (whole_body_reference.q,
                                # [archived branch] whole_body_reference.v, whole_body_reference.a)
                            # [archived branch] reference_kinematics = (reference_v, reference_a)
                        # [archived branch] except (ValueError, RuntimeError) as exc:
                            # [archived branch] save_reference_failure(args, exc, control_time, q, qd, foot_normal_forces,
                                                   # [archived branch] tracking_history, servo_history)
                            # [archived branch] raise
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
                    # [archived branch] if reference_manager is not None:
                        # [archived branch] from fabrics_sim.reference.posture import crouched_posture
                        # [archived branch] minimum_knee=.2 if args.reference_footsteps == "source" else .5
                        # [archived branch] posture, crouch_blend = crouched_posture(reference_q[:,7:],G1_JOINT_NAMES,control_time,
                                                                 # [archived branch] minimum_knee=minimum_knee)
                        # [archived branch] previous_posture,_ = crouched_posture(
                            # [archived branch] (reference_q if previous_reference_q is None else previous_reference_q)[:,7:],
                            # [archived branch] G1_JOINT_NAMES,control_time-FABRIC_TIMESTEP,minimum_knee=minimum_knee)
                        # [archived branch] next_posture,_ = crouched_posture(
                            # [archived branch] (reference_q if next_reference_q is None else next_reference_q)[:,7:],
                            # [archived branch] G1_JOINT_NAMES,control_time+FABRIC_TIMESTEP,minimum_knee=minimum_knee)
                        # [archived branch] posture_v,posture_a = finite_difference_reference(
                            # [archived branch] previous_posture,posture,next_posture,FABRIC_TIMESTEP,False,False)
                        # [archived branch] if reference_kinematics is not None:
                            # [archived branch] from fabrics_sim.reference.posture import crouched_posture_kinematics
                            # [archived branch] posture,posture_v,posture_a,crouch_blend = crouched_posture_kinematics(
                                # [archived branch] reference_q[:,7:],reference_v[:,6:],reference_a[:,6:],
                                # [archived branch] G1_JOINT_NAMES,control_time,minimum_knee=minimum_knee)
                        # [archived branch] fabric.set_joint_posture_target(posture,posture_v,posture_a)
                        # Adapted slow steps do not share the source clip's angular
                        # momentum; asking for that momentum conflicts with balance.
                        # [archived branch] if not args.source_momentum_reference:
                        # [archived branch] fabric.set_centroidal_momentum_target(torch.zeros(1,3,device=device))
                        # [archived branch] if args.reference_footsteps == "source":
                            # Stabilize the torso during dynamic transfer. A pelvis-
                            # only orientation target leaves waist pitch available
                            # to satisfy COM/hand tasks until it hits a joint stop.
                            # [archived branch] body_model=fabric.dynamics_tasks.model
                            # [archived branch] body_index=body_model.kinematics.get_link_index('torso_link')
                            # [archived branch] if reference_state is not None:
                                # [archived branch] body_quat=reference_state['transforms'][:,body_index,3:7]
                                # [archived branch] body_jac=reference_state['angular_jac'][:,body_index]
                                # [archived branch] body_v=(body_jac@reference_v[...,None]).squeeze(-1)
                                # [archived branch] body_a=(body_jac@reference_a[...,None]).squeeze(-1)+reference_state['angular_jac_bias'][:,body_index]
                            # [archived branch] else:
                                # [archived branch] body_quat=body_model.snapshot(reference_q)['transforms'][:,body_index,3:7].clone()
                                # [archived branch] before_quat=(body_quat if previous_reference_q is None else
                                    # [archived branch] body_model.snapshot(previous_reference_q)['transforms'][:,body_index,3:7].clone())
                                # [archived branch] after_quat=(body_quat if next_reference_q is None else
                                    # [archived branch] body_model.snapshot(next_reference_q)['transforms'][:,body_index,3:7].clone())
                                # [archived branch] body_v,body_a=angular_reference(before_quat,body_quat,after_quat,FABRIC_TIMESTEP,
                                    # [archived branch] previous_reference_q is None,next_reference_q is None)
                            # [archived branch] fabric.set_body_orientation_target(body_quat,body_v,body_a,link='torso_link')
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
                    # [archived branch] if pending_phase_mode is not None:
                        # [archived branch] planned_contact_mode = torch.full_like(commanded_contact_mode, pending_phase_mode)
                    # [archived branch] if phase_held:
                        # Do not look through to the opposite support phase
                        # while waiting at this phase boundary.
                        # [archived branch] support_mode = planned_contact_mode
                        # [archived branch] support_reference_q = reference_q
                # Use forces from the same physics snapshot as q and qd.
                # An extra controller-tick delay postpones touchdown and
                # loss-of-support handling by 20 ms.
                # [archived branch] if reference_manager is None:
                measured_left, measured_right, measured_mode = measured_contact_detector.update(foot_normal_forces)
                measured_contacts = torch.stack((measured_left, measured_right),dim=1)
                # [archived branch] else:
                    # [archived branch] planned_contact_mode = motion_planned_modes[motion_frame:motion_frame+1]
                    # [archived branch] support_mode = torch.full_like(commanded_contact_mode, managed_clock.support_mode)
                    # [archived branch] support_reference_q = reference_q
                # [archived branch] if args.single_support_test:
                    # [archived branch] support_index = 0 if args.single_support_test == "left" else 1
                    # [archived branch] support_mode_value = support_index + 1
                    # [archived branch] live_com, com_jac = fabric.center_of_mass_taskmap._position_and_jacobian(q)
                    # [archived branch] com_velocity = torch.bmm(com_jac, qd.unsqueeze(-1)).squeeze(-1)
                    # [archived branch] support_jac = left_j if support_index == 0 else right_j
                    # [archived branch] foot_speed = torch.linalg.vector_norm(torch.bmm(support_jac[:, :3], qd.unsqueeze(-1)).squeeze(-1), dim=1)
                    # [archived branch] ready = single_support_ready(live_com, com_velocity, com_balance_target,
                                                 # [archived branch] single_initial_feet[:, support_index, :3], foot_speed,
                                                 # [archived branch] foot_normal_forces, support_index)
                    # [archived branch] ready &= measured_contacts[:, support_index]
                    # [archived branch] ready &= control_time >= args.motion_warmup_duration + cycle_preparation + (args.support_transfer_duration if contact_cycle is None else 0.)
                    # [archived branch] if contact_cycle is not None:
                        # [archived branch] swing = 1 - support_index
                        # [archived branch] swing_jac = right_j if swing else left_j
                        # [archived branch] swing_speed = torch.linalg.vector_norm((swing_jac @ qd[..., None]).squeeze(-1)).item()
                        # [archived branch] cycle_feet = reference_feet_transforms(fabric, q)
                        # [archived branch] clearance = (cycle_feet[0, swing, 2] - single_initial_feet[0, swing, 2]).item()
                        # [archived branch] settled = bool((foot_normal_forces >= 20.).all()) and com_velocity.norm().item() < .05 and swing_speed < .03 and (live_com-com_balance_target).norm().item() < .03
                        # [archived branch] old_phase = contact_cycle.phase
                        # [archived branch] contact_cycle.update(control_time, ready=bool(ready.all()), clearance=clearance,
                                             # [archived branch] swing_load=foot_normal_forces[0, swing].item(),
                                             # [archived branch] swing_speed=swing_speed, settled=settled)
                        # [archived branch] if contact_cycle.phase != old_phase:
                            # [archived branch] print(f"contact cycle: {old_phase} -> {contact_cycle.phase} at {control_time:.3f}s", flush=True)
                        # [archived branch] if single_release_time is None and contact_cycle.phase == "lift":
                            # The landing target is the settled sole pose, not
                            # the pre-drop initialization height.
                            # [archived branch] single_initial_feet = cycle_feet.clone()
                            # [archived branch] single_release_time = control_time
                            # [archived branch] commanded_contact_mode.fill_(support_mode_value)
                    # [archived branch] single_ready_count = single_ready_count + 1 if bool(ready.all()) else 0
                    # [archived branch] if contact_cycle is None and single_release_time is None and single_ready_count >= 10:
                        # [archived branch] single_release_time = control_time
                        # [archived branch] commanded_contact_mode.fill_(support_mode_value)
                        # [archived branch] print(f"single support: release at {control_time:.3f}s, support={args.single_support_test}")
                    # [archived branch] if single_release_time is not None:
                        # [archived branch] planned_contact_mode = torch.full_like(commanded_contact_mode, support_mode_value)
                    # [archived branch] if contact_cycle is not None and contact_cycle.phase in ("settle", "done"):
                        # [archived branch] planned_contact_mode.fill_(3)
                    # [archived branch] if control_updates % 25 == 0:
                        # [archived branch] print(f"single support gate: ready_count={single_ready_count}, released={single_release_time is not None}, "
                              # [archived branch] f"com_error={torch.linalg.vector_norm(live_com-com_balance_target).item():.4f}, "
                              # [archived branch] f"com_speed={torch.linalg.vector_norm(com_velocity).item():.4f}, "
                              # [archived branch] f"foot_speed={foot_speed.item():.4f}, forces={foot_normal_forces.tolist()}")
                # [archived branch] if not unified_reference and reference_manager is not None and managed_clock.mode in (1,2):
                    # [archived branch] from fabrics_sim.reference.landing import LandingTransition
                    # [archived branch] from fabrics_sim.reference.footstep import grounded_step
                    # [archived branch] swing = 1 if managed_clock.mode == 1 else 0
                    # [archived branch] key = (reference_manager.landing_frame,swing)
                    # [archived branch] if key != landing_key:
                        # [archived branch] landing_key = key
                        # [archived branch] source_foot_bridge = None
                        # [archived branch] source_rotation_bridge = None
                        # [archived branch] endpoint = min(reference_manager.landing_frame or motion_frame,motion_trajectory.shape[0]-1)
                        # [archived branch] source_feet = reference_feet_all[endpoint].detach().cpu().numpy()
                        # [archived branch] if args.reference_footsteps == "source":
                            # [archived branch] goal = source_landing_pose(source_feet[swing])
                        # [archived branch] else:
                            # [archived branch] goal = grounded_step(source_feet[1-swing], source_feet[swing],
                                                 # [archived branch] contact_anchors[0,1-swing].detach().cpu().numpy())
                        # [archived branch] jac = right_j if swing else left_j
                        # [archived branch] twist=(jac@qd[...,None]).squeeze(-1)[0].detach().cpu().numpy()
                        # [archived branch] measured_pose = reference_feet_transforms(fabric, q)[0, swing].detach().cpu().numpy()
                        # [archived branch] swing_duration = 2.
                        # [archived branch] if args.reference_footsteps == 'source':
                            # [archived branch] swing_duration=max(.8,(endpoint-frame_position)/(motion_fps*args.motion_speed))
                            # [archived branch] reference_manager.synchronize_swing(swing_duration)
                            # [archived branch] source_com_bridge = source_foot_bridge = None
                        # [archived branch] landing_transition=LandingTransition(control_time,measured_pose,twist,goal,
                                                             # [archived branch] duration=swing_duration,clearance=.06,search_depth=.04,
                                                             # [archived branch] search_duration=1.5)
                        # [archived branch] if args.source_swing_reference:
                            # [archived branch] from fabrics_sim.reference.phase_clock import SwingClock
                            # [archived branch] if args.swing_touchdown_lead==0:
                                # One source phase for posture and feet. The
                                # measured-state bridge handles lift-off rates;
                                # a second clock would distort mid-swing timing.
                                # [archived branch] source_foot_clock=reference_manager.swing_clock
                            # [archived branch] else:
                                # [archived branch] source_foot_clock=SwingClock(frame_position/motion_fps,managed_clock.rate,
                                                            # [archived branch] endpoint/motion_fps,swing_duration,end_rate=0.)
                    # [archived branch] pose,twist,acceleration = landing_transition.sample(control_time)
                    # [archived branch] poses=reference_feet_transforms(fabric,reference_q).detach().clone()
                    # [archived branch] poses[0,swing]=torch.as_tensor(pose,device=device,dtype=q.dtype)
                    # The stance rows are disabled in physical task requests.
                    # [archived branch] v=torch.zeros(1,2,6,device=device,dtype=q.dtype);a=torch.zeros_like(v)
                    # [archived branch] v[0,swing]=torch.as_tensor(twist,device=device,dtype=q.dtype)
                    # [archived branch] a[0,swing]=torch.as_tensor(acceleration,device=device,dtype=q.dtype)
                    # [archived branch] if args.reference_footsteps == 'compact':
                        # [archived branch] fabric.set_feet_targets(poses[...,:3],poses[...,3:],v[...,:3],v[...,3:],a[...,:3],a[...,3:])
                    # [archived branch] else:
                        # Source mode retains the coupled FK foot/COM path and
                        # sole orientation at the shared retimed source phase.
                        # Correct only the terminal sole clearance on flat ground.
                        # [archived branch] from fabrics_sim.reference.landing import LandingTransition
                        # [archived branch] endpoint_pose=reference_feet_all[endpoint,swing]
                        # [archived branch] endpoint_rotation=_quat_xyzw_to_matrix(endpoint_pose[3:])
                        # [archived branch] terminal_height=(endpoint_pose[:3,None]+endpoint_rotation@corners.T)[2].min()
                        # [archived branch] clock=source_foot_clock if args.source_swing_reference else reference_manager.swing_clock
                        # [archived branch] elapsed=control_time-landing_transition.start_time
                        # [archived branch] fraction=1. if clock is None else min(1.,elapsed/clock.duration)
                        # [archived branch] blend=10*fraction**3-15*fraction**4+6*fraction**5
                        # [archived branch] correction=terminal_height.clamp(-.04,.04)
                        # [archived branch] foot_positions=fabric.feet_pose_taskmap.position_target.clone()
                        # [archived branch] foot_quaternions=fabric.feet_pose_taskmap.quaternion_target.clone()
                        # Reference derivatives are adjusted with the same blend.
                        # [archived branch] features=fabric.fabrics_features['feet_pose']['feet_pose_attractor']
                        # [archived branch] fv,fa=fabric.dynamics_tasks.references(features,torch.zeros(1,12,device=device))
                        # [archived branch] fv,fa=fv.reshape(1,2,6).clone(),fa.reshape(1,2,6).clone()
                        # [archived branch] if args.source_swing_reference:
                            # [archived branch] foot_phase,foot_rate,foot_acc=clock.sample(elapsed)
                            # [archived branch] foot_pose,foot_v,foot_a=source_foot_samplers[swing].kinematics(
                                # [archived branch] foot_phase*motion_fps,foot_rate*motion_fps,foot_acc*motion_fps)
                            # [archived branch] foot_positions[:,swing]=foot_pose[:,:3]
                            # [archived branch] foot_quaternions[:,swing]=foot_pose[:,3:]
                            # [archived branch] fv[:,swing]=foot_v;fa[:,swing]=foot_a
                        # [archived branch] foot_positions[:,swing,2]-=blend*correction
                        # [archived branch] duration=landing_transition.duration if clock is None else clock.duration
                        # [archived branch] fv[:,swing,2]-=30*fraction**2*(1-fraction)**2/duration*correction
                        # [archived branch] fa[:,swing,2]-=(60*fraction-180*fraction**2+120*fraction**3)/duration**2*correction
                        # [archived branch] if args.source_swing_reference:
                            # [archived branch] from fabrics_sim.reference.landing import contact_search
                            # [archived branch] search=contact_search(elapsed-duration,landing_transition.search_depth,
                                                  # [archived branch] landing_transition.search_duration)
                            # [archived branch] for target,offset in zip((foot_positions[:,swing,2],fv[:,swing,2],fa[:,swing,2]),search):
                                # [archived branch] target.add_(float(offset))
                        # [archived branch] from fabrics_sim.reference.bridge import ReferenceBridge
                        # [archived branch] bridge_duration=(duration if args.source_bridge_duration is None else
                                         # [archived branch] min(duration,args.source_bridge_duration))
                        # [archived branch] if source_foot_bridge is None:
                            # [archived branch] to_np=lambda x:x.detach().cpu().numpy()
                            # [archived branch] source_foot_bridge=ReferenceBridge(
                                # [archived branch] measured_pose[:3]-to_np(foot_positions[0,swing]),
                                # [archived branch] twist[:3]-to_np(fv[0,swing,:3]),
                                # [archived branch] -to_np(fa[0,swing,:3]), bridge_duration)
                        # [archived branch] offsets=source_foot_bridge.sample(control_time-landing_transition.start_time)
                        # [archived branch] for target,offset in zip((foot_positions[0,swing],fv[0,swing,:3],fa[0,swing,:3]),offsets):
                            # [archived branch] target+=torch.as_tensor(offset,device=device,dtype=q.dtype)
                        # [archived branch] if args.source_swing_reference:
                            # [archived branch] from fabrics_sim.reference.bridge import RotationReferenceBridge
                            # [archived branch] to_np=lambda x:x.detach().cpu().numpy()
                            # [archived branch] if source_rotation_bridge is None:
                                # [archived branch] source_rotation_bridge=RotationReferenceBridge(measured_pose[3:],twist[3:],
                                    # [archived branch] to_np(foot_quaternions[0,swing]),to_np(fv[0,swing,3:]),
                                    # [archived branch] to_np(fa[0,swing,3:]),bridge_duration)
                            # [archived branch] blended=source_rotation_bridge.sample(control_time-landing_transition.start_time,
                                # [archived branch] to_np(foot_quaternions[0,swing]),to_np(fv[0,swing,3:]),to_np(fa[0,swing,3:]))
                            # [archived branch] for target,value in zip((foot_quaternions[0,swing],fv[0,swing,3:],fa[0,swing,3:]),blended):
                                # [archived branch] target.copy_(torch.as_tensor(value,device=device,dtype=q.dtype))
                        # [archived branch] fabric.set_feet_targets(foot_positions,foot_quaternions,
                            # [archived branch] fv[:,:,:3],fv[:,:,3:],fa[:,:,:3],fa[:,:,3:])
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
                            _to_numpy(foot_normal_forces[0]),motion_fps,contact_plan_value,horizon=args.anticipate_unload_horizon,hysteresis=args.anticipate_unload_hysteresis,allow_overdue=args.anticipate_unload_overdue)
                    controller_contact_plan=torch.full_like(planned_contact_mode,contact_plan_value)
                # [archived branch] if reference_manager is None:
                # [archived branch] commanded_contact_mode, contact_ready_counter = (
                    # [archived branch] update_contact_mode_state_machine(
                        # [archived branch] commanded_contact_mode,
                        # [archived branch] controller_contact_plan,
                        # [archived branch] measured_contacts,
                        # [archived branch] contact_ready_counter,
                        # [archived branch] args.contact_switch_confirm_updates,
                        # Dynamic walking still needs measured touchdown/load gates;
                        # --require-support-load also changes the legacy source clock.
                        # [archived branch] foot_normal_forces=foot_normal_forces if (
                            # [archived branch] args.require_support_load or args.motion_start_in_motion) else None,
                    # [archived branch] )
                # [archived branch] )
                (commanded_contact_mode, contact_ready_counter) = update_contact_mode_state_machine(
                    commanded_contact_mode,
                    controller_contact_plan,
                    measured_contacts,
                    contact_ready_counter,
                    args.contact_switch_confirm_updates,
                    foot_normal_forces=foot_normal_forces,
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
                    # [archived branch] raw_com_target = support_aware_com_target(
                        # [archived branch] fabric,
                        # [archived branch] reference_q,
                        # [archived branch] balance_support_reference_q,
                        # [archived branch] balance_support_mode,
                        # [archived branch] support_anchors=contact_anchors[..., :3] if reference_manager is not None else None,
                        # [archived branch] anchored_contacts=torch.stack(((commanded_contact_mode == 1) | (commanded_contact_mode == 3),
                                                       # [archived branch] (commanded_contact_mode == 2) | (commanded_contact_mode == 3)), dim=1),
                    # [archived branch] )
                    raw_com_target = support_aware_com_target(
                        fabric,
                        reference_q,
                        balance_support_reference_q,
                        balance_support_mode,
                        support_anchors=None,
                        anchored_contacts=torch.stack(((commanded_contact_mode == 1) | (commanded_contact_mode == 3), (commanded_contact_mode == 2) | (commanded_contact_mode == 3)), dim=1),
                    )
                    # [archived branch] if reference_manager is not None and managed_clock.support_mode in (1,2):
                        # [archived branch] support_index = managed_clock.support_mode-1
                        # [archived branch] raw_com_target[:,:2] = .95*contact_anchors[:,support_index,:2]+.05*raw_com_target[:,:2]
                    # [archived branch] if reference_manager is not None and reference_manager.settling:
                        # [archived branch] support_index = reference_manager.landing_support-1
                        # Give the landed foot a sustained share of load before
                        # asking it to become the next support. Keeping COM fully
                        # over the old ankle would make the settle gate impossible.
                        # [archived branch] raw_com_target[:,:2] = (.8*contact_anchors[:,support_index,:2]
                                               # [archived branch] +.2*contact_anchors[:,1-support_index,:2])
                    # [archived branch] if reference_manager is not None:
                        # [archived branch] raw_com_target[:,2] = (fabric.compute_center_of_mass(reference_q).detach()[:,2]-.005*crouch_blend
                                              # [archived branch] if args.reference_footsteps == "source" else initial_center_of_mass[:,2]-.06*crouch_blend)
                    # [archived branch] if args.weight_transfer_test:
                        # [archived branch] transfer_target = support_aware_com_target(
                            # [archived branch] fabric, reference_q, reference_q, torch.full_like(commanded_contact_mode, 2))
                        # [archived branch] phase = max(0., min(1., (control_time - args.motion_warmup_duration) / 3.))
                        # [archived branch] blend = phase**3 * (10. - 15. * phase + 6. * phase**2)
                        # [archived branch] raw_com_target = initial_center_of_mass + blend * (transfer_target - initial_center_of_mass)
                    # [archived branch] if args.single_support_test:
                        # [archived branch] transfer_target = initial_center_of_mass.clone()
                        # [archived branch] transfer_target[:, :2] = (.95 * single_initial_feet[:, support_index, :2]
                                                 # [archived branch] + .05 * initial_center_of_mass[:, :2])
                        # [archived branch] phase = max(0., min(1., (control_time - args.motion_warmup_duration - cycle_preparation) / args.support_transfer_duration))
                        # [archived branch] blend = phase**3 * (10. - 15. * phase + 6. * phase**2)
                        # [archived branch] raw_com_target = initial_center_of_mass + blend * (transfer_target - initial_center_of_mass)
                        # [archived branch] if cycle_preparation:
                            # [archived branch] from fabrics_sim.solver.contact_cycle import smooth_height
                            # [archived branch] depth, depth_v, depth_a = smooth_height(control_time-args.motion_warmup_duration,
                                                                   # [archived branch] cycle_preparation, 0., args.cycle_crouch_depth)
                            # [archived branch] raw_com_target[:, 2] -= depth
                            # [archived branch] posture = reference_q[:, 7:].clone()
                            # [archived branch] posture_v, posture_a = torch.zeros_like(posture), torch.zeros_like(posture)
                            # [archived branch] for side in ("left", "right"):
                                # [archived branch] for joint, gain in (("hip_pitch", -8.), ("knee", 16.), ("ankle_pitch", -8.)):
                                    # [archived branch] index = G1_JOINT_NAMES.index(f"{side}_{joint}_joint")
                                    # [archived branch] posture[:, index] += gain * depth
                                    # [archived branch] posture_v[:, index], posture_a[:, index] = gain * depth_v, gain * depth_a
                            # [archived branch] fabric.set_joint_posture_target(posture, posture_v, posture_a)
                        # [archived branch] if contact_cycle is not None and contact_cycle.landing_time is not None:
                            # [archived branch] from fabrics_sim.solver.contact_cycle import smooth_height
                            # [archived branch] return_blend, _, _ = smooth_height(control_time-contact_cycle.landing_time, 3., 0., 1.)
                            # [archived branch] raw_com_target[:, :2] = ((1-return_blend)*raw_com_target[:, :2]
                                                    # [archived branch] + return_blend*initial_center_of_mass[:, :2])
                        # [archived branch] if single_release_time is not None:
                            # One-second quintic vertical lift; no PKL joint command or integrator reset.
                            # [archived branch] s = min(1., max(0., control_time - single_release_time))
                            # [archived branch] p = single_initial_feet[..., :3].clone()
                            # [archived branch] v = torch.zeros_like(p)
                            # [archived branch] a = torch.zeros_like(p)
                            # [archived branch] swing = 1 - support_index
                            # [archived branch] p[:, swing, 2] += .04 * s**3 * (10. - 15.*s + 6.*s*s)
                            # [archived branch] v[:, swing, 2] = .04 * 30.*s*s*(1.-s)**2
                            # [archived branch] a[:, swing, 2] = .04 * 60.*s*(1.-s)*(1.-2.*s)
                            # [archived branch] if contact_cycle is not None:
                                # [archived branch] dz, vz, az = contact_cycle.reference(control_time)
                                # [archived branch] p[:, swing, 2] = single_initial_feet[:, swing, 2] + dz
                                # [archived branch] v[:, swing, 2], a[:, swing, 2] = vz, az
                            # [archived branch] fabric.set_feet_targets(p, single_initial_feet[..., 3:], v, torch.zeros_like(v), a, torch.zeros_like(a))
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
                    # [archived branch] if reference_manager is not None:
                        # Preserve the hand gesture relative to COM when shortening
                        # steps; absolute PKL hand positions would pull the body
                        # toward footsteps that the manager intentionally shortened.
                        # [archived branch] if reference_state is not None:
                            # [archived branch] source_com=reference_state['com']
                            # [archived branch] source_v=(reference_state['com_jac']@reference_v[...,None]).squeeze(-1)
                            # [archived branch] source_a=(reference_state['com_jac']@reference_a[...,None]).squeeze(-1)+reference_state['com_jac_bias']
                        # [archived branch] else:
                            # [archived branch] source_com = fabric.compute_center_of_mass(reference_q).detach()
                            # [archived branch] previous_com = source_com if previous_reference_q is None else fabric.compute_center_of_mass(previous_reference_q).detach()
                            # [archived branch] next_com = source_com if next_reference_q is None else fabric.compute_center_of_mass(next_reference_q).detach()
                            # [archived branch] source_v, source_a = finite_difference_reference(
                                # [archived branch] previous_com, source_com, next_com, FABRIC_TIMESTEP,
                                # [archived branch] previous_reference_q is None, next_reference_q is None)
                        # [archived branch] features = fabric.fabrics_features["hand_pose"]["hand_pose_attractor"]
                        # [archived branch] hand_v, hand_a = fabric.dynamics_tasks.references(features, torch.zeros(1,12,device=device))
                        # [archived branch] hand_v, hand_a = hand_v.reshape(1,2,6).clone(), hand_a.reshape(1,2,6).clone()
                        # [archived branch] hand_v[:,:,:3] += (com_target_velocity-source_v)[:,None]
                        # [archived branch] hand_a[:,:,:3] += (com_target_acceleration-source_a)[:,None]
                        # [archived branch] fabric.set_hand_targets(
                            # [archived branch] fabric.hand_pose_taskmap.position_target+(com_balance_target-source_com)[:,None],
                            # [archived branch] fabric.hand_pose_taskmap.quaternion_target,
                            # [archived branch] hand_v[:,:,:3], hand_v[:,:,3:], hand_a[:,:,:3], hand_a[:,:,3:])
                    # [archived branch] if args.align_balance_pelvis:
                        # [archived branch] align_pelvis_balance_reference(
                            # [archived branch] fabric, reference_q, previous_reference_q, next_reference_q, FABRIC_TIMESTEP,
                            # [archived branch] com_balance_target, com_target_velocity, com_target_acceleration)
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
            # [archived branch] if reference_manager is not None and warming_up:
                # Let the spawn settle onto the floor, then keep the settled
                # support poses fixed. This updates reference anchors only.
                # [archived branch] contact_anchors = live_feet.detach().clone()

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
            # [archived branch] if elapsed_motion_time>=args.collision_barrier_bandwidth_start:
                # [archived branch] if not math.isfinite(args.collision_barrier_bandwidth) or args.collision_barrier_bandwidth<=0:
                    # [archived branch] raise ValueError('Collision barrier bandwidth must be finite and positive')
                # [archived branch] cb_phase=min(1.,max(0.,(elapsed_motion_time-args.collision_barrier_bandwidth_start)/.1))
                # [archived branch] fabric.dynamics_tasks.collision_barrier_bandwidth=8.+(args.collision_barrier_bandwidth-8.)*cb_phase**3*(10-15*cb_phase+6*cb_phase**2)
            if args.convex_wrist_collision and elapsed_motion_time>=args.convex_wrist_collision_start:
                fabric.dynamics_tasks.convex_wrist_geometry=delayed_convex_geometry
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
            # [archived branch] if args.landing_speed_limit is not None:
                # [archived branch] count=len(foot_landing_geometry[0][0])
                # [archived branch] landing_diagnostic=dict(landing_surface_height=np.full((1,count),np.nan),
                    # [archived branch] landing_surface_speed=np.full((1,count),np.nan),landing_speed_active=np.zeros((1,count),dtype=bool))
                # [archived branch] landing_mode = int(commanded_contact_mode.item()) if args.motion_start_in_motion else managed_clock.mode
                # [archived branch] landing_enabled = args.motion_start_in_motion or reference_manager.swing_clear
                # [archived branch] if (landing_mode in (1,2) and landing_enabled
                        # [archived branch] and control_time >= args.landing_speed_limit_start):
                    # [archived branch] from fabrics_sim.solver.landing_approach import landing_approach_constraints
                    # [archived branch] swing=1 if landing_mode==1 else 0
                    # [archived branch] C,d,height,speed,active=landing_approach_constraints(
                        # [archived branch] live_feet[:,swing,:3],_quat_xyzw_to_matrix(live_feet[:,swing,3:]),
                        # [archived branch] (left_j,right_j)[swing],contact_bias[:,swing],qd,*foot_landing_geometry[swing],
                        # [archived branch] touchdown_speed=args.landing_speed_limit)
                    # [archived branch] if task_acceleration_constraints is None:
                        # [archived branch] task_acceleration_constraints=C,d
                    # [archived branch] else:
                        # [archived branch] old_C,old_d=task_acceleration_constraints
                        # [archived branch] task_acceleration_constraints=torch.cat((old_C,C),1),torch.cat((old_d,d),1)
                    # [archived branch] landing_diagnostic=dict(landing_surface_height=_to_numpy(height),
                        # [archived branch] landing_surface_speed=_to_numpy(speed),landing_speed_active=_to_numpy(active))
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
                # [archived branch] if managed_clock is not None and managed_clock.mode == 3 and com_preview is not None:
                    # [archived branch] candidate = reference_manager.outgoing_foot(managed_clock.support_mode)
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
                # [archived branch] if toe_contact is None and candidate is not None:
                    # [archived branch] forces, points, normals, distances, counts, starts = map(_to_numpy,contact_data)
                    # [archived branch] start = int(starts.reshape(-1)[candidate]); count = int(counts.reshape(-1)[candidate])
                    # [archived branch] rotation = _quat_xyzw_to_matrix(live_feet[:,candidate,3:])
                    # [archived branch] local = (_to_numpy(rotation[0]).T @ (points[start:start+count]-_to_numpy(live_feet[0,candidate,:3])).T).T
                    # [archived branch] from fabrics_sim.solver.toe_contact import has_touching_toe_line, has_loaded_toe_line
                    # Incoming load is checked below. The outgoing toe may
                    # already be unloaded while its measured manifold still
                    # touches the floor; do not pin the heel in that case.
                    # [archived branch] touching = has_touching_toe_line(local,distances[start:start+count])
                    # [archived branch] source_rollover = False
                    # [archived branch] load = foot_normal_forces[0]
                    # [archived branch] ready = managed_clock.mode == 3 and touching and bool(load.sum()>=100.) and bool(
                        # [archived branch] load[1-candidate]>=args.toe_transfer_load_fraction*load.sum())
                    # [archived branch] if source_rollover:
                        # Actual loaded corners establish this line. No
                        # airborne support or pitch contact moment is assumed.
                        # [archived branch] ready=bool(load[candidate]>=100.) and has_loaded_toe_line(
                            # [archived branch] local,forces[start:start+count],distances[start:start+count])
                    # [archived branch] toe_ready_seconds = toe_ready_seconds+FABRIC_TIMESTEP if ready else 0.
                    # Once this foot is almost unloaded, holding all six flat
                    # contact coordinates can prevent its heel from releasing
                    # while the incoming heel takes an impact. Geometry must
                    # still establish BOTH touching toe endpoints, and the
                    # other foot must carry the load. Only heel pitch is freed;
                    # the manager's full-foot release gates remain unchanged.
                    # [archived branch] unloaded_toe_ready = (ready and managed_clock.mode == 3
                                          # [archived branch] and float(load[candidate]) <= 5.)
                    # [archived branch] if toe_contact is None and (unloaded_toe_ready or toe_ready_seconds>=.06-1e-9):
                        # [archived branch] from fabrics_sim.solver.toe_contact import ToeContact
                        # [archived branch] toe_foot = candidate
                        # [archived branch] if patch_manager is not None:
                            # [archived branch] from fabrics_sim.solver.observed_sole_contact import ObservedSoleContact
                            # [archived branch] toe_contact = ObservedSoleContact(live_feet[:,candidate,:3],rotation,control_time)
                        # [archived branch] else:
                            # [archived branch] toe_contact = ToeContact(live_feet[:,candidate,:3],rotation,control_time)
                        # [archived branch] print(f"Toe-line support: foot={toe_foot}, time={control_time:.2f}s",flush=True)
                        # [archived branch] toe_ready_seconds = 0.
                if toe_contact is None:
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
                    # [archived branch] if control_time >= args.rolling_contact_kinematics_start:
                        # [archived branch] patch_points = (patch.vertices if getattr(patch, 'vertices', None) is not None
                                        # [archived branch] else patch.local_point[:, None])
                        # [archived branch] omega = (jac[:, 3:] @ qd[..., None]).squeeze(-1)
                        # [archived branch] correction, matched = rolling_sphere_bias_correction(
                            # [archived branch] _quat_xyzw_to_matrix(live_feet[:, patch_foot, 3:]), omega,
                            # [archived branch] patch.frame, patch_points, *foot_landing_geometry[patch_foot])
                        # [archived branch] mapped_bias = mapped_bias.clone()
                        # [archived branch] mapped_bias[:, :3] += correction
                        # [archived branch] rolling_bias_correction[:, patch_foot] = correction
                        # [archived branch] rolling_bias_matched[:, patch_foot] = matched
                    if patch_foot == 0:
                        left_j,left_contact_bias = mapped_jac,mapped_bias
                    else:
                        right_j,right_contact_bias = mapped_jac,mapped_bias
                    pose_error[:,patch_foot] = mapped_error
                    flatten_task = toe_task
                    if unified_reference:
                        from fabrics_sim.reference.whole_body import support_orientation_request
                        foot_request = next(r for r in fabric.dynamics_tasks.last_requests if r.name == 'feet_pose')
                        toe_task = support_orientation_request(
                            foot_request, patch_foot, patch.frame, patch.point_support,
                            getattr(patch, 'polygon_support', False))
                    elif patch_manager is not None:
                        toe_task.name += f'_{patch_foot}'
                    # [archived branch] if (unified_reference and elapsed_motion_time > args.support_flatten_start
                            # [archived branch] and patch.flatten and not getattr(patch, 'polygon_support', False)
                            # [archived branch] and bool(foot_normal_forces[0, patch_foot] >= 20.)
                            # [archived branch] and int(planned_contact_mode.item()) & (1 << patch_foot)):
                        # Only a measured loaded, planned stance foot receives
                        # this soft landing-recovery objective. Its contact
                        # rows and physical wrench cone remain measured ones.
                        # [archived branch] flat_phase = min(1., (elapsed_motion_time-args.support_flatten_start)/.1)
                        # [archived branch] flat_blend = flat_phase**3*(10-15*flat_phase+6*flat_phase**2)
                        # [archived branch] toe_task.desired = torch.lerp(toe_task.desired, flatten_task.desired, flat_blend)
                        # [archived branch] toe_task.metric = torch.lerp(toe_task.metric, flatten_task.metric, flat_blend)
                        # [archived branch] support_flatten_weights[:, patch_foot] = flat_blend
                    contact_patch_tasks[patch_foot]=toe_task
                    fabric.dynamics_tasks.last_requests.append(toe_task)
            # [archived branch] if control_time>=args.late_contact_stabilization_start:
                # [archived branch] from dataclasses import replace as replace_projection
                # [archived branch] contact_phase=min(1.,max(0.,(control_time-args.late_contact_stabilization_start)/.04))
                # [archived branch] contact_blend=contact_phase**3*(10-15*contact_phase+6*contact_phase**2)
                # [archived branch] projector.projection_config=replace_projection(projector.projection_config,
                    # [archived branch] contact_kp=args.contact_stabilization_kp+contact_blend*(args.late_contact_stabilization_kp-args.contact_stabilization_kp),
                    # [archived branch] contact_kd=args.contact_stabilization_kd+contact_blend*(args.late_contact_stabilization_kd-args.contact_stabilization_kd),
                    # [archived branch] contact_stabilization_mask=(1.,1.,1.,1.,1.,1.))
            output = projector.step(
                acc_fabric=qdd_fabric,
                mass_matrix=mass,
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
            # [archived branch] qdd_dc = qdd_fabric if args.g1_md else output.acc_dc
            qdd_dc = output.acc_dc
            # With task_requests the QP ignores this projected reference;
            # it remains useful for diagnostics and extracting command sync.
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
            # [archived branch] friction_G, friction_b = (
                # [archived branch] build_single_support_wrench_constraints(
                    # [archived branch] contact_mode=output.contact_mode,
                    # [archived branch] foot_rotation_world=_quat_xyzw_to_matrix(
                        # [archived branch] live_feet[..., 3:7]
                    # [archived branch] ),
                    # [archived branch] friction_coefficient=(
                        # [archived branch] acceleration_qp.friction_coefficient
                    # [archived branch] ),
                    # [archived branch] yaw_friction_coefficient=(
                        # [archived branch] acceleration_qp.yaw_friction_coefficient
                    # [archived branch] ),
                    # [archived branch] toe_length=acceleration_qp.toe_length,
                    # [archived branch] heel_length=acceleration_qp.heel_length,
                    # [archived branch] half_width=acceleration_qp.half_width,
                    # [archived branch] sole_depth=.035 if args.use_sole_wrench_geometry else 0.,
                    # [archived branch] coupled_yaw=args.reference_manager,
                    # [archived branch] maximum_normal_force=(
                        # [archived branch] acceleration_qp.maximum_normal_force
                    # [archived branch] ),
                # [archived branch] )
            # [archived branch] )
            (friction_G, friction_b) = build_single_support_wrench_constraints(
                contact_mode=output.contact_mode,
                foot_rotation_world=_quat_xyzw_to_matrix(live_feet[..., 3:7]),
                friction_coefficient=acceleration_qp.friction_coefficient,
                yaw_friction_coefficient=acceleration_qp.yaw_friction_coefficient,
                toe_length=acceleration_qp.toe_length,
                heel_length=acceleration_qp.heel_length,
                half_width=acceleration_qp.half_width,
                sole_depth=0.035,
                coupled_yaw=args.reference_manager,
                maximum_normal_force=acceleration_qp.maximum_normal_force,
            )
            command_acc_lower = -acceleration_limits.unsqueeze(0).clone()
            friction_blocks=split_foot_wrench_constraints(friction_G,friction_b)
            for patch_foot,patch in contact_patches.items():
                # Toe wrench coordinates are local to its fixed horizontal
                # frame at its material anchor. Each foot has its own patch.
                # [archived branch] if elapsed_motion_time>=args.contact_friction_refine_start:
                    # [archived branch] from inscribed_patch_wrench import observed_patch_constraints
                    # [archived branch] patch_G,patch_b=observed_patch_constraints(patch,
                        # [archived branch] _quat_xyzw_to_matrix(live_feet[:,patch_foot,3:]),
                        # [archived branch] acceleration_qp.friction_coefficient,acceleration_qp.maximum_normal_force,
                        # [archived branch] args.contact_friction_rays,
                        # [archived branch] minimum_normal_force=20. if not patch.polygon_support and int(output.contact_mode.item()) in (1,2) else 0.,
                        # [archived branch] yaw_limit=None if patch.polygon_support else acceleration_qp.yaw_friction_coefficient)
                    # [archived branch] block=patch_G.new_zeros(1,patch_G.shape[1],12)
                    # [archived branch] block[:,:,6*patch_foot:6*(patch_foot+1)]=patch_G
                    # [archived branch] friction_blocks[patch_foot]=(block,patch_b)
                    # [archived branch] continue
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
            # [archived branch] if elapsed_motion_time>=args.support_load_floor_start:
                # [archived branch] from support_load_floor import support_load_floor
                # [archived branch] load_phase=min(1.,max(0.,(elapsed_motion_time-args.support_load_floor_start)/.06))
                # [archived branch] load_blend=load_phase**3*(10-15*load_phase+6*load_phase**2)
                # [archived branch] load_G,load_b=support_load_floor(output.contact_mode,foot_normal_forces,
                    # [archived branch] args.support_load_floor*load_blend,acceleration_qp.maximum_normal_force,
                    # [archived branch] preferred_mode=controller_contact_plan if args.support_load_floor_transfer else None)
                # [archived branch] friction_G=torch.cat((friction_G,load_G),dim=1)
                # [archived branch] friction_b=torch.cat((friction_b,load_b),dim=1)

            command_acc_upper = acceleration_limits.unsqueeze(0).clone()
            # [archived branch] if args.direct_wbc:
                # Only a one-step prediction for physical limit checks/logging;
                # these values are not sent as joint position commands.
            joint_position_prediction = joint_position.clone()
            joint_velocity_prediction = joint_velocity.clone()
            # [archived branch] if args.acceleration_and_braking_limits and not args.fixed_joint_test and not args.direct_wbc:
                # [archived branch] try:
                    # [archived branch] command_acc_lower[:, 6:], command_acc_upper[:, 6:] = integration_acceleration_bounds(
                        # [archived branch] joint_position_desired, joint_velocity_desired,
                        # [archived branch] joint_position_limits[:, :, 0], joint_position_limits[:, :, 1],
                        # [archived branch] joint_command_speed_limit, acceleration_limits[6:], FABRIC_TIMESTEP,
                    # [archived branch] )
                # [archived branch] except ValueError:
                    # [archived branch] failure_path = args.output_path.with_suffix(".failure.npz")
                    # [archived branch] failure_path.parent.mkdir(parents=True, exist_ok=True)
                    # [archived branch] np.savez_compressed(failure_path,
                        # [archived branch] source_frame_count=np.asarray(getattr(args,'source_frame_count',0)),
                        # [archived branch] source_start_frame=np.asarray(motion_start_frame),
                        # [archived branch] controller_options=np.asarray(json.dumps(vars(args), default=str, sort_keys=True)),
                        # [archived branch] failure_reason=np.asarray("infeasible command integration bounds"),
                        # [archived branch] control_time=np.asarray(control_updates * FABRIC_TIMESTEP),
                        # [archived branch] q=q.detach().cpu().numpy(), qd=qd.detach().cpu().numpy(),
                        # [archived branch] command_position=joint_position_desired.cpu().numpy(),
                        # [archived branch] command_velocity=joint_velocity_desired.cpu().numpy(),
                        # [archived branch] joint_position_limits=joint_position_limits.cpu().numpy())
                    # [archived branch] print(f"[fail] command integration bounds; saved: {failure_path}", flush=True)
                    # [archived branch] raise
            prediction_seed_position = joint_position_prediction.clone()
            # Physical stopping-distance bounds are separate from command bounds.
            physical_lower = torch.full_like(qd, -float("inf"))
            physical_upper = torch.full_like(qd, float("inf"))
            damper_time = None
            # [archived branch] if args.joint_limit_time_constant is not None:
                # [archived branch] blend = min(1., max(0., (control_time-args.motion_warmup_duration)/.5))
                # [archived branch] if blend > 0:
                    # [archived branch] damper_time = args.joint_limit_time_constant * (3*blend**2-2*blend**3)
            full_joint_braking = args.joint_position_limits or args.acceleration_and_braking_limits
            physical_braking_active = full_joint_braking or elapsed_motion_time >= args.physical_braking_start
            try:
                # [archived branch] if physical_braking_active:
                # [archived branch] braking_joint_start = (12 if (args.physical_braking_upper_only or args.physical_braking_waist_only)
                                       # [archived branch] and not full_joint_braking else 0)
                braking_joint_start = 0
                # [archived branch] braking_joint_end = (15 if args.physical_braking_waist_only
                                     # [archived branch] and not full_joint_braking else joint_position.shape[1])
                braking_joint_end = joint_position.shape[1]
                braking_slice = slice(braking_joint_start, braking_joint_end)
                acceleration_slice = slice(6+braking_joint_start, 6+braking_joint_end)
                physical_lower[:, acceleration_slice], physical_upper[:, acceleration_slice] = measured_joint_acceleration_bounds(
                    joint_position[:,braking_slice], joint_velocity[:,braking_slice], joint_position_limits[:, braking_slice, 0],
                    joint_position_limits[:, braking_slice, 1], FABRIC_TIMESTEP, velocity_damper_time=damper_time,
                    barrier_bandwidth=args.joint_limit_barrier_bandwidth,
                    impact_recovery=args.joint_limit_impact_recovery)
            except ValueError as exc:
                save_reference_failure(args,exc,control_time,q,qd,foot_normal_forces,
                                       tracking_history,servo_history)
                raise
            # [archived branch] if physical_braking_active and not full_joint_braking and args.physical_braking_upper_only:
                # [archived branch] physical_lower[:,:18]=-float('inf')
                # [archived branch] physical_upper[:,:18]=float('inf')
            prediction_seed_velocity = joint_velocity_prediction.clone()
            # servo_bias, servo_acc_gain = integrated_servo_model(
                # joint_position_desired, joint_velocity_desired,
                # joint_position, joint_velocity, kp, kd, FABRIC_TIMESTEP,
            # )
            # Compatibility diagnostics: direct WBC has no integrated servo.
            servo_bias = torch.zeros_like(joint_velocity)
            servo_acc_gain = torch.zeros_like(joint_velocity)
            # [archived branch] qp_reference = qdd_dc if args.g1_md or args.fixed_joint_test or args.direct_wbc else synchronize_command_acceleration(
                # [archived branch] qdd_dc, joint_position_desired, joint_velocity_desired,
                # [archived branch] joint_position, joint_velocity,
                # [archived branch] stiffness=20. if args.fast_command_sync else 1.,
                # [archived branch] damping=8. if args.fast_command_sync else 2.,
                # [archived branch] limit=30. if args.fast_command_sync else 5.,
            # [archived branch] )
            qp_reference = qdd_dc
            wrench_reference = None
            wrench_weights = None
            # COM tracking alone need not unload the outgoing foot: contact
            # moments can maintain a different load split. Request transfer
            # here, without imposing this extra objective throughout startup
            # and swing. Release still requires measured load and balance.
            managed_load_transfer = (reference_manager is not None and reference_manager.phase == 'PREPARE_STEP')
            # [archived branch] if args.mocap_pipeline:
                # Ask only the outgoing measured contact to unload. The full
                # dynamics QP determines all other forces from the rigid-body equations.
                # [archived branch] event = reference_manager.pending_takeoff if managed_load_transfer and args.mocap_unload_weight > 0 else None
                # [archived branch] if event is None:
                    # [archived branch] direct_unload_event = None
                # [archived branch] else:
                    # [archived branch] if event != direct_unload_event:
                        # [archived branch] direct_unload_event, direct_unload_started = event, control_time
                    # [archived branch] phase = min(1., max(0., control_time-direct_unload_started))
                    # [archived branch] weight = (mocap_transition.unload_weight(control_time) if mocap_transition is not None
                              # [archived branch] else phase**3 * (10.-15.*phase+6.*phase**2))
                    # [archived branch] outgoing = 2-int(reference_manager.modes[event])
                    # [archived branch] wrench_reference = torch.zeros((1,12),device=device,dtype=q.dtype)
                    # [archived branch] wrench_weights = torch.zeros_like(wrench_reference)
                    # [archived branch] if args.mocap_unload_test:
                        # [archived branch] wrench_weights[:,6*outgoing+2] = weight
                    # [archived branch] else:
                        # [archived branch] wrench_weights[:,6*outgoing:6*outgoing+6] = weight
            retain_source_support = False
            # [archived branch] if (control_time>=args.retain_planned_support_start and int(output.contact_mode.item())==3
                    # [archived branch] and int(planned_contact_mode.item())==3 and measured_steps is not None):
                # [archived branch] pending_events=[event for event in measured_steps.events
                    # [archived branch] if event['takeoff_frame']>frame_position and (event['takeoff_frame']-frame_position)/motion_fps<=.15]
                # [archived branch] if pending_events:
                    # [archived branch] event=pending_events[0];outgoing=1-event['support']
                    # Existing ground contact is required; this force objective never creates contact rows.
                    # [archived branch] if float(foot_normal_forces[0,outgoing])>=10.:
                        # [archived branch] retain_source_support=True
                        # [archived branch] wrench_reference=torch.zeros((1,12),device=device,dtype=q.dtype)
                        # [archived branch] wrench_weights=torch.zeros_like(wrench_reference)
                        # [archived branch] wrench_reference[:,6*outgoing+2]=args.retain_planned_support_force
                        # [archived branch] wrench_weights[:,6*outgoing+2]=1.
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
                    _to_numpy(foot_normal_forces[0]),motion_fps,unload_intended_mode,horizon=args.anticipate_unload_horizon,hysteresis=args.anticipate_unload_hysteresis,allow_overdue=args.anticipate_unload_overdue)
                if anticipated:unload_intended_reason='anticipate_next_takeoff_after_load_hold'
            dynamic_unloading = (args.motion_start_in_motion and args.dynamic_unload_weight > 0
                and int(output.contact_mode.item()) == 3 and unload_intended_mode in (1,2))
            if dynamic_unloading:
                outgoing = 2-unload_intended_mode
                if direct_unload_event != outgoing:
                    direct_unload_event, direct_unload_started = outgoing, control_time
                unload_ramp=args.late_unload_ramp if elapsed_motion_time>=args.late_unload_start else .05
                phase = min(1.,max(0.,(control_time-direct_unload_started)/unload_ramp))
                blend = phase**3*(10.-15.*phase+6.*phase**2)
                wrench_reference = torch.zeros((1,12),device=device,dtype=q.dtype)
                wrench_weights = torch.zeros_like(wrench_reference)
                wrench_weights[:,6*outgoing:6*outgoing+6] = blend
            elif args.motion_start_in_motion:
                direct_unload_event = None
            # The new transfer has its own measured gate; its force objective
            # must stay active after the source scheduler has stopped advancing.
            # [archived branch] acceleration_qp.wrench_weight = (args.mocap_unload_weight if wrench_weights is not None else
                # [archived branch] .01 if support_switch is not None or source_step_number > 1 else 1e-4)
            acceleration_qp.wrench_weight = args.mocap_unload_weight if wrench_weights is not None else 0.0001
            if dynamic_unloading:
                acceleration_qp.wrench_weight = args.late_unload_weight if elapsed_motion_time>=args.late_unload_start else args.dynamic_unload_weight
            # [archived branch] if retain_source_support:acceleration_qp.wrench_weight=args.retain_planned_support_weight
            height_priority = (False)
            swing_com_priority = (False)
            com_priority_projection = None
            # [archived branch] priority_tasks=("center_of_mass",) if (args.balance_priority or height_priority or swing_com_priority or
                # [archived branch] (args.transfer_balance_priority and (transfer_reference is not None or
                 # [archived branch] (com_preview is not None and int(output.contact_mode.item()) == 3
                  # [archived branch] and reference_manager.phase in ("PREPARE_STEP","LANDING","SETTLE"))))) else ()
            priority_tasks = ('center_of_mass',)
            # [archived branch] priority_rows={"center_of_mass": (2,)} if height_priority and com_priority_projection is None else {}
            priority_rows = {}
            # [archived branch] if args.centroidal_balance_priority and 'center_of_mass' in priority_tasks:
                # [archived branch] priority_tasks += ('centroidal_momentum',)
            # [archived branch] if args.horizontal_balance_priority and 'center_of_mass' in priority_tasks:
                # [archived branch] priority_rows['center_of_mass']=(0,1)
            if ((args.swing_clearance_priority or args.swing_position_priority or args.swing_pose_priority)
                    and (com_preview is not None or args.motion_start_in_motion)
                    and int(output.contact_mode.item()) in (1, 2)):
                priority_tasks += ("feet_pose",)
                # [archived branch] if not args.swing_pose_priority:
                # [archived branch] priority_rows["feet_pose"] = (0, 1, 2, 6, 7, 8) if args.swing_position_priority else (2, 8)
                priority_rows['feet_pose'] = (0, 1, 2, 6, 7, 8)
            if 'feet_pose' in priority_tasks and elapsed_motion_time>=args.full_swing_pose_start:
                priority_rows['feet_pose']=tuple(range(12))
            # [archived branch] if elapsed_motion_time>=args.transition_swing_priority_start:
                # [archived branch] from transition_swing_task import activate_transition_swing
                # [archived branch] transition_rows=activate_transition_swing(fabric.dynamics_tasks.last_requests,
                    # [archived branch] int(output.contact_mode.item()),unload_intended_mode,foot_normal_forces[0])
                # [archived branch] if transition_rows is not None:
                    # [archived branch] priority_tasks+=("feet_pose",)
                    # [archived branch] priority_rows["feet_pose"]=transition_rows
            if priority_tasks and (args.upper_body_priority or args.whole_body_posture_priority):
                priority_tasks += ("body_orientation", "joint_posture")
                # [archived branch] priority_rows["joint_posture"] = tuple(range(0 if args.whole_body_posture_priority else 12,29))
                priority_rows['joint_posture'] = tuple(range(0, 29))
            # [archived branch] if priority_tasks and (args.body_orientation_priority or args.balance_body_priority):
                # [archived branch] priority_tasks += ("body_orientation", "source_pelvis_orientation")
            # [archived branch] if priority_tasks and args.landing_orientation_priority and int(output.contact_mode.item()) == 3:
                # [archived branch] for patch_foot,patch in contact_patches.items():
                    # [archived branch] if patch.flatten:
                        # Point/edge contacts expose real rotational freedom.
                        # Completing that rotation creates a bearing sole;
                        # hard contact rows and available moments stay unchanged.
                        # [archived branch] priority_tasks += (contact_patch_tasks[patch_foot].name,)
            # [archived branch] if elapsed_motion_time>=args.support_orientation_priority_start:
                # [archived branch] for support_request in contact_patch_tasks.values():
                    # [archived branch] priority_tasks += (support_request.name,)
                    # [archived branch] support_scale=args.support_orientation_metric_scale
                    # [archived branch] if args.support_orientation_roll_pitch_only:
                        # A point/edge contact may freely tilt. Keep those
                        # source-foot orientation rows in the first solve;
                        # yaw remains secondary during loaded support.
                        # [archived branch] priority_rows[support_request.name]=(0,1)
                        # [archived branch] tilt_phase=min(1.,max(0.,(elapsed_motion_time-args.support_orientation_priority_start)/.06))
                        # [archived branch] support_scale*=tilt_phase**3*(10-15*tilt_phase+6*tilt_phase**2)
                    # [archived branch] support_request.metric *= support_scale
            priority_levels = None
            # Body selection can overlap the whole-body/upper-body selection.
            # Each task must occur once in the hierarchy's partition.
            priority_tasks = tuple(dict.fromkeys(priority_tasks))
            # [archived branch] if priority_tasks and (args.separate_balance_priority or args.posture_last_priority):
                # [archived branch] remaining = list(priority_tasks)
                # [archived branch] levels = []
                # [archived branch] if args.separate_balance_priority and 'center_of_mass' in remaining:
                    # [archived branch] balance = (('center_of_mass','body_orientation','source_pelvis_orientation')
                               # [archived branch] if args.balance_body_priority else
                               # [archived branch] ('center_of_mass','centroidal_momentum')
                               # [archived branch] if args.centroidal_balance_priority else ('center_of_mass',))
                    # [archived branch] levels.append(balance)
                    # [archived branch] remaining = [name for name in remaining if name not in balance]
                # [archived branch] if args.posture_last_priority:
                    # [archived branch] remaining = [name for name in remaining if name != 'joint_posture']
                    # [archived branch] if 'joint_posture' not in priority_tasks:
                        # [archived branch] priority_tasks += ('joint_posture',)
                    # [archived branch] priority_rows.pop('joint_posture', None)
                # [archived branch] if remaining:
                    # [archived branch] levels.append(tuple(remaining))
                # [archived branch] if args.posture_last_priority:
                    # [archived branch] levels.append(('joint_posture',))
                # [archived branch] priority_levels = tuple(levels)
            # [archived branch] if support_switch is not None or source_step_number > 1:
                # [archived branch] priority_tasks = ("center_of_mass", "joint_posture")
                # [archived branch] priority_rows = {"joint_posture": tuple(range(12))}
                # [archived branch] priority_levels = (("center_of_mass",), ("joint_posture",))
            # [archived branch] if elapsed_motion_time>=args.stance_posture_secondary_start and 'joint_posture' in priority_tasks:
                # Keep the complete posture task in the final solve. The first
                # stage must let loaded legs realize balance/contact tracking.
                # [archived branch] loaded_mode=int(output.contact_mode.item())
                # [archived branch] primary_joint_rows=list(range(12,29))
                # [archived branch] if not loaded_mode&1:primary_joint_rows+=list(range(6))
                # [archived branch] if not loaded_mode&2:primary_joint_rows+=list(range(6,12))
                # [archived branch] priority_rows['joint_posture']=tuple(sorted(primary_joint_rows))
            # [archived branch] if elapsed_motion_time>=args.leg_posture_secondary_start and 'joint_posture' in priority_tasks:
                # Foot-space tracking and fixed contact anchors determine the
                # legs at the current floating-base pose. Keep source leg
                # posture as a secondary objective rather than locking its
                # incompatible source-base solution in the same first stage.
                # [archived branch] priority_rows['joint_posture']=tuple(range(12,29))
            from fabrics_sim.reference.source_priority import source_swing_priority
            # [archived branch] source_priority = source_swing_priority(args.source_swing_position_priority or args.source_swing_pose_priority,
                # [archived branch] source_step_rollout is not None, support_switch is not None,
                # [archived branch] source_steps_finished, int(output.contact_mode.item()), pose=args.source_swing_pose_priority,
                # [archived branch] body_priority="balance" if args.balance_body_priority else "swing" if args.body_orientation_priority else None)
            source_priority = source_swing_priority(
                args.source_swing_position_priority or args.source_swing_pose_priority,
                source_step_rollout is not None,
                support_switch is not None,
                source_steps_finished,
                int(output.contact_mode.item()),
                pose=args.source_swing_pose_priority,
                body_priority=None,
            )
            if source_priority is not None:
                priority_tasks, priority_rows, priority_levels = source_priority
                com_priority_projection = None
            # [archived branch] if control_time>=args.swing_balance_priority_start:
                # [archived branch] priority_tasks=('feet_pose',) if args.swing_only_priority else ('center_of_mass','feet_pose')
                # [archived branch] priority_rows={'feet_pose':tuple(range(12)) if elapsed_motion_time>=args.full_swing_pose_start else (0,1,2,6,7,8)}
                # [archived branch] priority_levels=None
                # [archived branch] com_priority_projection=None
            # [archived branch] if elapsed_motion_time>=args.late_momentum_priority_start:
                # [archived branch] priority_tasks=tuple(dict.fromkeys(priority_tasks+('centroidal_momentum',)))
                # [archived branch] priority_levels=None
                # [archived branch] if args.late_momentum_roll_only:
                    # [archived branch] priority_rows['centroidal_momentum']=(0,)
                # [archived branch] for request in fabric.dynamics_tasks.last_requests:
                    # [archived branch] if request.name=='centroidal_momentum':
                        # [archived branch] scale=args.late_momentum_metric_scale
                        # [archived branch] if args.late_momentum_feedback_damping is not None:
                            # [archived branch] if not math.isfinite(args.late_momentum_feedback_damping) or args.late_momentum_feedback_damping<=0:
                                # [archived branch] raise ValueError('Momentum feedback damping must be finite and positive')
                            # [archived branch] dynamics=fabric.dynamics_tasks
                            # [archived branch] if args.late_momentum_zero_target:
                                # [archived branch] request.desired=-args.late_momentum_feedback_damping*dynamics.last_snapshot['momentum']
                            # [archived branch] else:
                                # [archived branch] request.desired=dynamics.momentum_rate_reference-args.late_momentum_feedback_damping*(dynamics.last_snapshot['momentum']-dynamics.momentum_reference)
                            # [archived branch] mp_phase=min(1.,max(0.,(elapsed_motion_time-args.late_momentum_priority_start)/.06))
                            # [archived branch] scale*=mp_phase**3*(10-15*mp_phase+6*mp_phase**2)
                        # [archived branch] if args.late_momentum_total_rate_limit is not None:
                            # [archived branch] if not math.isfinite(args.late_momentum_total_rate_limit) or args.late_momentum_total_rate_limit<=0:
                                # [archived branch] raise ValueError('Momentum rate limit must be finite and positive')
                            # [archived branch] request.desired=request.desired.clamp(-args.late_momentum_total_rate_limit,args.late_momentum_total_rate_limit)
                        # [archived branch] request.metric*=scale
            # [archived branch] if elapsed_motion_time>=args.late_pelvis_priority_start:
                # [archived branch] priority_tasks=tuple(dict.fromkeys(priority_tasks+('source_pelvis_orientation',)))
                # [archived branch] priority_levels=None
                # [archived branch] if args.late_pelvis_yaw_only:
                    # [archived branch] priority_rows['source_pelvis_orientation']=(2,)
                # [archived branch] if args.late_pelvis_metric_scale!=1.:
                    # [archived branch] if not math.isfinite(args.late_pelvis_metric_scale) or args.late_pelvis_metric_scale<=0:
                        # [archived branch] raise ValueError('Late pelvis metric scale must be finite and positive')
                    # [archived branch] for request in fabric.dynamics_tasks.last_requests:
                        # [archived branch] if request.name=='source_pelvis_orientation':
                            # [archived branch] request.metric*=args.late_pelvis_metric_scale
            # [archived branch] if elapsed_motion_time>=args.late_balance_hierarchy_start:
                # [archived branch] balance_names=(('source_pelvis_orientation',) if args.late_balance_hierarchy=='body'
                    # [archived branch] else ('center_of_mass',) if args.late_balance_hierarchy in ('y','xy','xyz')
                    # [archived branch] else ('center_of_mass','body_orientation','source_pelvis_orientation'))
                # [archived branch] priority_tasks=tuple(dict.fromkeys(priority_tasks+balance_names))
                # [archived branch] if 'center_of_mass' in balance_names:
                    # [archived branch] priority_rows['center_of_mass'] = ((0,1,2) if args.late_balance_hierarchy in ('xyz','xyz-body')
                        # [archived branch] else (1,) if args.late_balance_hierarchy == 'y' else (0,1))
                # [archived branch] remaining=tuple(n for n in priority_tasks if n not in balance_names)
                # [archived branch] priority_levels=(balance_names,remaining) if remaining else (balance_names,)
            if args.joint_limit_avoidance_weight > 0:
                avoidance_request=joint_limit_avoidance_request(
                    joint_position, joint_velocity, joint_position_limits[:, :, 0],
                    joint_position_limits[:, :, 1], args.joint_limit_avoidance_weight)
                # [archived branch] if elapsed_motion_time>=args.joint_limit_priority_start:
                    # A full-rank first stage otherwise leaves this braking task
                    # no acceleration freedom in the lower-priority solve.
                    # [archived branch] brake_phase=min(1.,max(0.,(elapsed_motion_time-args.joint_limit_priority_start)/.06))
                    # [archived branch] avoidance_request.metric*=brake_phase**3*(10-15*brake_phase+6*brake_phase**2)
                    # [archived branch] if args.joint_limit_priority_upper_only:
                        # [archived branch] avoidance_request.metric[:, 12:, 12:] *= args.joint_limit_priority_scale
                    # [archived branch] else:
                        # [archived branch] avoidance_request.metric *= args.joint_limit_priority_scale
                    # [archived branch] priority_tasks=tuple(dict.fromkeys(priority_tasks+(avoidance_request.name,)))
                    # [archived branch] if args.joint_limit_priority_upper_only:
                        # [archived branch] priority_rows[avoidance_request.name]=tuple(range(12,29))
                    # [archived branch] if priority_levels is not None:
                        # [archived branch] priority_levels=(priority_levels[0]+(avoidance_request.name,),)+priority_levels[1:]
                fabric.dynamics_tasks.last_requests.append(avoidance_request)
            # [archived branch] if args.joint_velocity_limits:
            speed_C,speed_d = joint_velocity_constraints(joint_velocity,joint_velocity_limits,FABRIC_TIMESTEP)
            task_acceleration_constraints = ((speed_C,speed_d) if task_acceleration_constraints is None else
                (torch.cat((task_acceleration_constraints[0],speed_C),dim=1),
                 torch.cat((task_acceleration_constraints[1],speed_d),dim=1)))
            knee_guard_diagnostic = {}
            # [archived branch] if args.support_knee_min_angle is not None:
                # [archived branch] from g1_support_knee import support_knee_constraints
                # [archived branch] knee_guard_active = elapsed_motion_time >= args.support_knee_guard_start
                # [archived branch] knee_C, knee_d, knee_fields = support_knee_constraints(
                    # [archived branch] joint_position, joint_velocity,
                    # [archived branch] current_contact_active if knee_guard_active else torch.zeros_like(current_contact_active),
                    # [archived branch] minimum_angle=args.support_knee_min_angle,
                    # [archived branch] bandwidth=args.support_knee_guard_bandwidth,
                    # [archived branch] knee_indices=tuple(G1_JOINT_NAMES.index(f'{side}_knee_joint') for side in ('left', 'right')))
                # [archived branch] knee_guard_diagnostic = {key: _to_numpy(value).copy() for key, value in knee_fields.items()}
                # [archived branch] if knee_guard_active:
                    # [archived branch] task_acceleration_constraints = ((knee_C, knee_d) if task_acceleration_constraints is None else
                        # [archived branch] (torch.cat((task_acceleration_constraints[0], knee_C), dim=1),
                         # [archived branch] torch.cat((task_acceleration_constraints[1], knee_d), dim=1)))
            # [archived branch] if args.g1_md:
                # [archived branch] from fabrics_sim.solver.fabric_mit import compose_task_policy
                # [archived branch] try:
                    # [archived branch] qdd_fabric, composed_policy = compose_task_policy(
                        # [archived branch] fabric.dynamics_tasks.last_requests, qd,
                        # [archived branch] fabric.dynamics_tasks.config["regularization"])
                # [archived branch] except (ValueError, RuntimeError, FloatingPointError) as exc:
                    # [archived branch] save_reference_failure(args, exc, control_time, q, qd, foot_normal_forces,
                                           # [archived branch] tracking_history, servo_history)
                    # [archived branch] raise
                # [archived branch] qdd_dc = qdd_fabric
                # Preserve integration history, but continuously pull command
                # error back toward measurements. The existing QP correction
                # map excludes this synchronization from physical acceleration.
                # [archived branch] qp_reference = synchronize_command_acceleration(
                    # [archived branch] qdd_fabric, joint_position_desired, joint_velocity_desired,
                    # [archived branch] joint_position, joint_velocity, stiffness=20., damping=8., limit=30.)
            # Diagnostic: independently bound vertical swing feedback so that
            # horizontal lag cannot consume the entire XYZ feedback allowance.
            feedback_phase = min(1., max(0., (elapsed_motion_time-args.vertical_feedback_start)/.02))
            feedback_blend = feedback_phase**3*(10.-15.*feedback_phase+6.*feedback_phase**2)
            if feedback_blend > 0:
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
            if horizontal_blend>0:
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
            if flight_feet and flight_blend > 0:
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
            # late_phase=min(1.,max(0.,(elapsed_motion_time-args.late_leg_feedback_start)/.04))
            # late_blend=late_phase**3*(10-15*late_phase+6*late_phase**2)
            # if late_blend>0:
                # selected=([0,1] if args.late_leg_feedback_side=='both' else [1])
                # selected=[f for f in selected if args.late_leg_feedback_stance or f in flight_feet]
                # rows=[j for f in selected for j in range(6*f,6*f+6)]
                # if rows:
                    # joint_params=fabric.fabric_params['joint_posture_attractor']
                    # for request in fabric.dynamics_tasks.last_requests:
                        # if request.name!='joint_posture':continue
                        # rv,ra=fabric.dynamics_tasks.references(fabric.fabrics_features['joint_posture']['joint_posture_attractor'],request.desired)
                        # feedback=-joint_params['stiffness']*(q[:,7:]-fabric.joint_posture_taskmap.position_target)-joint_params['damping']*(qd[:,6:]-rv)
                        # new=ra[:,rows]+feedback[:,rows].clamp(-args.late_leg_feedback_limit,args.late_leg_feedback_limit)
                        # request.desired[:,rows]+=late_blend*(new-request.desired[:,rows])
            # phase=min(1.,max(0.,(elapsed_motion_time-args.waist_feedback_start)/.06));waist_blend=phase**3*(10-15*phase+6*phase**2)
            # if waist_blend>0:
                # joint_params=fabric.fabric_params['joint_posture_attractor']
                # for request in fabric.dynamics_tasks.last_requests:
                    # if request.name!='joint_posture':continue
                    # rv,ra=fabric.dynamics_tasks.references(fabric.fabrics_features['joint_posture']['joint_posture_attractor'],request.desired)
                    # feedback=-joint_params['stiffness']*(q[:,7:]-fabric.joint_posture_taskmap.position_target)-joint_params['damping']*(qd[:,6:]-rv)
                    # rows=list(range(12,15));new=ra[:,rows]+feedback[:,rows].clamp(-args.waist_feedback_limit,args.waist_feedback_limit)
                    # request.desired[:,rows]+=waist_blend*(new-request.desired[:,rows])
                    # for row in rows:request.metric[:,row,row]*=1+(args.waist_metric_scale-1)*waist_blend
            # com_feedback_phase=min(1.,max(0.,(elapsed_motion_time-args.com_feedback_start)/.06))
            # com_feedback_blend=com_feedback_phase**3*(10-15*com_feedback_phase+6*com_feedback_phase**2)
            # if com_feedback_blend>0:
                # if not math.isfinite(args.com_feedback_limit) or args.com_feedback_limit<=0 or not math.isfinite(args.com_feedback_gain_scale) or args.com_feedback_gain_scale<=0:
                    # raise ValueError('COM feedback parameters must be finite and positive')
                # dynamics=fabric.dynamics_tasks;state=dynamics.last_snapshot;params=fabric.fabric_params['center_of_mass_attractor']
                # for request in dynamics.last_requests:
                    # if request.name!='center_of_mass':continue
                    # rv,ra=dynamics.references(fabric.fabrics_features['center_of_mass']['center_of_mass_attractor'],request.desired)
                    # K=q.new_tensor(params['stiffness'])*args.com_feedback_gain_scale
                    # D=q.new_tensor(params['damping'])*math.sqrt(args.com_feedback_gain_scale)
                    # error=state['com']-fabric.center_of_mass_taskmap.position_target
                    # velocity=(state['com_jac']@qd[...,None]).squeeze(-1)-rv
                    # desired=ra+(-K*error-D*velocity).clamp(-args.com_feedback_limit,args.com_feedback_limit)
                    # axes=slice(0,2) if args.com_feedback_axes=='xy' else slice(None)
                    # request.desired[:,axes]+=com_feedback_blend*(desired[:,axes]-request.desired[:,axes])
            com_phase=min(1.,max(0.,(elapsed_motion_time-args.com_feedback_metric_start)/.06))
            com_blend=com_phase**3*(10-15*com_phase+6*com_phase**2)
            if com_blend>0 and args.com_feedback_metric_scale!=1:
                for request in fabric.dynamics_tasks.last_requests:
                    if request.name=='center_of_mass':
                        request.metric*=1+(args.com_feedback_metric_scale-1)*com_blend
            global_phase=min(1.,max(0.,(elapsed_motion_time-args.global_posture_feedback_start)/.06))
            global_blend=global_phase**3*(10-15*global_phase+6*global_phase**2)
            if global_blend>0:
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
            # [archived branch] if elapsed_motion_time >= args.stance_posture_metric_start:
                # [archived branch] stance_phase = min(1., (elapsed_motion_time-args.stance_posture_metric_start)
                                   # [archived branch] / args.stance_posture_metric_ramp)
                # [archived branch] stance_blend = stance_phase**3*(10.-15.*stance_phase+6.*stance_phase**2)
                # [archived branch] stance_posture_scale += (args.stance_posture_metric_scale-1.)*stance_blend
                # [archived branch] for request in fabric.dynamics_tasks.last_requests:
                    # [archived branch] if request.name == 'joint_posture':
                        # [archived branch] request.metric = scale_stance_posture_metric(
                            # [archived branch] request.metric, current_contact_active, stance_posture_scale)
            upper_phase=min(1.,max(0.,(elapsed_motion_time-args.upper_posture_metric_start)/.06))
            upper_blend=upper_phase**3*(10-15*upper_phase+6*upper_phase**2)
            if upper_blend>0 and args.upper_posture_metric_scale!=1:
                for request in fabric.dynamics_tasks.last_requests:
                    if request.name=='joint_posture':
                        for joint_index in range(12,29):
                            request.metric[:,joint_index,joint_index]*=1+(args.upper_posture_metric_scale-1)*upper_blend
            late_foot_phase=min(1.,max(0.,(elapsed_motion_time-args.late_foot_metric_start)/.04))
            late_foot_blend=late_foot_phase**3*(10-15*late_foot_phase+6*late_foot_phase**2)
            if late_foot_blend>0 and args.late_foot_metric_scale!=1:
                for request in fabric.dynamics_tasks.last_requests:
                    if request.name=='feet_pose':request.metric*=1+(args.late_foot_metric_scale-1)*late_foot_blend
            body_feedback_phase=min(1.,max(0.,(elapsed_motion_time-args.body_feedback_start)/.06))
            body_feedback_blend=body_feedback_phase**3*(10-15*body_feedback_phase+6*body_feedback_phase**2)
            if body_feedback_blend>0:
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
            if body_blend>0 and args.late_body_angular_metric_scale!=1:
                if not math.isfinite(args.late_body_angular_metric_scale) or args.late_body_angular_metric_scale<=0:
                    raise ValueError('Late body angular metric must be finite and positive')
                for request in fabric.dynamics_tasks.last_requests:
                    if request.name=='body_orientation':
                        request.metric*=1+(args.late_body_angular_metric_scale-1)*body_blend
            angular_phase=min(1.,max(0.,(elapsed_motion_time-args.late_foot_angular_metric_start)/.06))
            angular_blend=angular_phase**3*(10-15*angular_phase+6*angular_phase**2)
            # [archived branch] if args.late_foot_angular_metric_end < args.late_foot_angular_metric_start:
                # [archived branch] raise ValueError('Foot orientation weight window must end after its start')
            angular_end_phase=min(1.,max(0.,(elapsed_motion_time-args.late_foot_angular_metric_end)/.06))
            angular_blend*=1-angular_end_phase**3*(10-15*angular_end_phase+6*angular_end_phase**2)
            if angular_blend>0 and args.late_foot_angular_metric_scale!=1:
                if not math.isfinite(args.late_foot_angular_metric_scale) or args.late_foot_angular_metric_scale<=0:
                    raise ValueError('Late foot angular metric must be finite and positive')
                for request in fabric.dynamics_tasks.last_requests:
                    if request.name=='feet_pose':
                        angular_rows=(3,4,5,9,10,11) if args.late_foot_angular_metric_side=='both' else ((3,4,5) if args.late_foot_angular_metric_side=='left' else (9,10,11))
                        for row in angular_rows:
                            request.metric[:,row,row]*=1+(args.late_foot_angular_metric_scale-1)*angular_blend
            # Bound the soft total task request, including feedforward spikes.
            # Unified reference q/v/a and physical acceleration constraints stay intact.
            total_phase=min(1.,max(0.,(elapsed_motion_time-args.posture_total_accel_start)/.06))
            total_blend=total_phase**3*(10-15*total_phase+6*total_phase**2)
            if total_blend>0:
                if not math.isfinite(args.posture_total_accel_limit) or args.posture_total_accel_limit<=0:
                    raise ValueError('Posture total acceleration limit must be finite and positive')
                for request in fabric.dynamics_tasks.last_requests:
                    if request.name=='joint_posture':
                        bounded=request.desired.clamp(-args.posture_total_accel_limit,args.posture_total_accel_limit)
                        request.desired+=total_blend*(bounded-request.desired)
            if (elapsed_motion_time>=args.tracking_tube_start and elapsed_motion_time<args.tracking_tube_end):
                from tracking_tube import tracking_tube_constraints
                tube_q,tube_v,tube_a=tube_source_sampler.kinematics(frame_position,frame_rate,frame_acceleration)
                joint_tube_rate=args.tracking_tube_joint_bandwidth
                # [archived branch] if elapsed_motion_time>=args.late_joint_tube_start:
                    # [archived branch] if not math.isfinite(args.late_joint_tube_bandwidth) or args.late_joint_tube_bandwidth<=0:
                        # [archived branch] raise ValueError('Late joint tube bandwidth must be finite and positive')
                    # [archived branch] tube_phase=min(1.,max(0.,(elapsed_motion_time-args.late_joint_tube_start)/.06))
                    # [archived branch] tube_blend=tube_phase**3*(10-15*tube_phase+6*tube_phase**2)
                    # [archived branch] original_rate=args.tracking_tube_joint_bandwidth or args.tracking_tube_bandwidth
                    # [archived branch] joint_tube_rate=original_rate+tube_blend*(args.late_joint_tube_bandwidth-original_rate)
                tube_rate=args.tracking_tube_bandwidth
                orientation_tube_rate=args.tracking_tube_orientation_bandwidth
                # [archived branch] if elapsed_motion_time>=args.late_tracking_tube_start:
                    # [archived branch] if not math.isfinite(args.late_tracking_tube_bandwidth) or args.late_tracking_tube_bandwidth<=0:
                        # [archived branch] raise ValueError('Late tracking tube bandwidth must be positive')
                    # [archived branch] phase=min(1.,max(0.,(elapsed_motion_time-args.late_tracking_tube_start)/.1))
                    # [archived branch] blend=phase**3*(10.-15.*phase+6.*phase**2)
                    # [archived branch] tube_rate+=(args.late_tracking_tube_bandwidth-tube_rate)*blend
                    # [archived branch] joint_tube_rate=(joint_tube_rate or args.tracking_tube_bandwidth)*(1-blend)+args.late_tracking_tube_bandwidth*blend
                    # [archived branch] orientation_tube_rate=(orientation_tube_rate or args.tracking_tube_bandwidth)*(1-blend)+args.late_tracking_tube_bandwidth*blend
                # [archived branch] tube_C,tube_d,tube_diagnostics=tracking_tube_constraints(q,qd,tube_q,tube_v,tube_a,bandwidth=tube_rate,joint_bandwidth=joint_tube_rate,orientation_bandwidth=orientation_tube_rate,
                    # [archived branch] position_radius=(args.position_tube_radius if elapsed_motion_time>=args.position_tube_start else .195),
                    # [archived branch] orientation_radius=(args.orientation_tube_radius
                        # [archived branch] if elapsed_motion_time>=args.orientation_tube_start else .34))
                (tube_C, tube_d, tube_diagnostics) = tracking_tube_constraints(
                    q,
                    qd,
                    tube_q,
                    tube_v,
                    tube_a,
                    bandwidth=tube_rate,
                    joint_bandwidth=joint_tube_rate,
                    orientation_bandwidth=orientation_tube_rate,
                    position_radius=args.position_tube_radius if elapsed_motion_time >= args.position_tube_start else 0.195,
                    orientation_radius=0.34,
                )
                # [archived branch] if elapsed_motion_time>=args.position_tube_end:
                    # Keep physical constraints and original-source validation;
                    # only this auxiliary acceleration barrier is neutralized.
                    # [archived branch] tube_C=tube_C.clone();tube_d=tube_d.clone()
                    # [archived branch] tube_C[:,1]=0.;tube_d[:,1]=0.
                # [archived branch] if elapsed_motion_time>=args.orientation_tube_end:
                    # Preserve the three-row recovery interface. This neutralizes
                    # only the auxiliary orientation CBF, not any physical row
                    # or the independent original-source fidelity auditor.
                    # [archived branch] tube_C=tube_C.clone();tube_d=tube_d.clone()
                    # [archived branch] tube_C[:,2]=0.;tube_d[:,2]=0.
                task_acceleration_constraints=((tube_C,tube_d) if task_acceleration_constraints is None else
                    (torch.cat((task_acceleration_constraints[0],tube_C),dim=1),torch.cat((task_acceleration_constraints[1],tube_d),dim=1)))
            acceleration_qp.priority_wrench_reference = (args.tracking_profile is not None
                and (retain_source_support or (dynamic_unloading and control_time >= .92)))
            # [archived branch] if elapsed_motion_time>=args.centroidal_preview_start:
                # [archived branch] if 'late_centroidal_preview' not in locals():
                    # [archived branch] if args.centroidal_preview_walking:
                        # [archived branch] from walking_centroidal_preview import CentroidalPreview
                    # [archived branch] elif args.centroidal_preview_persist_stance:
                        # [archived branch] from persistent_centroidal_preview import CentroidalPreview
                    # [archived branch] else:
                        # [archived branch] from late_centroidal_preview import CentroidalPreview
                    # [archived branch] late_centroidal_preview=CentroidalPreview(fabric,motion_sampler,_to_numpy(motion_planned_modes),motion_fps)
                # [archived branch] if args.centroidal_preview_walking:
                    # [archived branch] late_centroidal_preview.reference_clock=lambda offset: (
                        # [archived branch] motion_start_frame+motion_fps*args.motion_speed*smooth_clock(wall_motion_time+offset,args.clock_transition_start,args.clock_transition_ramp,args.clock_final_speed,args.clock_second_start,args.clock_second_ramp,args.clock_second_speed,args.clock_third_start,args.clock_third_ramp,args.clock_third_speed)[0],
                        # [archived branch] motion_fps*args.motion_speed*smooth_clock(wall_motion_time+offset,args.clock_transition_start,args.clock_transition_ramp,args.clock_final_speed,args.clock_second_start,args.clock_second_ramp,args.clock_second_speed,args.clock_third_start,args.clock_third_ramp,args.clock_third_speed)[1],
                        # [archived branch] motion_fps*args.motion_speed*smooth_clock(wall_motion_time+offset,args.clock_transition_start,args.clock_transition_ramp,args.clock_final_speed,args.clock_second_start,args.clock_second_ramp,args.clock_second_speed,args.clock_third_start,args.clock_third_ramp,args.clock_third_speed)[2])
                # [archived branch] try:
                    # [archived branch] preview_acc,preview_rate=late_centroidal_preview.update(control_time,frame_position,q,qd,
                        # [archived branch] live_feet,int(output.contact_mode.item()),contact_patches,friction_G,friction_b)
                # [archived branch] except RuntimeError as exc:
                    # [archived branch] args.output_path.with_suffix('.preview.json').write_text(json.dumps(late_centroidal_preview.records,indent=2)+'\n')
                    # [archived branch] save_reference_failure(args,exc,control_time,q,qd,foot_normal_forces,tracking_history,servo_history)
                    # [archived branch] raise
                # [archived branch] preview_phase=min(1.,max(0.,(elapsed_motion_time-args.centroidal_preview_start)/.1))
                # [archived branch] preview_blend=preview_phase**3*(10-15*preview_phase+6*preview_phase**2)
                # [archived branch] for request in fabric.dynamics_tasks.last_requests:
                    # [archived branch] if request.name=='center_of_mass':
                        # [archived branch] axes=slice(0,2) if args.centroidal_preview_walking else slice(None)
                        # [archived branch] request.desired[:,axes]=(1-preview_blend)*request.desired[:,axes]+preview_blend*q.new_tensor(preview_acc)[None,axes]
                    # [archived branch] if request.name=='centroidal_momentum' and args.centroidal_preview_momentum:
                        # [archived branch] request.desired[:]=(1-preview_blend)*request.desired+preview_blend*q.new_tensor(preview_rate)[None]
                # [archived branch] if args.centroidal_preview_momentum:
                    # [archived branch] priority_tasks=tuple(dict.fromkeys(priority_tasks+('centroidal_momentum',)))
                    # [archived branch] priority_levels=None
                # [archived branch] if control_updates%10==0:
                    # [archived branch] args.output_path.with_suffix('.preview.json').write_text(json.dumps(late_centroidal_preview.records,indent=2)+'\n')
            # [archived branch] if elapsed_motion_time>=args.minimum_support_acceleration_start and int(output.contact_mode.item())!=0 and float(foot_normal_forces.sum())>=20.:
                # [archived branch] support_phase=min(1.,max(0.,(elapsed_motion_time-args.minimum_support_acceleration_start)/.06))
                # [archived branch] support_blend=support_phase**3*(10-15*support_phase+6*support_phase**2)
                # [archived branch] if not math.isfinite(args.minimum_support_acceleration) or not -9.81<args.minimum_support_acceleration<=0:
                    # [archived branch] raise ValueError('Minimum support acceleration must be in (-9.81,0]')
                # [archived branch] if not math.isfinite(args.minimum_support_acceleration_metric_scale) or args.minimum_support_acceleration_metric_scale<=0:
                    # [archived branch] raise ValueError('Support acceleration metric scale must be positive')
                # [archived branch] for request in fabric.dynamics_tasks.last_requests:
                    # [archived branch] if request.name=='center_of_mass':
                        # [archived branch] supported=request.desired[:,2].clamp_min(args.minimum_support_acceleration)
                        # [archived branch] request.desired[:,2]+=support_blend*(supported-request.desired[:,2])
                        # [archived branch] request.metric[:,2,2]*=1+(args.minimum_support_acceleration_metric_scale-1)*support_blend
            # [archived branch] if elapsed_motion_time>=args.configuration_tracking_start:
                # [archived branch] if not unified_reference or args.configuration_tracking_ramp<=0 or not math.isfinite(args.configuration_tracking_ramp):
                    # [archived branch] raise ValueError('Configuration tracking requires a unified reference and a positive finite ramp')
                # [archived branch] from configuration_tracking import configuration_tracking_request
                # [archived branch] configuration_request=configuration_tracking_request(q,qd,whole_body_reference,fabric.dynamics_tasks.last_requests,args.configuration_angular_weight)
                # [archived branch] configuration_phase=min(1.,max(0.,(elapsed_motion_time-args.configuration_tracking_start)/args.configuration_tracking_ramp))
                # [archived branch] configuration_blend=configuration_phase**3*(10-15*configuration_phase+6*configuration_phase**2)
                # [archived branch] if configuration_blend>0:
                    # [archived branch] for request in fabric.dynamics_tasks.last_requests:
                        # [archived branch] if request.name in priority_tasks and not (args.configuration_keep_feet and request.name=='feet_pose'):
                            # [archived branch] request.metric*=1-configuration_blend
                    # [archived branch] configuration_request.metric*=configuration_blend
                    # [archived branch] fabric.dynamics_tasks.last_requests.append(configuration_request)
                    # [archived branch] priority_tasks=tuple(dict.fromkeys(priority_tasks+(configuration_request.name,)))
                    # [archived branch] priority_levels=None
            if elapsed_motion_time>=args.tracking_margin_start:
                from tracking_margin import margin_weights,allocate_margin_weights
                margin_q,margin_v,_=tube_source_sampler.kinematics(frame_position,frame_rate,frame_acceleration)
                margin_scale,margin_risk=margin_weights(q,qd,margin_q,margin_v,
                    reaction_time=args.tracking_margin_reaction_time,maximum=args.tracking_margin_maximum)
                if 'filtered_margin_scale' not in locals():filtered_margin_scale=torch.ones_like(margin_scale)
                filtered_margin_scale+=(1-math.exp(-FABRIC_TIMESTEP/.025))*(margin_scale-filtered_margin_scale)
                margin_phase=min(1.,max(0.,(elapsed_motion_time-args.tracking_margin_start)/.1))
                allocate_margin_weights(fabric.dynamics_tasks.last_requests,filtered_margin_scale,
                    margin_phase**3*(10-15*margin_phase+6*margin_phase**2))
            # [archived branch] if elapsed_motion_time>=args.pelvis_orientation_replace_start:
                # [archived branch] if not unified_reference or not math.isfinite(args.pelvis_orientation_replace_ramp) or args.pelvis_orientation_replace_ramp<=0:
                    # [archived branch] raise ValueError('Pelvis orientation replacement requires unified references and a positive ramp')
                # [archived branch] from pelvis_orientation_allocation import replace_torso_weight
                # [archived branch] orientation_phase=min(1.,max(0.,(elapsed_motion_time-args.pelvis_orientation_replace_start)/args.pelvis_orientation_replace_ramp))
                # [archived branch] orientation_blend=orientation_phase**3*(10-15*orientation_phase+6*orientation_phase**2)
                # [archived branch] if orientation_blend>0:
                    # [archived branch] replace_torso_weight(fabric.dynamics_tasks.last_requests,orientation_blend)
                    # [archived branch] priority_tasks=tuple(dict.fromkeys(priority_tasks+('source_pelvis_orientation',)))
                    # [archived branch] priority_levels=None
            root_task_weight = 0.
            # [archived branch] if elapsed_motion_time > args.root_position_task_start:
                # [archived branch] root_phase = min(1., (elapsed_motion_time-args.root_position_task_start)
                                 # [archived branch] / args.root_position_task_ramp)
                # [archived branch] root_task_weight = args.root_position_task_weight*root_phase**3*(10-15*root_phase+6*root_phase**2)
                # [archived branch] fabric.dynamics_tasks.last_requests.append(root_position_request(
                    # [archived branch] q, qd, reference_q, reference_v, reference_a, root_task_weight))
                # [archived branch] priority_tasks = tuple(dict.fromkeys(priority_tasks+('root_position',)))
                # [archived branch] if priority_levels is not None:
                    # [archived branch] priority_levels = (priority_levels[0]+('root_position',),)+priority_levels[1:]
            if not fabric.joint_posture_enabled:
                # Existing motion profiles may still select posture priority.
                # Remove its rows and any empty stages before the QP solve.
                priority_tasks = tuple(name for name in priority_tasks if name != 'joint_posture')
                priority_rows.pop('joint_posture', None)
                if priority_levels is not None:
                    priority_levels = tuple(
                        tuple(name for name in level if name != 'joint_posture')
                        for level in priority_levels)
                    priority_levels = tuple(level for level in priority_levels if level) or None
            # shoulder_weight = shoulder_roll_multiplier(
                # control_time, args.shoulder_roll_weight_start,
                # args.shoulder_roll_weight_ramp, args.shoulder_roll_weight_scale) if fabric.joint_posture_enabled else 1.
            # if shoulder_weight > 1 and shoulder_guard is None:
                # apply_shoulder_roll_weights(fabric.dynamics_tasks.last_requests,
                                            # G1_JOINT_NAMES, shoulder_weight, args.shoulder_roll_weight_side)
            secondary_task_blend = 0.
            # [archived branch] if elapsed_motion_time > args.weighted_tasks_start:
                # [archived branch] if priority_levels is not None or com_priority_projection is not None:
                    # [archived branch] raise ValueError('Weighted transition requires a single unprojected priority stage')
                # [archived branch] secondary_phase = min(1., (elapsed_motion_time-args.weighted_tasks_start)
                                      # [archived branch] / args.weighted_tasks_ramp)
                # [archived branch] secondary_task_blend = secondary_phase**3*(10-15*secondary_phase+6*secondary_phase**2)
                # [archived branch] blend_secondary_task_metrics(fabric.dynamics_tasks.last_requests,
                                             # [archived branch] priority_tasks, priority_rows, secondary_task_blend)
                # [archived branch] priority_tasks, priority_rows = (), {}
            current_effort_limits=qp_effort_limits
            if elapsed_motion_time>=args.model_effort_start:
                effort_phase=min(1.,max(0.,(elapsed_motion_time-args.model_effort_start)/args.model_effort_ramp))
                effort_blend=effort_phase**3*(10-15*effort_phase+6*effort_phase**2)
                current_effort_limits=.8*(legacy_effort_limits+effort_blend*(effort_limits-legacy_effort_limits))
            # [archived branch] if elapsed_motion_time>=args.synchronous_effort_budget_start:
                # [archived branch] from synchronous_effort_budget import effort_fraction
                # [archived branch] current_effort_limits=effort_limits*effort_fraction(elapsed_motion_time,args.synchronous_effort_budget_start,
                    # [archived branch] args.synchronous_effort_budget_fraction,FABRIC_TIMESTEP,PD_TIMESTEP,
                    # [archived branch] direct_wbc=args.direct_wbc,tracking_servo=args.tracking_servo)
            previous_validated_torque = qp_output.joint_torque.detach().clone()
            torque_continuity_weight = 0.
            # [archived branch] if torque_continuity_solver is not None:
                # [archived branch] torque_phase = min(1., max(0., (elapsed_motion_time-args.torque_continuity_start)
                                           # [archived branch] / args.torque_continuity_ramp))
                # [archived branch] torque_continuity_weight = args.torque_continuity_weight*torque_phase**3*(10-15*torque_phase+6*torque_phase**2)
                # [archived branch] torque_continuity_solver.update(previous_validated_torque, torque_continuity_weight)
            qp_wall_start = time.perf_counter()
            qp_solve=acceleration_qp.solve
            acceleration_qp.tube_recovery_used=False
            acceleration_qp.orientation_tube_recovery_bandwidth=float('nan')
            if args.relax_joint_tube and (elapsed_motion_time>=args.tracking_tube_start and elapsed_motion_time<args.tracking_tube_end):
                from functools import partial
                from tube_recovery import solve_with_joint_tube_recovery
                qp_solve=partial(solve_with_joint_tube_recovery,acceleration_qp,tube_C=tube_C,tube_d=tube_d,recovery_metric=args.tube_recovery_metric)
                # [archived branch] if args.recover_orientation_tube:
                from tube_recovery import solve_with_orientation_tube_recovery
                qp_solve=partial(solve_with_orientation_tube_recovery,acceleration_qp,tube_C=tube_C,tube_d=tube_d,
                    tube_diagnostics=tube_diagnostics,orientation_bandwidth=(orientation_tube_rate or tube_rate),
                    recovery_metric=args.tube_recovery_metric)
            # [archived branch] if elapsed_motion_time>=args.full_body_preview_start:
                # [archived branch] if not args.direct_wbc or not args.tracking_servo or str(args.device)!='cpu' or args.contact_pause_end is not None or abs(FABRIC_TIMESTEP-.002)>1e-12:
                    # [archived branch] raise ValueError('Full-body preview requires synchronous 500 Hz direct CPU tracking without a pause clock')
                # [archived branch] if 'full_body_preview' not in locals():
                    # [archived branch] from preview_process import ProcessPreview
                    # [archived branch] from online_full_body_preview import OnlineFullBodyPreview
                    # [archived branch] from fabrics_sim.reference.manager import contact_sequence_modes
                    # [archived branch] preview_modes=contact_sequence_modes(_to_numpy(motion_contact_sequence),len(motion_trajectory))
                    # [archived branch] preview_geometry=[tuple(_to_numpy(x) for x in row) for row in foot_landing_geometry]
                    # [archived branch] if (args.full_body_preview_state_feedback or args.full_body_preview_release_points) and not args.full_body_preview_fine_control:
                        # [archived branch] raise ValueError('State feedback and contact release require fine preview control')
                    # [archived branch] if args.full_body_preview_fine_control and not args.full_body_preview_collision_barrier:
                        # [archived branch] raise ValueError('Fine preview control requires future collision barriers')
                    # [archived branch] if args.full_body_preview_exact_rotation and not (args.full_body_preview_fine_control and args.full_body_preview_release_points):
                        # [archived branch] raise ValueError('Exact preview rotation requires fine control and contact release')
                    # [archived branch] if args.full_body_preview_recover_joint_reserve and not args.full_body_preview_exact_rotation:
                        # [archived branch] raise ValueError('Joint reserve recovery requires exact preview rotation')
                    # [archived branch] if args.full_body_preview_damp_first_control and not args.full_body_preview_recover_joint_reserve:
                        # [archived branch] raise ValueError('First-control damping requires the joint-reserve recovery predictor')
                    # [archived branch] if args.full_body_preview_surface_kinematics and not args.full_body_preview_separating_points:
                        # [archived branch] raise ValueError('Surface kinematics requires separation-aware fine preview')
                    # [archived branch] if args.full_body_preview_separating_points and not args.full_body_preview_damp_first_control:
                        # [archived branch] raise ValueError('Separating-point prediction requires first-control damping')
                    # [archived branch] preview_configuration=dict(urdf=fabric.urdf_path,names=G1_JOINT_NAMES,
                        # [archived branch] reference_path=str(args.output_path.with_suffix('.control_reference.npz')),
                        # [archived branch] modes=preview_modes,geometry=preview_geometry,
                        # [archived branch] limits=_to_numpy(joint_position_limits)[0],effort=_to_numpy(current_effort_limits).ravel(),
                        # [archived branch] velocity=_to_numpy(joint_velocity_limits).ravel(),horizon=args.full_body_preview_horizon,
                        # [archived branch] guarded=args.full_body_preview_guarded,warm_start=args.full_body_preview_warm_start,fine_control=args.full_body_preview_fine_control,release_points=args.full_body_preview_release_points,
                        # [archived branch] native_contact_timing=args.full_body_preview_native_contact_timing,
                        # [archived branch] exact_rotation=args.full_body_preview_exact_rotation,
                        # [archived branch] recover_joint_reserve=args.full_body_preview_recover_joint_reserve,
                        # [archived branch] damp_first_control=args.full_body_preview_damp_first_control,
                        # [archived branch] separating_points=args.full_body_preview_separating_points,
                        # [archived branch] surface_kinematics=args.full_body_preview_surface_kinematics,
                        # [archived branch] contact_timing_start_frame=int(math.ceil(args.full_body_preview_start*motion_fps)),
                        # [archived branch] contact_schedule_path=str(args.output_path.with_suffix('.preview_contact_schedule.npz')))
                    # [archived branch] if args.full_body_preview_collision_barrier:
                        # [archived branch] tasks=fabric.dynamics_tasks
                        # [archived branch] if not tasks.wrist_hip_barrier or tasks.convex_wrist_geometry is not None:
                            # [archived branch] raise ValueError('Preview collision barrier requires the enabled runtime sphere covers')
                        # [archived branch] sphere_names=tasks.sphere_names
                        # [archived branch] centers=_to_numpy(tasks.sphere_centers);radii=_to_numpy(tasks.sphere_radii)
                        # [archived branch] spheres={name:dict(link=tasks.model.kinematics.link_names[tasks.sphere_indices[i]],center=centers[i].tolist(),radius=float(radii[i])) for i,name in enumerate(sphere_names)}
                        # [archived branch] pairs=[[sphere_names[int(a)],sphere_names[int(b)]] for a,b in _to_numpy(tasks.pairs)[tasks.original_collision_pairs:tasks.barrier_pair_count]]
                        # [archived branch] preview_configuration['collision_geometry']=dict(spheres=spheres,pairs=pairs,safe_distance=tasks.config['collision']['safe_distance'])
                    # [archived branch] preview_planner=ProcessPreview(preview_configuration,args.output_path.with_suffix('.planner_worker.log'))
                    # [archived branch] full_body_preview=OnlineFullBodyPreview(preview_planner,args.output_path.with_suffix('.full_body_preview.json'),period=args.full_body_preview_period,track_wrench=args.full_body_preview_track_wrench,reuse_verified=args.full_body_preview_reuse_verified,follow_plan=args.full_body_preview_fine_control,allow_projection_correction=args.full_body_preview_state_feedback)
                # [archived branch] from functools import partial
                # [archived branch] from observed_preview_patch import observed_native_patch_indices
                # [archived branch] try:
                    # [archived branch] initial_preview_patch=observed_native_patch_indices(_to_numpy(live_feet),preview_geometry,tuple(map(_to_numpy,contact_data)),int(output.contact_mode.item()))
                # [archived branch] except ValueError as error:
                    # [archived branch] full_body_preview.observation_failure(control_time,frame_position,error)
                    # [archived branch] mismatch_path=args.output_path.with_suffix('.initial_patch_mismatch.npz')
                    # [archived branch] if not mismatch_path.exists():
                        # [archived branch] np.savez_compressed(mismatch_path,time=control_time,frame=frame_position,q=_to_numpy(q),v=_to_numpy(qd),feet=_to_numpy(live_feet),mode=int(output.contact_mode.item()),
                            # [archived branch] **{key:_to_numpy(value) for key,value in zip(('forces','points','normals','distances','counts','starts'),contact_data)})
                # [archived branch] else:
                    # [archived branch] qp_solve=partial(full_body_preview.solve,qp_solve,acceleration_qp,q,qd,control_time,frame_position,frame_rate,initial_patch_indices=initial_preview_patch)
            # [archived branch] if shoulder_guard is not None:
                # [archived branch] from functools import partial
                # [archived branch] qp_solve = partial(shoulder_guard.solve, qp_solve, acceleration_qp,
                    # [archived branch] joint_names=G1_JOINT_NAMES, multiplier=shoulder_weight, side=args.shoulder_roll_weight_side)
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
                known_joint_torque=None,
                servo_bias=None,
                servo_acc_gain=None,
                command_acc_correction=qp_reference - qdd_dc,
                contact_wrench_reference=wrench_reference,
                contact_wrench_weights=wrench_weights,
                track_predicted_base=args.qp_track_predicted_base,
                friction_G=friction_G,
                friction_b=friction_b,
                # task_requests=([composed_policy] if args.g1_md else ...),
                task_requests=fabric.dynamics_tasks.last_requests,
                task_tracks_predicted=not args.task_track_nominal,
                priority_tasks=priority_tasks,
                priority_requests=None,
                priority_levels=priority_levels,
                priority_rows=priority_rows,
                priority_projections={"center_of_mass": com_priority_projection}
                    if com_priority_projection is not None else None,
                acceleration_constraints=task_acceleration_constraints,
                contact_row_mask=contact_row_mask,
                # Physical braking bounds use the current measured joint state.
                predicted_acc_limit=None,
                predicted_acc_lower=physical_lower if physical_braking_active else None,
                predicted_acc_upper=physical_upper if physical_braking_active else None,
            )
            for snapshot_time in args.qp_snapshot_times:
                if abs(control_time-snapshot_time)<FABRIC_TIMESTEP/4:
                    snapshot_arrays={'q':_to_numpy(q),'v':_to_numpy(qd),'time':control_time,
                        'reference_q':_to_numpy(reference_q),'reference_v':_to_numpy(reference_v),
                        'predicted_acc':_to_numpy(qp_output.predicted_acc),'feet':_to_numpy(live_feet),
                        'contact_mode':_to_numpy(output.contact_mode),'forces':_to_numpy(foot_normal_forces)}
                    for key,value in vars(acceleration_qp.last_priority_problems[0]).items():
                        if isinstance(value,torch.Tensor):snapshot_arrays['problem_'+key]=_to_numpy(value)
                    for request in fabric.dynamics_tasks.last_requests:
                        for key in ('A','bias','desired','metric','active'):
                            snapshot_arrays['task_'+request.name+'_'+key]=_to_numpy(getattr(request,key))
                    np.savez_compressed(args.output_path.with_suffix(f'.snapshot_{snapshot_time:.3f}.npz'),**snapshot_arrays)
            if acceleration_qp.tube_recovery_used:
                saved_C,saved_d=task_acceleration_constraints
                task_acceleration_constraints=(torch.cat((saved_C[:,:-3],saved_C[:,-2:]),1),torch.cat((saved_d[:,:-3],saved_d[:,-2:]),1))
                print(f'Joint tracking tube softened at {control_time:.3f}s; physical QP validated={bool(qp_output.validated.all())}',flush=True)
            # [archived branch] if args.g1_md and fabric.dynamics_tasks is not None:
                # Diagnostics remain task-space quantities even though the QP
                # now tracks one metric-preserving root policy, not leaf tasks.
                # [archived branch] qp_output.task_residuals = {
                    # [archived branch] request.name: request.residual(qp_output.predicted_acc)
                    # [archived branch] for request in fabric.dynamics_tasks.last_requests
                # [archived branch] }
            # Reading validation synchronizes GPU work before measuring solve
            # latency. A 2 ms simulation period does not certify 2 ms wall time.
            qp_validated = bool(qp_output.validated.all())
            com_diagnostic = {"qp_wall_seconds": time.perf_counter()-qp_wall_start,
                              "contact_sample_dt": contact_sample_dt}
            # [archived branch] if args.shoulder_roll_weight_scale > 1:
                # [archived branch] com_diagnostic['shoulder_roll_weight_multiplier'] = (
                    # [archived branch] shoulder_weight if shoulder_guard is None else shoulder_guard.diagnostic['shoulder_guard_multiplier'])
            # [archived branch] if shoulder_guard is not None:
                # [archived branch] com_diagnostic.update(shoulder_guard.diagnostic)
            # [archived branch] if math.isfinite(args.weighted_tasks_start):
                # [archived branch] com_diagnostic['secondary_task_blend'] = secondary_task_blend
            # [archived branch] if math.isfinite(args.support_flatten_start):
                # [archived branch] com_diagnostic['support_flatten_weight'] = _to_numpy(support_flatten_weights).copy()
            # [archived branch] if math.isfinite(args.root_position_task_start):
                # [archived branch] com_diagnostic['root_position_task_weight'] = root_task_weight
            # [archived branch] if math.isfinite(args.stance_posture_metric_start):
                # [archived branch] com_diagnostic['stance_posture_metric_scale'] = stance_posture_scale
            # [archived branch] if math.isfinite(args.rolling_contact_kinematics_start):
                # [archived branch] com_diagnostic.update(rolling_bias_correction=_to_numpy(rolling_bias_correction).copy(),
                                      # [archived branch] rolling_bias_matched=_to_numpy(rolling_bias_matched).copy())
            com_diagnostic.update(torque_continuity_weight=torque_continuity_weight,
                torque_step_change=_to_numpy(qp_output.joint_torque-previous_validated_torque).copy())
            if mocap_transition is not None:
                com_diagnostic.update(mocap_transition_phase=mocap_transition.phase,
                    mocap_transition_unload_weight=mocap_transition.unload_weight(control_time))
            predicted_joint_velocity = joint_velocity + FABRIC_TIMESTEP*qp_output.predicted_acc[:,6:]
            velocity_violation = (predicted_joint_velocity.abs()-joint_velocity_limits).clamp_min(0.)
            # [archived branch] if args.joint_velocity_limits:
            qp_validated = qp_validated and bool((velocity_violation <= 1e-4).all())
            com_diagnostic['predicted_joint_velocity'] = _to_numpy(predicted_joint_velocity).copy()
            com_diagnostic['joint_velocity_limits'] = _to_numpy(joint_velocity_limits).copy()
            com_diagnostic['joint_velocity_violation'] = _to_numpy(velocity_violation).copy()
            com_diagnostic['measured_joint_velocity_violation'] = _to_numpy((joint_velocity.abs()-joint_velocity_limits).clamp_min(0.)).copy()
            com_diagnostic['joint_velocity_limits_enabled'] = args.joint_velocity_limits
            com_diagnostic.update(sole_barrier_diagnostic)
            com_diagnostic.update(knee_guard_diagnostic)
            com_diagnostic.update(landing_diagnostic)
            com_diagnostic.update(momentum_bound_diagnostic)
            com_diagnostic.update(com_replan_diagnostic)
            # [archived branch] if reference_manager is not None:
                # [archived branch] com_diagnostic.update(reference_settle_time=reference_manager.settle_time,
                    # [archived branch] reference_load_confirmation=reference_manager.confirmed*FABRIC_TIMESTEP,
                    # [archived branch] sole_min_height=_to_numpy(observed_corners[...,2].amin(-1)),
                    # [archived branch] sole_max_height=_to_numpy(observed_corners[...,2].amax(-1)))
            # [archived branch] if args.observed_contact_patches:
            patch_desired=np.full((2,3),np.nan);patch_residual=np.full((2,3),np.nan)
            patch_flatten=np.zeros(2,dtype=bool)
            for foot,task in contact_patch_tasks.items():
                width=task.desired.shape[-1]
                patch_desired[foot,:width]=_to_numpy(task.desired[0])
                patch_residual[foot,:width]=_to_numpy(qp_output.task_residuals[task.name][0])
                patch_flatten[foot]=contact_patches[foot].flatten
            com_diagnostic.update(contact_orientation_desired=patch_desired,
                                  contact_orientation_residual=patch_residual,
                                  contact_flatten=patch_flatten)
            # [archived branch] if args.wrist_hip_barrier:
            com_diagnostic['collision_acceleration_constraint_error'] = _to_numpy(
                acceleration_qp.last_acceleration_constraint_error).copy()
            if args.qp_solver in ('osqp','daqp'):
                com_diagnostic["qp_solver_status"] = np.asarray(acceleration_qp.solver.last_status)
                com_diagnostic["qp_workspace_reused"] = np.asarray(acceleration_qp.solver.last_workspace_reused)
                com_diagnostic["qp_priority_solver_status"] = np.asarray(
                    acceleration_qp.last_priority_solver_status or ['inactive'])
            com_diagnostic['source_swing_position_priority_active'] = np.asarray(source_priority is not None and args.source_swing_position_priority)
            com_diagnostic['source_swing_pose_priority_active'] = np.asarray(source_priority is not None and args.source_swing_pose_priority)
            # [archived branch] if args.transfer_balance_priority:
                # Keep the history schema constant before and after the first
                # transfer, so delayed activation does not drop these fields.
                # [archived branch] com_diagnostic.update(
                    # [archived branch] balance_priority_active=np.asarray(acceleration_qp.last_priority_errors is not None),
                    # [archived branch] swing_height_priority_active=np.asarray(height_priority),
                    # [archived branch] swing_com_priority_active=np.asarray(swing_com_priority),
                    # [archived branch] priority_stage_constraint_error=np.zeros(1, dtype=np.float32),
                    # [archived branch] priority_lock_error=np.zeros(1, dtype=np.float32))
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
                    com_diagnostic['task_active_' + name] = name in qp_output.task_residuals
                com_diagnostic.update({"task_residual_" + name: residual.detach().cpu().numpy().copy()
                                       for name, residual in qp_output.task_residuals.items()})
                # [archived branch] if args.convex_wrist_collision:
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
                # [archived branch] if "collision_distance" in snapshot and not args.convex_wrist_collision:
                    # [archived branch] com_diagnostic["collision_distance"] = snapshot["collision_distance"].cpu().numpy().copy()
            if any(control_updates == round(t / FABRIC_TIMESTEP) for t in args.diagnose_qp_time):
                # Only serialize: no solver/controller input is changed.
                _, diagnostic_com_jac = fabric.center_of_mass_taskmap._position_and_jacobian(q)
                snapshot_path = args.output_path.with_suffix(f".qp_{control_updates:04d}.npz")
                snapshot_path.parent.mkdir(parents=True, exist_ok=True)
                tensors = dict(mass=mass, contact_jacobian=contact_jacobian,
                               contact_mode=output.contact_mode, bias_force=bias_force,
                               com_jacobian=diagnostic_com_jac, servo_bias=servo_bias,
                               servo_acc_gain=servo_acc_gain, correction=qp_reference-qdd_dc,
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
                                       if fabric.dynamics_tasks is not None else ())
                       for field in ("A", "bias", "desired", "metric", "active")},
                    problem_solution=torch.cat((qp_output.acc, qp_output.nominal_contact_wrench,
                                                qp_output.joint_torque), dim=1).detach().cpu().numpy())
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
                    problem_solution=torch.cat((qp_output.acc, qp_output.nominal_contact_wrench,
                                                qp_output.joint_torque), dim=1).cpu().numpy(),
                    self_contact_events=np.asarray(self_contact_events, dtype=str).reshape(-1, 4),
                    require_support_load=np.asarray(args.require_support_load),
                    phase_hold_duration=np.asarray(phase_hold_updates * FABRIC_TIMESTEP),
                    motion_frame_position=np.asarray(frame_position if motion_trajectory is not None else np.nan),
                    command_acc_correction=(qp_reference - qdd_dc).detach().cpu().numpy(),
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

            # One-step prediction ONLY for history and existing validation.
            # Seeds were reset to measured q/v above; these are not motor targets.
            joint_velocity_prediction = (
                joint_velocity_prediction
                + FABRIC_TIMESTEP * qdd_dynamics_feasible[:, 6:35]
            )
            joint_position_prediction = (
                joint_position_prediction
                + FABRIC_TIMESTEP * joint_velocity_prediction
            )
            # [archived branch] if args.fixed_joint_test:
                # [archived branch] joint_position_desired = integration_initial_position.clone()
                # [archived branch] joint_velocity_desired = torch.zeros_like(integration_initial_velocity)
                # [archived branch] joint_torque_feedforward = initial_joint_feedforward.clone()
            # position_desired_isaac = joint_position_desired[
                # :, fabric_to_isaac_joint
            # ]
            # velocity_desired_isaac = joint_velocity_desired[
                # :, fabric_to_isaac_joint
            # ]
            torque_feedforward_isaac = joint_torque_feedforward[
                :, fabric_to_isaac_joint
            ]
            # [archived branch] joint_torque_pd = (
                # [archived branch] torch.zeros_like(joint_velocity) if args.tracking_servo else -kd*joint_velocity if args.direct_wbc else
                # [archived branch] kp * (joint_position_desired - joint_position)
                # [archived branch] + kd * (joint_velocity_desired - joint_velocity)
            # [archived branch] )
            joint_torque_pd = torch.zeros_like(joint_velocity)
            joint_torque_unclipped = (
                joint_torque_feedforward + joint_torque_pd
            )
            joint_torque = torch.clamp(
                joint_torque_unclipped,
                min=-effort_limits,
                max=effort_limits,
            )
            # [archived branch] if args.mocap_unload_test:
                # [archived branch] world_wrench = qp_output.contact_wrench.reshape(1, 2, 6).clone()
                # [archived branch] for foot, patch in contact_patches.items():
                    # [archived branch] world_wrench[:, foot] = patch.wrench_to_world_ankle(
                        # [archived branch] world_wrench[:, foot], _quat_xyzw_to_matrix(live_feet[:, foot, 3:]))
                # [archived branch] com_diagnostic.update(
                    # [archived branch] unload_support=-1 if transfer_reference is None else transfer_reference.support,
                    # [archived branch] unload_elapsed=-1. if transfer_reference is None else control_time-transfer_reference.start_time,
                    # [archived branch] contact_wrench_world_ankle=_to_numpy(world_wrench).copy(),
                    # [archived branch] unload_com_velocity=_to_numpy(actual_com_velocity).copy(),
                    # [archived branch] unload_effort_ratio=_to_numpy((qp_output.total_joint_torque.abs()/effort_limits).amax(-1)).copy(),
                    # [archived branch] unload_qp_validated=_to_numpy(qp_output.validated).copy(),
                    # [archived branch] unload_base_error=_to_numpy(qp_output.base_dynamics_error).copy(),
                    # [archived branch] unload_contact_error=_to_numpy(qp_output.contact_acceleration_error).copy(),
                    # [archived branch] unload_friction_error=_to_numpy(qp_output.friction_violation).copy(),
                    # [archived branch] unload_bound_error=_to_numpy(qp_output.bound_violation).copy())
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
                    "qdd_qp_reference": qp_reference.detach().cpu().numpy().copy(),
                    "qdd_predicted": qp_output.predicted_acc.detach().cpu().numpy().copy(),
                    "contact_wrench": qp_output.contact_wrench.detach().cpu().numpy().copy(),
                    "wrench_reference": (torch.zeros_like(qp_output.contact_wrench) if wrench_reference is None else wrench_reference).detach().cpu().numpy().copy(),
                    "wrench_objective_weights": (torch.full_like(qp_output.contact_wrench, acceleration_qp.wrench_weight)
                        if wrench_weights is None else acceleration_qp.wrench_weight*wrench_weights).detach().cpu().numpy().copy(),
                    "wrench_target_active": wrench_reference is not None,
                    "pelvis_target": fabric.pelvis_pose_taskmap.position_target.detach().cpu().numpy().copy(),
                    "qdd_nominal": qp_output.nominal_acc.detach().cpu().numpy().copy(),
                    "command_acc_correction": (qp_reference - qdd_dc).detach().cpu().numpy().copy(),
                    "joint_position_desired": joint_position_prediction.detach().cpu().numpy().copy(),
                    "joint_velocity_desired": joint_velocity_prediction.detach().cpu().numpy().copy(),
                    "integration_seed_position": prediction_seed_position.detach().cpu().numpy().copy(),
                    "integration_seed_velocity": prediction_seed_velocity.detach().cpu().numpy().copy(),
                    "torque_feedforward": joint_torque_feedforward.detach().cpu().numpy().copy(),
                    "torque_qp_total": qp_output.total_joint_torque.detach().cpu().numpy().copy(),
                    "torque_pd_estimate": joint_torque_pd.detach().cpu().numpy().copy(),
                    "unload_intended_mode": unload_intended_mode,
                    "unload_intended_reason": unload_intended_reason,
                    "effective_qp_effort_limits": _to_numpy(current_effort_limits).copy(),
                    "joint_tube_recovery_used": acceleration_qp.tube_recovery_used,
                    "orientation_tube_recovery_bandwidth": acceleration_qp.orientation_tube_recovery_bandwidth,
                    "tracking_tube_diagnostics": (_to_numpy(tube_diagnostics).copy() if (elapsed_motion_time>=args.tracking_tube_start and elapsed_motion_time<args.tracking_tube_end) else np.full((1,3,3),np.nan)),
                    "tracking_tube_bound": (_to_numpy(tube_d).copy() if (elapsed_motion_time>=args.tracking_tube_start and elapsed_motion_time<args.tracking_tube_end) else np.full((1,3),np.nan)),
                    "tracking_tube_residual": (_to_numpy((tube_C@qp_output.predicted_acc[...,None]).squeeze(-1)-tube_d).copy() if (elapsed_motion_time>=args.tracking_tube_start and elapsed_motion_time<args.tracking_tube_end) else np.full((1,3),np.nan)),
                })
            if fidelity_stop is None and motion_trajectory is not None:
                from analyze_motion_tracking import summarize_tracking
                sample_audit=summarize_tracking(dict(history_frame=np.asarray([frame_position]),history_q=_to_numpy(q),history_reference_q=_to_numpy(audit_reference_q),source_frame_count=len(motion_trajectory),source_start_frame=0))
                if not sample_audit['pkl_fidelity_passed']:
                    fidelity_stop = dict(
                        time_s=float(control_time),
                        source_frame=float(frame_position),
                        joint_rmse_rad=sample_audit['peak_joint_rmse'],
                        root_position_m=sample_audit['peak_root_position_error'],
                        root_orientation_rad=sample_audit['peak_root_orientation_error'],
                    )
                    print('Original-source fidelity limit exceeded: '+json.dumps(fidelity_stop),flush=True)
                    # [archived branch] if args.stop_at_fidelity_loss:
                    break
            # [archived branch] if (args.mocap_first_step_report is not None and transfer_reference is not None
                    # [archived branch] and source_step_rollout is None and control_time >= transfer_reference.start_time+3.5-FABRIC_TIMESTEP):
                # [archived branch] try:
                    # [archived branch] from analyze_contact_transfer import summarize_unloading
                    # [archived branch] prefix = {'history_'+key: np.asarray([row[key] for row in tracking_history])
                              # [archived branch] for key in tracking_history[0]}
                    # Pre-release controller gate uses recorded physical QP and speed checks;
                    # final full-controller and landing validation still runs at shutdown.
                    # [archived branch] prefix['motion_tracking_valid'] = bool(
                        # [archived branch] np.all(prefix['history_unload_qp_validated'])
                        # [archived branch] and np.max(prefix['history_joint_velocity_violation'])<=1e-4
                        # [archived branch] and np.isfinite(prefix['history_q']).all())
                    # [archived branch] probe_unload_report = summarize_unloading(prefix)
                    # [archived branch] probe_unload_report['controller_gate_scope'] = 'pre-release recorded QP, speed and finite-state checks'
                    # [archived branch] args.output_path.with_suffix('.probe_unloading.json').write_text(json.dumps(probe_unload_report,indent=2)+"\n")
                    # [archived branch] if not probe_unload_report['unloading_passed']:
                        # [archived branch] raise RuntimeError('First-step probe requires verified measured unloading')
                    # [archived branch] from fabrics_sim.reference.optimized_source_rollout import OptimizedSourceRollout
                    # [archived branch] from g1_armature import unitree_armatures
                    # [archived branch] source_step_rollout = OptimizedSourceRollout(fabric,
                        # [archived branch] WholeBodyReference(q.clone(),qd.clone(),qp_output.predicted_acc.clone()),
                        # [archived branch] G1_JOINT_NAMES,control_motion_trajectory,FABRIC_TIMESTEP,
                        # [archived branch] q.new_tensor(unitree_armatures(G1_JOINT_NAMES)),current_effort_limits,
                        # [archived branch] args.mocap_first_step_report,args.output_path.with_suffix('.probe_plan.json'),
                        # [archived branch] project_nominal_torque=args.mocap_first_step_project_torque)
                    # [archived branch] source_step_start = control_time
                    # [archived branch] source_step_number = 1
                    # [archived branch] transfer_reference.release_time = control_time
                    # [archived branch] reference_manager.phase = 'SWING'
                    # [archived branch] reference_manager.pending_takeoff = None
                    # [archived branch] fabric.dynamics_tasks.preserve_reference_acceleration = True
                    # [archived branch] print(f'Live-boundary first-step probe started: time={control_time:.3f}s, target_frame=129',flush=True)
                # [archived branch] except (ValueError,RuntimeError) as exc:
                    # [archived branch] save_reference_failure(args,exc,control_time,q,qd,foot_normal_forces,tracking_history,servo_history)
                    # [archived branch] raise
            if args.motion_start_in_motion and measured_steps is not None:
                from analyze_contact_transfer import sole_clearances
                feet_observed = _to_numpy(live_feet[0])
                measured_steps.update(frame_position, control_time, int(measured_mode.item()),
                    _to_numpy(foot_normal_forces[0]), sole_clearances(feet_observed,
                        np.stack([_to_numpy(g[0]) for g in foot_landing_geometry]),
                        np.stack([_to_numpy(g[1]) for g in foot_landing_geometry])),
                    feet_observed[:,:3], FABRIC_TIMESTEP, reference_feet=source_feet_reference)
                # [archived branch] if measured_steps.complete and not args.track_full_motion:
                    # [archived branch] stopped_at_tracking_steps = True
                    # [archived branch] print(f'Measured {args.tracking_steps} source landings at {control_time:.3f}s; '
                          # [archived branch] 'continuous-walking endpoint, not a stationary settle', flush=True)
                    # [archived branch] break
            if args.track_full_motion and frame_position >= len(motion_trajectory)-1:
                stopped_at_motion_end = True
                print(f'Full source endpoint reached: frame={frame_position:g}, time={control_time:.3f}s; '
                      'continuous endpoint, no stationary settling claim', flush=True)
                break
            control_updates += 1
            # [archived branch] if args.contact_aware_servo:
            # [archived branch] if args.nominal_implicit_servo:
            # servo_kp_bank = nominal_kp[None].expand(4,-1)
            # servo_kd_bank = nominal_kd[None].expand(4,-1)
            # [archived branch] else:
                # [archived branch] servo_kp_bank, servo_kd_bank = contact_servo_gain_bank(
                    # [archived branch] mass, contact_jacobian, contact_row_mask, nominal_kp, nominal_kd,
                    # [archived branch] PD_TIMESTEP, args.servo_bandwidth)
            # servo_response_bank = implicit_servo_response(mass,servo_kp_bank,servo_kd_bank,PD_TIMESTEP)
            # servo_response_bank = servo_response_bank[:,fabric_to_isaac_joint][:,:,fabric_to_isaac_joint]
            # servo_kp_bank = servo_kp_bank[:,fabric_to_isaac_joint]
            # servo_kd_bank = servo_kd_bank[:,fabric_to_isaac_joint]

        # actual_position_isaac = torch.as_tensor(
            # _to_numpy(robot.get_joint_positions()),
            # device=device,
            # dtype=q.dtype,
        # )
        actual_velocity_isaac = torch.as_tensor(
            _to_numpy(robot.get_joint_velocities()),
            device=device,
            dtype=q.dtype,
        )
        # [archived branch] if args.tracking_servo:
        # [archived branch] if args.contact_aware_servo:
        # fast_forces = ground_support_forces(_to_numpy(
            # foot_contact_view.get_contact_force_matrix(dt=contact_sample_dt)))[:,2]
        # loaded_mode = int(fast_forces[0] >= 20.) + 2*int(fast_forces[1] >= 20.)
        # servo_contact_mode = loaded_mode & int(output.contact_mode.item())
        # kp_isaac, kd_isaac = servo_kp_bank[servo_contact_mode], servo_kd_bank[servo_contact_mode]
        # elapsed=(step % fabric_steps_per_update)*PD_TIMESTEP
        # seed_position=integration_seed_position[:,fabric_to_isaac_joint]
        # seed_velocity=integration_seed_velocity[:,fabric_to_isaac_joint]
        # seed_acceleration=qp_output.predicted_acc[:,6:][:,fabric_to_isaac_joint]
        # servo_position=seed_position+seed_velocity*elapsed+.5*seed_acceleration*elapsed**2
        # servo_velocity=seed_velocity+seed_acceleration*elapsed
        # joint_torque_pd_isaac=(kp_isaac*(servo_position-actual_position_isaac)
                               # +kd_isaac*(servo_velocity-actual_velocity_isaac))
        # [archived branch] if args.contact_aware_servo:
        # rhs = joint_torque_pd_isaac + PD_TIMESTEP*kp_isaac*(servo_velocity-actual_velocity_isaac)
        # joint_torque_pd_isaac = (servo_response_bank[servo_contact_mode] @ rhs[...,None]).squeeze(-1)
        # [archived branch] else:
            # [archived branch] joint_torque_pd_isaac = (
                # [archived branch] -kd_isaac*actual_velocity_isaac if args.direct_wbc else
                # [archived branch] kp_isaac * (position_desired_isaac - actual_position_isaac)
                # [archived branch] + kd_isaac * (velocity_desired_isaac - actual_velocity_isaac)
            # [archived branch] )
        # joint_torque_isaac_tensor = (
            # torque_feedforward_isaac + joint_torque_pd_isaac
        # )
        # Synchronous 500 Hz QP/effort loop: no inter-tick servo interval.
        # Task-space feedback is already in the QP acceleration objectives.
        joint_torque_isaac_tensor = torque_feedforward_isaac
        joint_torque_isaac_tensor = torch.clamp(
            joint_torque_isaac_tensor,
            min=-effort_limits_isaac,
            max=effort_limits_isaac,
        )
        # Send the QP effort in Isaac joint order, after the final effort limit.
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
            "unclipped": torque_feedforward_isaac.detach().cpu().numpy().copy(),
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
            (joint_position_after_control - joint_position_prediction).square(),
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
    # [archived branch] if managed_clock is not None:
        # [archived branch] reference_status = (
            # [archived branch] "source_finished" if managed_clock.frame >= len(motion_trajectory)-1 else
            # [archived branch] "waiting_for_contact" if managed_clock.phase in ("PREPARE_STEP", "LANDING", "SETTLE") and managed_clock.waiting_seconds > 0 else
            # [archived branch] "initializing" if managed_clock.phase == "INITIALIZE" else "advancing"
        # [archived branch] )
        # [archived branch] if source_step_rollout is not None:
            # [archived branch] reference_status = "diagnostic_" + managed_clock.phase.lower()
        # [archived branch] elif args.unified_transfer_test and transfer_reference is not None:
            # [archived branch] reference_status = "diagnostic_" + transfer_reference.phase.lower()
        # [archived branch] print(f"ReferenceManager: status={reference_status}, phase={managed_clock.phase}, "
              # [archived branch] f"source frame={managed_clock.frame:.2f}, rate={managed_clock.rate:.3f}, "
              # [archived branch] f"waiting={managed_clock.waiting_seconds:.2f}s; controller validation alone does not certify gait success")
    # [archived branch] if args.unified_transfer_test and not args.unified_source_step_test:
        # [archived branch] recent = tracking_history[-int(round(1./FABRIC_TIMESTEP)):]
        # [archived branch] transfer_valid = (transfer_reference is not None and transfer_reference.release_time is not None
                          # [archived branch] and (executed_steps-1)*PD_TIMESTEP >= transfer_reference.release_time+3.
                          # [archived branch] and len(recent) == int(round(1./FABRIC_TIMESTEP)))
        # [archived branch] if transfer_valid:
            # [archived branch] from analyze_contact_transfer import sole_clearances
            # [archived branch] support = transfer_reference.support
            # [archived branch] for row in recent:
                # [archived branch] feet = row["feet"].reshape(2, 7)
                # [archived branch] transfer_valid &= (np.all(row["commanded_mode"] == support+1)
                    # [archived branch] and np.linalg.norm(row["com"]-row["com_target"]) <= .03
                    # [archived branch] and sole_clearances(feet)[1-support] >= .02
                    # [archived branch] and row["foot_forces"][1-support, 2] < 10.
                    # [archived branch] and row["foot_forces"][support, 2] >= 100.)
        # [archived branch] motion_tracking_valid &= bool(transfer_valid)
        # [archived branch] print(f"Whole-body transfer validation: valid={bool(transfer_valid)}, release_time={None if transfer_reference is None else transfer_reference.release_time}; requires 3s after release and 1s stable single support")
    # [archived branch] if args.unified_source_step_test or args.mocap_first_step_report is not None:
        # [archived branch] last = tracking_history[-1]
        # [archived branch] swing = 1-transfer_reference.support if transfer_reference is not None else 1
        # [archived branch] landing_error = (float(np.linalg.norm(last["feet"].reshape(2,7)[swing,:3]-source_step_rollout.goal[0,:3].cpu().numpy()))
                         # [archived branch] if source_step_rollout is not None else float('inf'))
        # [archived branch] step_valid = bool(stopped_at_source_step_end and source_step_rollout is not None and source_step_rollout.touchdown and landing_error <= .04
                          # [archived branch] and np.linalg.norm(last["com"]-last["com_target"]) <= .05)
        # [archived branch] motion_tracking_valid &= step_valid
        # [archived branch] print(f"First source-step validation: valid={step_valid}, landing_error={landing_error:.6f} m; no continuous-walking certification")
        # [archived branch] if args.mocap_first_step_report is not None:
            # [archived branch] args.output_path.with_suffix('.probe_result.json').write_text(json.dumps(dict(
                # [archived branch] scope='bounded_first_source_step_probe', full_four_step_tracking_passed=False,
                # [archived branch] nominal_torque_projection_requested=args.mocap_first_step_project_torque,
                # [archived branch] step_valid=step_valid, landing_error_m=landing_error,
                # [archived branch] measured_touchdown=bool(source_step_rollout is not None and source_step_rollout.touchdown),
                # [archived branch] controller_valid=bool(motion_tracking_valid.all()),
                # [archived branch] unloading=probe_unload_report),indent=2)+"\n")
    final_right_load_fraction = float("nan")
    # [archived branch] if args.single_support_test and contact_cycle is None:
        # [archived branch] recent = tracking_history[-50:]
        # [archived branch] support_index = 0 if args.single_support_test == "left" else 1
        # [archived branch] hold_valid = (single_release_time is not None and
                      # [archived branch] (executed_steps - 1) * PD_TIMESTEP >= single_release_time + 3. and len(recent) == 50)
        # [archived branch] for sample in recent:
            # [archived branch] feet = sample["feet"].reshape(2, 7)
            # [archived branch] hold_valid &= (bool(np.all(sample["commanded_mode"] == support_index + 1))
                           # [archived branch] and np.linalg.norm(sample["com"] - sample["com_target"]) <= .03
                           # [archived branch] and feet[1-support_index, 2] - single_initial_feet[0, 1-support_index, 2].item() >= .025
                           # [archived branch] and sample["foot_forces"][1-support_index, 2] < 10.)
        # numpy comparisons above yield np.bool_; torch in-place boolean AND
        # must receive a Python bool rather than numpy scalar conversion.
        # [archived branch] hold_valid = bool(hold_valid)
        # [archived branch] motion_tracking_valid &= hold_valid
        # [archived branch] print(f"single support validation: valid={hold_valid}, release_time={single_release_time}; requires 3s post-release and final 1s stable clearance/COM/contact")
    if contact_cycle is not None:
        cycle_valid = contact_cycle.phase == "done" and contact_cycle.ever_clear
        recent = tracking_history[-50:]
        cycle_valid &= len(recent) == 50 and all(
            np.all(s["commanded_mode"] == 3) and np.all(s["foot_forces"][:, 2] >= 20.)
            and np.linalg.norm(s["com"] - s["com_target"]) <= .03 for s in recent)
        motion_tracking_valid &= bool(cycle_valid)
        print(f"contact cycle validation: valid={bool(cycle_valid)}, phase={contact_cycle.phase}")
    # [archived branch] if args.weight_transfer_test:
        # [archived branch] recent_forces = np.asarray([sample["foot_forces"][:, 2]
                                    # [archived branch] for sample in tracking_history[-10:]])
        # [archived branch] mean_load = np.maximum(recent_forces, 0.).mean(axis=0)
        # [archived branch] final_right_load_fraction = float(mean_load[1] / max(mean_load.sum(), 1e-8))
        # [archived branch] transfer_valid = (
            # [archived branch] final_right_load_fraction >= .8
            # [archived branch] and center_of_mass_error_norm.max().item() <= .03
            # [archived branch] and (executed_steps - 1) * PD_TIMESTEP >= args.motion_warmup_duration + 3.
            # [archived branch] and all(np.all(sample["commanded_mode"] == 3) for sample in tracking_history)
        # [archived branch] )
        # [archived branch] motion_tracking_valid &= transfer_valid
        # [archived branch] print(f"weight transfer: right load fraction={final_right_load_fraction:.3f}, "
              # [archived branch] f"valid={transfer_valid} (requires >=0.8, COM error <=0.03 m, completed ramp, double support)")
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
            joint_velocity_prediction,
            (1, 29),
        ),
        "joint_torque_feedforward": (
            joint_torque_feedforward,
            (1, 29),
        ),
        "joint_torque_pd": (joint_torque_pd, (1, 29)),
        "joint_position_desired": (
            joint_position_prediction,
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
    # [archived branch] validation_scope = (
        # [archived branch] "first_source_step_probe" if args.mocap_first_step_report is not None else
        # [archived branch] "double_support_unloading_only" if args.mocap_unload_test else
        # [archived branch] "unified_first_source_step" if args.unified_source_step_test else
        # [archived branch] "unified_first_transfer" if args.unified_transfer_test else
        # [archived branch] "single_support" if args.single_support_test else
        # [archived branch] "weight_transfer" if args.weight_transfer_test else
        # [archived branch] "fixed_joint_servo" if args.fixed_joint_test else
        # [archived branch] "static_reference" if motion_trajectory is None or motion_frame == motion_start_frame else
        # [archived branch] "motion_segment"
    # [archived branch] )
    validation_scope = 'static_reference' if motion_trajectory is None or motion_frame == motion_start_frame else 'motion_segment'
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
            joint_position_prediction.detach().cpu().numpy()
        ),
        joint_velocity_desired=(
            joint_velocity_prediction.detach().cpu().numpy()
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
        integration_initial_position=initial_joint_position.cpu().numpy(),
        integration_initial_velocity=initial_joint_velocity.cpu().numpy(),
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
    # [archived branch] print(
        # [archived branch] "[pass] numerical pipeline: pelvis + COM + feet + hand Fabric -> "
        # [archived branch] + ("metric summation -> constrained dynamics -> persistent MIT integration" if args.g1_md else
         # [archived branch] "contact projection -> dynamics acceleration QP -> integration")
    # [archived branch] )
    print('[pass] numerical pipeline: feedback tasks -> constrained dynamics QP -> direct joint effort')
    tracking_failed = (
        motion_trajectory is not None
        and (not bool(motion_tracking_valid.all())
             or ((args.until_motion_end or args.track_full_motion) and not stopped_at_motion_end))
    )
    # [archived branch] if args.mocap_unload_test and args.mocap_first_step_report is None:
        # [archived branch] from analyze_contact_transfer import summarize_unloading
        # [archived branch] with np.load(args.output_path) as recording:
            # [archived branch] unload_report = summarize_unloading(recording)
        # [archived branch] args.output_path.with_suffix(".unloading.json").write_text(json.dumps(unload_report, indent=2)+"\n")
        # [archived branch] tracking_failed |= not unload_report["unloading_passed"]
        # [archived branch] print("Double-support unloading: "+json.dumps(unload_report), flush=True)
    # [archived branch] if args.unified_transfer_test and not args.unified_source_step_test:
        # [archived branch] from analyze_contact_transfer import summarize_transfer
        # [archived branch] with np.load(args.output_path) as recording:
            # [archived branch] transfer_report = summarize_transfer(recording)
        # [archived branch] args.output_path.with_suffix(".transfer.json").write_text(json.dumps(transfer_report, indent=2)+"\n")
        # [archived branch] tracking_failed |= not transfer_report["transfer_passed"]
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
        # [archived branch] if not args.mocap_unload_test:
        tracking_failed |= not steps_report['steps_tracking_passed']
    # [archived branch] if (args.until_motion_end and not args.tracking_steps) or args.track_full_motion:
    from analyze_motion_tracking import summarize_tracking, summarize_full_motion
    with np.load(args.output_path) as recording:
        # [archived branch] full_tracking = (summarize_full_motion(recording, json.loads(
            # [archived branch] args.output_path.with_suffix('.steps_plan.json').read_text()))
            # [archived branch] if args.track_full_motion else summarize_tracking(recording))
        full_tracking = summarize_full_motion(recording, json.loads(args.output_path.with_suffix('.steps_plan.json').read_text()))
    if fidelity_stop is not None:
        full_tracking['first_fidelity_exceeded']=fidelity_stop
        # [archived branch] if args.stop_at_fidelity_loss:
        full_tracking.update(stop_reason='Original-source fidelity limit exceeded',fidelity_stop=fidelity_stop)
        # [archived branch] else:
            # [archived branch] full_tracking['diagnostic_continuation_after_fidelity_loss']=True
    full_report_path = args.output_path.with_suffix('.tracking.json')
    full_report_path.write_text(json.dumps(full_tracking,indent=2)+'\n')
    # [archived branch] passed = full_tracking['full_motion_tracking_passed' if args.track_full_motion else 'full_tracking_passed']
    passed = full_tracking['full_motion_tracking_passed']
    tracking_failed |= not passed
    print(f"[{'pass' if passed else 'fail'}] full source coverage and PKL fidelity: {full_report_path}")
    if motion_trajectory is not None:
        print(
            f"[{'pass' if bool(motion_tracking_valid.all()) else 'fail'}] {validation_scope} controller validation (not PKL fidelity)"
        )
        # [archived branch] if args.until_motion_end and not stopped_at_motion_end:
            # [archived branch] print(f"[fail] simulation budget exhausted before stable source completion: frame {frame_position:.2f}/{len(motion_trajectory)-1}")
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
