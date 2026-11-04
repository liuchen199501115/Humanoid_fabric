"""Isaac controller configuration for examples/g1.md.

Natural-form composition supplies the acceleration objective. A constrained
full-body dynamics solve is still necessary; an arbitrary Fabric acceleration
need not admit unilateral contact forces or actuator torques.
"""


def configure_fabric_mit(args):
    """Select persistent integration and explicit PD accounted for in dynamics."""
    incompatible = (
        'fixed_joint_test', 'mocap_unload_test', 'mocap_first_step_report',
        'source_swing_pose_priority', 'source_swing_position_priority',
        'separate_balance_priority', 'balance_body_priority',
        'balance_priority', 'transfer_balance_priority', 'swing_clearance_priority',
        'swing_position_priority', 'swing_pose_priority',
    )
    selected = [name for name in incompatible if getattr(args, name, False)]
    if selected:
        raise ValueError('--g1-md excludes diagnostic/priority modes: ' + ', '.join(selected))
    args.task_profile = 'dynamics-aware'
    args.direct_wbc = False
    # These alternate servos predict from live state rather than using the
    # persistent integrated desired state required by the MIT command path.
    args.tracking_servo = False
    args.contact_aware_servo = False
    args.nominal_implicit_servo = False
    args.control_frequency = 1000
    args.task_track_nominal = False
    args.qp_track_predicted_base = True
    args.fast_command_sync = False
    if getattr(args, 'mocap_pipeline', False) and args.mocap_unload_weight == 0:
        args.mocap_unload_weight = 1e-4


def compose_task_acceleration(requests, velocity, regularization, *, return_metric=False):
    """Pull back and sum ALL current leaves, including late contact/barrier tasks."""
    import math
    import torch
    if not math.isfinite(regularization) or regularization <= 0:
        raise ValueError('Fabric root regularization must be finite and positive')
    metric = regularization * torch.eye(velocity.shape[-1], device=velocity.device,
                                        dtype=torch.float64).expand(velocity.shape[0], -1, -1).clone()
    force = torch.zeros_like(velocity, dtype=torch.float64)
    for request in requests:
        pulled_metric, pulled_force = request.natural_form(dtype=torch.float64)
        metric = metric + pulled_metric
        force = force + pulled_force
    # Weighted Gram products can differ in their last bits across the diagonal.
    # Accumulate in double precision, then explicitly preserve symmetry. This
    # changes neither task weights nor physical feasibility tolerances.
    metric = .5 * (metric + metric.transpose(-1, -2))
    acceleration = torch.linalg.solve(metric, -force.unsqueeze(-1)).squeeze(-1)
    if not torch.isfinite(acceleration).all():
        raise FloatingPointError('Fabric composition produced non-finite acceleration')
    return (acceleration, metric) if return_metric else acceleration.to(velocity.dtype)


def compose_task_policy(requests, velocity, regularization):
    """Return one root policy retaining the summed metric for constrained dynamics.

    With predicted-acceleration tracking, the QP applies this metric to all
    floating-base and joint coordinates including the response to MIT feedback.
    The individual leaf objectives are not added again.
    """
    import torch
    from fabrics_sim.fabrics.task_request import TaskRequest
    acceleration, metric = compose_task_acceleration(
        requests, velocity, regularization, return_metric=True)
    identity = torch.eye(velocity.shape[-1],device=velocity.device,dtype=metric.dtype)
    root = TaskRequest('fabric_composed', identity.expand(velocity.shape[0],-1,-1),
                       torch.zeros_like(acceleration), acceleration, metric,
                       torch.ones_like(velocity,dtype=torch.bool))
    return acceleration.to(velocity.dtype), root
