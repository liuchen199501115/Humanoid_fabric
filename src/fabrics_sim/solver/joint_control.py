"""Explicit joint servo utilities in floating-base generalized order."""

import torch
import math


def bounded_reference_step(position, velocity, target, dt, time_constant,
                           max_acceleration=.5, max_speed=.2):
    """Second-order reference filter with per-axis speed/acceleration bounds."""
    if any(not math.isfinite(x) or x <= 0 for x in (dt, time_constant, max_acceleration, max_speed)):
        raise ValueError("Reference filter parameters must be finite and positive.")
    if position.shape != velocity.shape or position.shape != target.shape:
        raise ValueError("Reference position, velocity and target must share a shape.")
    if not all(torch.isfinite(x).all() for x in (position, velocity, target)):
        raise ValueError("Reference filter states must be finite.")
    if (velocity.abs() > max_speed + 1e-7).any():
        raise ValueError("Initial reference velocity exceeds the speed limit.")
    omega = 1. / time_constant
    acceleration = (omega**2 * (target - position) - 2. * omega * velocity).clamp(
        -max_acceleration, max_acceleration)
    acceleration = torch.maximum(acceleration, (-max_speed - velocity) / dt)
    acceleration = torch.minimum(acceleration, (max_speed - velocity) / dt)
    next_velocity = velocity + dt * acceleration
    return position + dt * next_velocity, next_velocity, acceleration


def explicit_pd_gains(mass, nominal_kp, nominal_kd, dt, bandwidth=30.0):
    """Bound gains using free-system effective inertia and a coupled step budget.

    This is a local discrete-time safeguard, not a contact stability proof.
    The floating base is included when estimating each joint's effective inertia.
    """
    if dt <= 0 or bandwidth <= 0:
        raise ValueError("dt and bandwidth must be positive")
    inverse_mass = torch.linalg.inv(mass)
    return _pd_gains_from_mobility(inverse_mass, nominal_kp, nominal_kd, dt, bandwidth)


def _pd_gains_from_mobility(inverse_mass, nominal_kp, nominal_kd, dt, bandwidth):
    inertia = inverse_mass.diagonal(dim1=-2, dim2=-1)[:, 6:].clamp_min(1e-12).reciprocal()
    kp = torch.minimum(nominal_kp, inertia * bandwidth**2)
    kd = torch.minimum(nominal_kd, 2.0 * torch.sqrt(kp * inertia))
    # A symmetric similarity transform bounds the coupled feedback modes.
    budget = kd * dt + kp * dt**2
    root = torch.sqrt(budget)
    normalized = root[:, :, None] * inverse_mass[:, 6:, 6:] * root[:, None, :]
    largest = torch.linalg.eigvalsh(normalized).amax(dim=-1)
    scale = torch.clamp(0.5 / largest.clamp_min(1e-12), max=1.0)[:, None]
    return kp * scale, kd * scale


def contact_servo_gain_bank(mass, contact_jacobian, row_mask, nominal_kp, nominal_kd,
                            dt, bandwidth=30.):
    """Gain sets for free/left/right/double support of one floating-base robot.

    Constrained mobility is M^-1 - M^-1 J.T (J M^-1 J.T)^-1 J M^-1.
    Computing in float64 avoids cancellation at the low-inertia ankle axes.
    Measured loss of support must select a lower-contact gain set immediately;
    this local linear safeguard does not establish contact stability.
    """
    if mass.shape != (1,35,35) or contact_jacobian.shape != (1,12,35):
        raise ValueError('Contact servo gains require one G1 mass/Jacobian snapshot')
    if row_mask is None:
        row_mask = torch.ones(1,12,device=mass.device,dtype=torch.bool)
    if row_mask.shape != (1,12) or row_mask.dtype != torch.bool or dt <= 0 or bandwidth <= 0:
        raise ValueError('Invalid contact servo mask or period')
    inverse = torch.linalg.inv(mass.double()).expand(4,-1,-1)
    modes = torch.arange(4, device=mass.device)
    active = torch.stack(((modes & 1) != 0, (modes & 2) != 0),-1).repeat_interleave(6,-1)
    active = active & row_mask
    J = contact_jacobian.double().expand(4,-1,-1) * active[...,None]
    mobility = J @ inverse @ J.transpose(-1,-2) + torch.diag_embed((~active).double())
    response = inverse @ J.transpose(-1,-2)
    constrained = inverse-response @ torch.linalg.solve(mobility, J @ inverse)
    constrained = .5*(constrained+constrained.transpose(-1,-2))
    kp,kd = _pd_gains_from_mobility(constrained,nominal_kp.double(),nominal_kd.double(),dt,bandwidth)
    return kp.to(mass.dtype),kd.to(mass.dtype)


def implicit_servo_response(mass, kp, kd, dt):
    """Torque response for backward-Euler joint error feedback.

    tau = [I + diag(dt*Kd+dt^2*Kp) M_free^-1_jj]^-1
          [Kp*position_error + (Kd+dt*Kp)*velocity_error].
    Use free mobility even when requesting stance gains: nonzero normal load
    alone does not guarantee that all foot rotations remain constrained.
    """
    if (mass.ndim != 3 or mass.shape[-2:] != (35,35) or kp.shape != kd.shape
            or kp.ndim != 2 or kp.shape[-1] != 29 or dt <= 0
            or not all(torch.isfinite(x).all() for x in (mass,kp,kd))
            or (kp < 0).any() or (kd < 0).any()):
        raise ValueError('Invalid implicit servo model')
    inverse = torch.linalg.inv(mass.double())[:,6:,6:]
    budget = dt*kd.double()+dt**2*kp.double()
    identity = torch.eye(29,device=mass.device,dtype=torch.float64).expand(len(kp),-1,-1)
    response = torch.linalg.solve(identity+budget[...,None]*inverse, identity)
    return response.to(mass.dtype)


def integration_acceleration_bounds(position, velocity, lower, upper, speed_limit, acceleration_limit, dt):
    """Acceleration bounds for the next semi-implicit command and stopping room.

    Fail explicitly if no bounded acceleration can maintain the command
    limits. Never hide an infeasible command by clipping the integrated pose.
    """
    if dt <= 0:
        raise ValueError("dt must be positive")
    distance_lo = (position - lower).clamp_min(0)
    distance_hi = (upper - position).clamp_min(0)
    brake_step = acceleration_limit * dt
    speed_hi = torch.minimum(speed_limit, torch.sqrt(brake_step**2 + 2 * acceleration_limit * distance_hi) - brake_step)
    speed_lo = -torch.minimum(speed_limit, torch.sqrt(brake_step**2 + 2 * acceleration_limit * distance_lo) - brake_step)
    minimum = torch.maximum(-acceleration_limit, (speed_lo - velocity) / dt)
    maximum = torch.minimum(acceleration_limit, (speed_hi - velocity) / dt)
    minimum = torch.maximum(minimum, (lower - position - dt * velocity) / dt**2)
    maximum = torch.minimum(maximum, (upper - position - dt * velocity) / dt**2)
    if (minimum > maximum).any():
        raise ValueError("Joint command cannot stay inside limits with the allowed acceleration.")
    return minimum, maximum


def integrated_servo_model(position, velocity, measured_position, measured_velocity, kp, kd, dt):
    """Exact affine PD law for the NEXT semi-implicit joint target.

    tau_PD = bias + gain * a_command. Existing command tracking errors stay
    in the feedback; they are not subtracted from the feedforward command.
    """
    if dt <= 0:
        raise ValueError("dt must be positive")
    bias = kp * (position + dt * velocity - measured_position) + kd * (velocity - measured_velocity)
    gain = (kp * dt**2 + kd * dt).expand_as(position)
    return bias, gain


def measured_joint_acceleration_bounds(position, velocity, lower, upper, dt,
                                      braking_acceleration=5., margin=.01,
                                      velocity_damper_time=None, barrier_bandwidth=None,
                                      impact_recovery=False):
    """One-step velocity bounds from remaining physical stopping distance.

    These constrain predicted acceleration, not the command integrator.
    Outside the conservative stopping-speed envelope, recover at the braking
    budget while keeping next-step position within the limit margin. This is
    a discrete prediction, not a feasibility guarantee under contact changes.
    Optional velocity_damper_time adds a linear distance-to-limit speed bound;
    it can require stronger braking and is not capped at the recovery budget.
    Optional impact_recovery also demands enough braking for the measured
    outward speed and remaining distance after an impact. Effort/acceleration
    limits still belong to the QP; infeasible recovery is not silently clipped.
    """
    if dt <= 0 or braking_acceleration <= 0 or margin < 0:
        raise ValueError("Invalid measured-joint braking parameters.")
    if ((upper - lower) <= 2 * margin).any():
        raise ValueError("Joint margin consumes the available travel.")
    speed_lo = -torch.sqrt(2 * braking_acceleration * (position-lower-margin).clamp_min(0))
    speed_hi = torch.sqrt(2 * braking_acceleration * (upper-margin-position).clamp_min(0))
    # An impact can put the measured velocity outside this conservative
    # envelope. Demand bounded braking toward it, not instantaneous return.
    # Actual next-step position limits remain hard even during recovery.
    speed_lo = torch.minimum(speed_lo, velocity + braking_acceleration * dt)
    speed_hi = torch.maximum(speed_hi, velocity - braking_acceleration * dt)
    minimum = (speed_lo-velocity)/dt
    maximum = (speed_hi-velocity)/dt
    minimum = torch.maximum(minimum, (lower+margin-position-dt*velocity)/dt**2)
    maximum = torch.minimum(maximum, (upper-margin-position-dt*velocity)/dt**2)
    if impact_recovery:
        # A contact impulse can invalidate the nominal stopping envelope and
        # the initial-state condition of a second-order distance barrier.
        # Constant braking v^2/(2*d) suffices in the ideal continuous model;
        # semi-implicit integration is conservative for a non-reversing stop.
        # Cap only at stopping in one tick, not at the actuator's ability:
        # the unchanged QP must reject a physically unavailable deceleration.
        outward_lo=(-velocity).clamp_min(0)
        outward_hi=velocity.clamp_min(0)
        required_lo=torch.minimum(outward_lo.square()/(2*(position-lower-margin).clamp_min(1e-8)),
                                  outward_lo/dt)
        required_hi=torch.minimum(outward_hi.square()/(2*(upper-margin-position).clamp_min(1e-8)),
                                  outward_hi/dt)
        outside_lo=outward_lo.square()>2*braking_acceleration*(position-lower-margin).clamp_min(0)
        outside_hi=outward_hi.square()>2*braking_acceleration*(upper-margin-position).clamp_min(0)
        minimum=torch.where(outside_lo,torch.maximum(minimum,required_lo),minimum)
        maximum=torch.where(outside_hi,torch.minimum(maximum,-required_hi),maximum)
    if velocity_damper_time is not None:
        if not math.isfinite(velocity_damper_time) or velocity_damper_time <= 0:
            raise ValueError('Velocity damper time must be finite and positive')
        # A linear distance/speed envelope starts stronger braking BEFORE
        # the last position interval. Unlike the fallback above, recovery is
        # not capped at 5 rad/s^2 once a disturbance leaves that envelope.
        # This adds bounds only; physical feasibility is still checked by QP.
        minimum = torch.maximum(minimum,
            (-(position-lower-margin).clamp_min(0)/velocity_damper_time-velocity)/dt)
        maximum = torch.minimum(maximum,
            ((upper-margin-position).clamp_min(0)/velocity_damper_time-velocity)/dt)
    if barrier_bandwidth is not None:
        if not math.isfinite(barrier_bandwidth) or barrier_bandwidth <= 0:
            raise ValueError('Joint barrier bandwidth must be finite and positive')
        # Relative-degree-two joint-distance barrier: h''+2*w*h'+w^2*h>=0.
        # Unlike one-step velocity recovery, this asks for increasing braking
        # before the last position interval. Reserve 0.05 rad where travel
        # permits; measured impacts/model error still require runtime checking.
        room = torch.minimum(torch.full_like(lower, max(margin, .05)),
                             (upper-lower)*.25)
        omega = barrier_bandwidth
        minimum = torch.maximum(minimum,
            -2*omega*velocity-omega**2*(position-lower-room))
        maximum = torch.minimum(maximum,
            -2*omega*velocity+omega**2*(upper-room-position))
    if (minimum > maximum).any():
        bad=minimum>maximum
        raise ValueError(
            'Measured joint state cannot satisfy next-step position limits: '
            f'indices={bad.nonzero().tolist()}, position={position[bad].tolist()}, '
            f'velocity={velocity[bad].tolist()}, minimum_acceleration={minimum[bad].tolist()}, '
            f'maximum_acceleration={maximum[bad].tolist()}.')
    return minimum, maximum


def synchronize_command_acceleration(reference, position, velocity, measured_position,
                                     measured_velocity, stiffness=1., damping=2., limit=5.):
    """Continuous reference-integrator feedback, applied BEFORE constrained QP.

    The default synchronization bandwidth is deliberately below the task and
    joint servo bandwidths. In the ideal unconstrained case the error follows a
    damped second-order system. No position or velocity is overwritten.
    QP constraints take precedence over this soft acceleration request.
    """
    correction = (stiffness * (measured_position - position)
                  + damping * (measured_velocity - velocity)).clamp(-limit, limit)
    result = reference.clone()
    result[:, 6:] += correction
    return result


def joint_limit_avoidance_request(position, velocity, lower, upper, weight,
                                  bandwidth=8., margin=.05, acceleration_limit=50.):
    """Soft anticipatory braking; physical joint bounds remain separate.

    Distance-barrier acceleration is requested before a joint reaches its
    stop. As a soft task it can yield during a measured landing impulse,
    unlike imposing an immediately infeasible additional hard bound.
    """
    from fabrics_sim.fabrics.task_request import TaskRequest
    if (position.ndim != 2 or position.shape[1] != 29
            or any(x.shape != position.shape for x in (velocity, lower, upper))
            or not all(torch.isfinite(x).all() for x in (position, velocity, lower, upper))
            or (upper <= lower).any()
            or not all(math.isfinite(x) and x > 0 for x in
                       (weight, bandwidth, margin, acceleration_limit))):
        raise ValueError('Invalid joint-limit avoidance inputs')
    room = torch.minimum(torch.full_like(lower, margin), .25*(upper-lower))
    brake_low = (-2*bandwidth*velocity-bandwidth**2*(position-lower-room)).clamp_min(0.)
    brake_high = (-2*bandwidth*velocity+bandwidth**2*(upper-room-position)).clamp_max(0.)
    desired = (brake_low+brake_high).clamp(-acceleration_limit, acceleration_limit)
    # Reach the full metric by 1 rad/s^2; otherwise a persistent competing
    # posture request can push the soft equilibrium beyond the reserved room.
    gate = desired.abs().clamp(max=1.)
    A = position.new_zeros(position.shape[0], 29, 35)
    A[:, :, 6:] = torch.eye(29, device=position.device, dtype=position.dtype)
    return TaskRequest('joint_limit_avoidance', A, torch.zeros_like(position), desired,
                       torch.diag_embed(weight*gate), gate > 0, 'high')


def servo_health(velocity, torque, unclipped):
    """Detect sustained high-speed reversals and prolonged saturation.

    Arrays have shape [physics_steps, batch, joints]. Five consecutive
    reversals above 5 rad/s flag the previously observed step-to-step limit
    cycle. A joint clipped for >5% of samples fails the saturation check.
    These are explicit experiment acceptance thresholds, not hardware limits.
    """
    finite = torch.isfinite(velocity).all() & torch.isfinite(torque).all() & torch.isfinite(unclipped).all()
    flips = (velocity[1:] * velocity[:-1] < 0) & (velocity[1:].abs() > 5) & (velocity[:-1].abs() > 5)
    sustained = (flips.unfold(0, 5, 1).all(dim=-1).any()
                 if flips.shape[0] >= 5 else torch.tensor(False, device=velocity.device))
    saturation_fraction = ((torque - unclipped).abs() > 1e-4).to(velocity.dtype).mean(dim=0).amax()
    return finite & ~sustained & (saturation_fraction <= .05), sustained, saturation_fraction


def joint_velocity_constraints(velocity, speed_limits, dt):
    """C @ physical_qdd <= d enforces |v + dt*qdd_joint| <= vmax.

    No position-dependent braking, fixed acceleration cap, or base speed bound.
    Already excessive speed requests recovery in one step; infeasibility is not
    hidden by clipping the measured state or relaxing the torque constraints.
    """
    if velocity.ndim != 2 or velocity.shape[1] != 29:
        raise ValueError('Expected G1 joint velocity [B,29]')
    limits=torch.as_tensor(speed_limits,device=velocity.device,dtype=velocity.dtype)
    if limits.shape == (29,):limits=limits[None].expand_as(velocity)
    if (limits.shape != velocity.shape or not math.isfinite(dt) or dt <= 0
            or not torch.isfinite(velocity).all() or not torch.isfinite(limits).all()
            or (limits <= 0).any()):
        raise ValueError('Velocity and positive speed limits must be finite; timestep must be positive')
    selector=velocity.new_zeros(velocity.shape[0],29,35)
    selector[:,:,6:]=torch.eye(29,device=velocity.device,dtype=velocity.dtype)*dt
    return torch.cat((selector,-selector),dim=1),torch.cat((limits-velocity,limits+velocity),dim=1)
