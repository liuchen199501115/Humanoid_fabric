"""Offline whole-body iteration with explicit rejection and final dense audit."""
import numpy as np
import torch

from .source_step import centroidal_support_check


@torch.no_grad()
def evaluate_candidate(plan, times, velocity_limits, dynamics_limits=None):
    plan.q = plan.initial.q.clone()
    model = plan.model
    rows = []
    for time in times:
        try:
            reference = plan.sample(float(time))
        except RuntimeError as exc:
            raise RuntimeError(f"Candidate sample t={float(time):.6f}s: {exc}") from exc
        state, _, _ = model.evaluate(reference.q, reference.v, plan.fabric.integrate_root_position,
                                     plan.fabric.dynamics_tasks.config['difference_step'])
        force, cop, passed = centroidal_support_check(model, reference, state, plan.feet[plan.support])
        rate = ((state['momentum_matrix']@reference.a[..., None]).squeeze(-1)
                +state['momentum_matrix_bias'])
        extra = {}
        if dynamics_limits is not None:
            from .inverse_dynamics import required_single_support
            from fabrics_sim.solver.AccelerationQP import build_single_support_wrench_constraints
            physical = required_single_support(model, reference, plan.fabric.integrate_root_position,
                                              plan.feet[plan.support], dynamics_limits['armature'])
            G, b = build_single_support_wrench_constraints(contact_mode=torch.tensor([plan.support+1],device=reference.q.device),
                foot_rotation_world=state['rotations'][:,plan.feet], toe_length=.12, heel_length=.05, half_width=.025,
                sole_depth=.035, coupled_yaw=True)
            wrench = reference.q.new_zeros(1,12)
            wrench[:,6*plan.support:6*(plan.support+1)] = physical['wrench']
            residual = ((G@wrench[...,None]).squeeze(-1)-b).clamp_min(0)[:,20*plan.support:20*(plan.support+1)]
            scale = residual.new_full((1,20), float(model.com_task.link_masses.sum())*9.81)
            scale[:,6:] *= .1
            joint_violation = 0.
            joint_detail = None
            if 'joint_bounds' in dynamics_limits:
                from fabrics_sim.solver.joint_control import measured_joint_acceleration_bounds
                bounds = dynamics_limits['joint_bounds']
                lower, upper = measured_joint_acceleration_bounds(
                    reference.q[:, 7:], reference.v[:, 6:], bounds['lower'], bounds['upper'],
                    bounds['dt'], barrier_bandwidth=bounds['bandwidth'],
                    velocity_damper_time=bounds.get('velocity_damper_time'),
                    impact_recovery=bounds.get('impact_recovery', False))
                violation = torch.maximum(lower-reference.a[:, 6:], reference.a[:, 6:]-upper).clamp_min(0)
                normalized = violation/dynamics_limits['acceleration'][6:]
                joint_violation = float(normalized.max())
                j = int(normalized.argmax())
                joint_detail = dict(index=j,position=float(reference.q[0,7+j]),velocity=float(reference.v[0,6+j]),
                    acceleration=float(reference.a[0,6+j]),lower_acceleration=float(lower[0,j]),upper_acceleration=float(upper[0,j]))
            extra = dict(
                         acceleration_limit_checked='acceleration' in dynamics_limits,
                         joint_braking_checked='joint_bounds' in dynamics_limits,
                         joint_bound_detail=joint_detail,
                         peak_torque_joint_index=int((physical['torque'].abs()/dynamics_limits['effort']).argmax()),
                         peak_wrench_constraint_index=int((residual/scale).argmax()),
                         support_contact_velocity_max=float(physical['contact_velocity'].abs().max()),
                         support_contact_acceleration_max=float(physical['contact_acceleration'].abs().max()),
                         joint_bound_score=joint_violation,exact_wrench_violation=float(residual.max()),
                         exact_wrench_score=float((residual/scale).max()),
                         torque_ratio=float((physical['torque'].abs()/dynamics_limits['effort']).max()),
                         acceleration_ratio=float((reference.a.abs()/dynamics_limits['acceleration']).max())
                             if 'acceleration' in dynamics_limits else 0.)
        swing_link = plan.feet[1-plan.support]
        sole = reference.q.new_tensor([[-.05,-.025,-.035],[-.05,.025,-.035],
                                       [.12,-.025,-.035],[.12,.025,-.035]])
        corner_z = (state['transforms'][:,swing_link,2,None]
                    +state['rotations'][:,swing_link,2,:]@sole.T)
        rows.append(dict(swing_sole_min_height=float(corner_z.min()), **extra,time=float(time), q=reference.q.numpy(), v=reference.v.numpy(), a=reference.a.numpy(),
            frame=plan.start_frame+(plan.end_frame-plan.start_frame)*plan.progress(float(time))[0],
            com=state['com'].numpy(), com_velocity=(state['com_jac']@reference.v[..., None]).squeeze(-1).numpy(),
            angular_momentum_rate=rate.numpy(), cop=cop.numpy(), force=force.numpy(),
            wrench_passed=bool(passed.all()), velocity_ratio=float((reference.v[:, 6:].abs()/velocity_limits).max())))
    return rows


def constraint_components(rows):
    """Nonnegative normalized violation arrays, retaining every constraint family."""
    if not rows:raise ValueError("At least one candidate sample is required")
    cop = np.asarray([r['cop'] for r in rows]).reshape(-1, 2)
    force = np.asarray([r['force'] for r in rows]).reshape(-1, 3)
    velocity = np.asarray([r['velocity_ratio'] for r in rows])
    if not all(np.isfinite(value).all() for value in (cop, force, velocity)):
        raise ValueError('Candidate constraints must be finite')
    outside = np.maximum(np.array([-.04, -.02])-cop, cop-np.array([.11, .02]))/np.array([.15, .04])
    friction = np.abs(force[:, :2])/np.maximum(force[:, 2:3], 1e-8)-.6
    exact = np.asarray([[r.get('exact_wrench_score', 0.), r.get('torque_ratio', 0.), r.get('acceleration_ratio', 0.), r.get('joint_bound_score', 0.)] for r in rows])
    if not np.isfinite(exact).all():
        raise ValueError('Candidate dynamics constraints must be finite')
    swing_height=np.asarray([r.get('swing_sole_min_height',0.) for r in rows])
    if not np.isfinite(swing_height).all():
        raise ValueError('Swing sole heights must be finite')
    values = dict(swing_ground=(-swing_height-2e-5)/.01,
                  support_region=outside, friction=friction,
                  velocity=velocity-1., normal_force=(20.-force[:,2])/100.,
                  exact_wrench=exact[:,0], torque=exact[:,1]-1.,
                  acceleration=exact[:,2]-1., joint_braking=exact[:,3])
    return {key:np.maximum(value,0.) for key,value in values.items()}


def constraint_score(rows):
    """Unchanged strict gate: maximum over all normalized necessary bounds."""
    return max(float(value.max()) for value in constraint_components(rows).values())


def constraint_family_penalty(rows):
    """Mean squared penalty per family; a larger violation cannot hide another."""
    return {key:float(np.mean(value**2)) for key,value in constraint_components(rows).items()}


def correct_candidate(plan, times, velocity_limits, iterations=8, progress=None, dynamics_limits=None):
    """Audit a full-body candidate without a reduced-model correction step.

    The iterations argument remains for existing audit callers; reference
    optimization is owned by the full-body parameter optimizer.
    """
    if type(iterations) is not int or iterations < 0:
        raise ValueError('Iterations must be a nonnegative integer')
    rows = evaluate_candidate(plan, times, velocity_limits, dynamics_limits)
    score = constraint_score(rows)
    entry = dict(iteration=0, maximum_normalized_violation=score,
                 wrench_pass_fraction=float(np.mean([row['wrench_passed'] for row in rows])),
                 max_velocity_ratio=max(row['velocity_ratio'] for row in rows))
    if progress is not None:
        progress(entry)
    plan.q = plan.initial.q.clone()
    return rows, dict(history=[entry], rejected=[],
                     sampled_necessary_constraints_passed=score == 0.)


def relative_peak_objective(peaks, baseline, zero_tolerance=1e-8):
    """Balance improvements relative to each starting peak, without hiding a family.

    A maximum ratio below one means every initially violated family improved.
    Previously satisfied families retain a numerical tolerance only. This search
    objective never replaces the absolute physical acceptance gate.
    """
    if not peaks or peaks.keys() != baseline.keys():
        raise ValueError('Current and baseline peaks must have identical nonempty keys')
    current=np.asarray(list(peaks.values()),dtype=float)
    initial=np.asarray([baseline[k] for k in peaks],dtype=float)
    if (not np.isfinite(zero_tolerance) or zero_tolerance <= 0
            or not np.isfinite(current).all() or not np.isfinite(initial).all()
            or (current < 0).any() or (initial < 0).any()):
        raise ValueError('Peaks must be finite and nonnegative')
    ratios=current/np.maximum(initial,zero_tolerance)
    return float(ratios.max()+.01*np.mean(ratios**2))
