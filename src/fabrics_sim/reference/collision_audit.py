"""Check reference acceleration against the enabled runtime collision barrier."""
import torch
from fabrics_sim.solver.collision_barrier import wrist_collision_barrier


def collision_reference_check(fabric, reference):
    tasks = fabric.dynamics_tasks
    if not (getattr(tasks, 'collision_enabled', False) and getattr(tasks, 'wrist_hip_barrier', False)):
        return dict(checked=False, passed=True, maximum_violation=0., minimum_cover_distance=None)
    epsilon = tasks.config['difference_step']
    state, plus, minus = tasks.model.evaluate(reference.q, reference.v,
                                               fabric.integrate_root_position, epsilon)
    geometry=getattr(tasks,'collision_geometry',tasks._sphere_geometry)
    pair_count=getattr(tasks,'barrier_pair_count',len(tasks.pairs))
    distance, jacobian = geometry(state)
    _, jp = geometry(plus)
    _, jm = geometry(minus)
    bias = (((jp-jm)/(2*epsilon))@reference.v[..., None]).squeeze(-1)
    speed = (jacobian@reference.v[..., None]).squeeze(-1)
    G, b = wrist_collision_barrier(distance, jacobian, bias, speed,
        tasks.original_collision_pairs, pair_count, tasks.config['collision']['safe_distance'])
    residual = ((G@reference.a[..., None]).squeeze(-1)-b).clamp_min(0)
    violation = float(residual.max()) if residual.numel() else 0.
    covers = distance[:, tasks.original_collision_pairs:pair_count]
    minimum = float(covers.min()) if covers.numel() else None
    return dict(checked=True, passed=violation <= 1e-5, maximum_violation=violation,
                minimum_cover_distance=minimum)


def audit_collision_rows(fabric, rows, template):
    from .whole_body import WholeBodyReference
    checked=False; maximum=0.; minimum=None; count=0; worst_time=None; violation_times=[]
    for row in rows:
        ref=WholeBodyReference(*[template.new_tensor(row[k]) for k in ('q','v','a')])
        result=collision_reference_check(fabric,ref)
        if not result['checked']:
            break
        checked=True
        if result['maximum_violation'] > maximum:
            maximum=result['maximum_violation']; worst_time=row['time']
        if result['minimum_cover_distance'] is not None:
            minimum=result['minimum_cover_distance'] if minimum is None else min(minimum,result['minimum_cover_distance'])
        count += int(not result['passed'])
        if not result['passed']:violation_times.append(float(row['time']))
    return dict(checked=checked, passed=count==0, maximum_violation=maximum,
                minimum_cover_distance=minimum, violating_samples=count, worst_time=worst_time,
                violation_times=violation_times,
                acceleration_residual_tolerance=1e-5,
                geometry='convex_mesh' if getattr(fabric.dynamics_tasks,'convex_wrist_geometry',None) is not None else 'sphere_cover')
