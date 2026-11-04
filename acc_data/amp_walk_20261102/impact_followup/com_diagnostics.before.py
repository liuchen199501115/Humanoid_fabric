"""Read-only COM acceleration attribution; all vectors are in the world frame."""
import numpy as np
import torch

from fabrics_sim.fabric_terms.center_of_mass import CenterOfMassAttractor


def com_acceleration_snapshot(fabric, q, velocity, accelerations):
    """Map physical generalized accelerations using J*a + Jdot*v.

    Match Fabric's forward directional difference (epsilon=1e-5). Re-evaluate
    the actual state last so taskmap kinematic buffers do not retain q_eps.
    These are instantaneous predictions, not interval-average measurements.
    """
    task = fabric.center_of_mass_taskmap
    eps = 1e-5
    q_eps = fabric.integrate_root_position(q, velocity, eps)
    _, jac_eps = task._position_and_jacobian(q_eps)
    jac_eps = jac_eps.clone()
    position, jac = task._position_and_jacobian(q)
    com_velocity = torch.bmm(jac, velocity.unsqueeze(-1)).squeeze(-1)
    curvature = torch.bmm((jac_eps - jac) / eps, velocity.unsqueeze(-1)).squeeze(-1)
    params = fabric.fabric_params["center_of_mass_attractor"]
    attractor = CenterOfMassAttractor(True, dict(params), q.device, graph_capturable=False)
    features = fabric.fabrics_features["center_of_mass"]["center_of_mass_attractor"]
    result = {
        "com_velocity": com_velocity,
        "com_curvature": curvature,
        "com_acc_requested": attractor.requested_acceleration(position - task.position_target, com_velocity, features),
        "com_task_metric": torch.as_tensor(params["metric"], device=q.device) * params["toggle"],
    }
    for name, acceleration in accelerations.items():
        result["com_acc_" + name] = torch.bmm(jac, acceleration.unsqueeze(-1)).squeeze(-1) + curvature
    return {key: value.detach().cpu().numpy().copy() for key, value in result.items()}


def complete_observed_interval(history, time, velocity):
    """Attach (v[t+1]-v[t])/dt to prediction at t, not prediction at t+1.

    This is observed average COM acceleration from measured q/v and the URDF
    COM model, not a direct PhysX COM accelerometer. Last interval stays invalid
    unless a later state sample is available; contact changes may cause impulses.
    """
    if not history or "com_velocity" not in history[-1]:
        return
    previous = history[-1]
    dt = float(time - previous["time"])
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("COM diagnostic times must increase.")
    previous["com_acc_observed"] = (velocity - previous["com_velocity"]) / dt
    previous["com_observed_dt"] = dt
    previous["com_observed_valid"] = True
