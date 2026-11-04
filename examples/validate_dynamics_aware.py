"""CPU numerical validation with the real G1 URDF; this is not a physics rollout.

Run: PYTHONPATH=src python examples/validate_dynamics_aware.py --output /tmp/g1_tasks.json
"""
import argparse
import json
from pathlib import Path
import time

import torch
import warp as wp

from fabrics_sim.fabrics.g1_pose_fabric import G1PoseFabric
from fabrics_sim.fabrics.task_request import TaskRequest
from fabrics_sim.solver.AccelerationQP import G1DynamicsAccelerationQP, build_single_support_wrench_constraints


def validate():
    wp.config.kernel_cache_dir = "/tmp/g1_warp_cache"
    wp.init()
    torch.set_num_threads(1)
    fabric = G1PoseFabric(4, "cpu", robot_name="g1_29dof_rev_1_0", task_profile="dynamics-aware")
    q = torch.zeros(4, 36)
    q[:, 2], q[:, 6] = .8, 1.
    v = torch.zeros(4, 35)
    fabric.set_joint_posture_target(q[:, 7:])
    fabric.set_center_of_mass_target(fabric.compute_center_of_mass(q))
    fabric.set_pelvis_target(q[:, :3], q[:, 3:7])
    for taskmap, setter in ((fabric.feet_pose_taskmap, fabric.set_feet_targets),
                            (fabric.hand_pose_taskmap, fabric.set_hand_targets)):
        taskmap(q, None)
        pose = taskmap.robot_kinematics.batch_link_transforms_torch[:, taskmap.link_indices].clone()
        setter(pose[..., :3], pose[..., 3:])
    modes = torch.arange(4)
    contacts = torch.stack(((modes == 1) | (modes == 3), (modes == 2) | (modes == 3)), -1)
    fabric.set_contact_state(contacts)
    start = time.perf_counter()
    requests = fabric.compute_task_requests(q, v)
    elapsed = time.perf_counter() - start
    state = fabric.dynamics_tasks.last_snapshot
    model = fabric.dynamics_tasks.model
    # At zero velocity the bias is exactly gravity. Construct all dynamics
    # from the same URDF link COM Jacobians and world-frame inertias.
    masses = model.com_task.link_masses.double()
    jv, jw = state["link_com_jac"].double(), state["angular_jac"].double()
    rotations = state["rotations"].double()
    inertias = rotations @ model.inertias.double()[None] @ rotations.transpose(-1, -2)
    M = (masses[None, :, None, None] * (jv.transpose(-1, -2) @ jv)
         + jw.transpose(-1, -2) @ inertias @ jw).sum(1)
    gravity = torch.tensor([0., 0., 9.81], dtype=torch.float64)
    h = (jv.transpose(-1, -2) @ (masses[None, :, None] * gravity)[..., None]).sum(1).squeeze(-1)
    feet = [model.kinematics.get_link_index(name) for name in ("left_ankle_roll_link", "right_ankle_roll_link")]
    J = torch.cat((state["origin_jac"][:, feet], state["angular_jac"][:, feet]), 2).flatten(1, 2).double()
    tasks = [TaskRequest(t.name, t.A.double(), t.bias.double(), t.desired.double(),
                         t.metric.double(), t.active, t.priority) for t in requests]
    G, b = build_single_support_wrench_constraints(contact_mode=modes,
        foot_rotation_world=state["rotations"][:, feet].double())
    solver = G1DynamicsAccelerationQP(max_iter=10000, rho=1., eps_abs=1e-7, eps_rel=1e-7,
                                    validation_tolerance=1e-3, wrench_weight=1e-4)
    result = solver.solve(acc_reference=torch.zeros(4, 35, dtype=torch.float64), mass_matrix=M,
        bias_force=h, contact_jacobian=J, jdot_qdot=torch.zeros(4, 12, dtype=torch.float64),
        contact_mode=modes, task_requests=tasks, friction_G=G, friction_b=b,
        acc_lower=-100., acc_upper=100., effort_limits=100.)
    # QP constraints are checked independently from the solver's own flags.
    force_residual = (M @ result.predicted_acc[..., None]).squeeze(-1) + h
    force_residual[:, 6:] -= result.total_joint_torque
    force_residual -= (J.transpose(1, 2) @ result.contact_wrench[..., None]).squeeze(-1)
    contact_residual = (J @ result.predicted_acc[..., None]).squeeze(-1) * contacts.repeat_interleave(6, 1)
    assert result.validated.all(), f"WBC validation failed: {result.validated}"
    assert force_residual.abs().max() < 1e-3
    assert contact_residual.abs().max() < 1e-3
    # In flight the requested stationary COM cannot defeat gravity.
    com_task = tasks[0]
    achieved = (com_task.A @ result.predicted_acc[..., None]).squeeze(-1) + com_task.bias
    torch.testing.assert_close(achieved[0], -gravity, atol=1e-3, rtol=0)
    return {
        "scope": "stationary real-URDF dynamics/QP snapshots; not a physics rollout",
        "robot": fabric.g1_robot_name, "device": "cpu", "modes": ["flight", "left", "right", "double"],
        "total_mass_kg": masses.sum().item(), "task_evaluation_ms_batch4": elapsed * 1000,
        "validated": result.validated.tolist(), "solver_converged": result.converged.tolist(),
        "iterations": result.iters, "dynamics_residual_max": force_residual.abs().amax(1).tolist(),
        "contact_residual_max": contact_residual.abs().amax(1).tolist(),
        "friction_violation": result.friction_violation.tolist(),
        "effort_violation": result.effort_violation.tolist(),
        "achieved_com_acceleration": achieved.tolist(),
        "minimum_proxy_distance": state["collision_distance"].amin(1).tolist(),
        "task_residual_norm": {k: torch.linalg.vector_norm(value, dim=-1).tolist() for k, value in result.task_residuals.items()},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("/tmp/g1_dynamics_validation.json"))
    args = parser.parse_args()
    report = validate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
