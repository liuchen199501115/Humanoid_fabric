"""One-step free-flight inverse-dynamics check; no tracking or hidden drives."""
import numpy as np
import torch
from pathlib import Path
import argparse
from test_g1 import initialize_isaac_dynamics, read_live_isaac_snapshot, offset_left_shoulder_reference
from test_g1_two_tasks import G1_JOINT_NAMES, PD_TIMESTEP, load_motion_trajectory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--disable-self-collisions", action="store_true")
    parser.add_argument("--left-shoulder-roll-offset", type=float, default=0.0,
                        help="Diagnostic reference adjustment in radians; original PKL is unchanged.")
    parser.add_argument("--output-path", type=Path, default=Path("/tmp/g1_free_response.npz"))
    parser.add_argument("--snapshot", type=Path,
                        help="Start from a saved tracking QP state, raised above the ground.")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--motor-armature", action="store_true")
    args = parser.parse_args()
    snapshot = np.load(args.snapshot) if args.snapshot else None
    if snapshot is None:
        q, _ = load_motion_trajectory(Path("retarget/output/B2A_131_02_stageii.pkl"))
        q = offset_left_shoulder_reference(q[:1], args.left_shoulder_roll_offset)
        v = torch.zeros((1, 35), dtype=q.dtype)
    else:
        q = torch.from_numpy(snapshot['q'].copy())
        v = torch.from_numpy(snapshot['qd'].copy())
    q[:, 2] += 2.0  # Exclude ground impulses from the dynamics comparison.
    *_, names, app, world, robot, contacts = initialize_isaac_dynamics(
        q, v, True, enable_self_collisions=False if args.disable_self_collisions else None,
        report_contacts=True)
    from omni.physx import get_physx_simulation_interface
    mass_before = np.asarray(robot._physics_view.get_generalized_mass_matrices()).copy()
    if args.motor_armature:
        from g1_armature import unitree_armatures
        robot.set_armatures(unitree_armatures(names))
        from isaacsim.core.simulation_manager import SimulationManager
        SimulationManager.get_physics_sim_view().update_articulations_kinematic()
    mass_after = np.asarray(robot._physics_view.get_generalized_mass_matrices()).copy()
    print('MASS ARMATURE DELTA', np.diagonal(mass_after - mass_before, axis1=-2, axis2=-1), flush=True)
    from pxr import PhysicsSchemaTools
    pairs = set()
    contact_records = []
    def report(headers, data):
        for header in headers:
            pair = (str(PhysicsSchemaTools.intToSdfPath(header.collider0)),
                    str(PhysicsSchemaTools.intToSdfPath(header.collider1)))
            if header.num_contact_data and pair not in pairs:
                pairs.add(pair)
                separation = min(data[i].separation for i in range(
                    header.contact_data_offset, header.contact_data_offset + header.num_contact_data))
                contact_records.append((*pair, str(separation)))
                print("CONTACT", pair, "separation", separation, flush=True)
    subscription = get_physx_simulation_interface().subscribe_contact_report_events(report)
    print("SELF COLLISIONS", robot.get_enabled_self_collisions(), flush=True)
    passive = {}
    for name in ('dof_friction_coefficients', 'dof_friction_properties',
                 'dof_armatures', 'dof_stiffnesses', 'dof_dampings'):
        passive[name] = np.asarray(getattr(robot._physics_view, 'get_' + name)()).copy()
        print(name, passive[name].tolist(), flush=True)
    order = [G1_JOINT_NAMES.index(n) for n in names]
    limits = np.asarray(robot.get_dof_limits())[0]
    initial = q[0, 7:].numpy()[order]
    print("INITIAL LIMIT VIOLATIONS", [(names[i], float(initial[i]), limits[i].tolist())
          for i in range(29) if initial[i] < limits[i, 0] or initial[i] > limits[i, 1]], flush=True)
    results = []
    try:
        for step in range(args.steps):
            q0, v0, mass, bias, _, _ = read_live_isaac_snapshot(robot, names, "cpu", q.dtype)
            tau = torch.zeros_like(v0)
            # Small excitation to test both passive and actuated dynamics.
            if step >= 10:
                tau[:, 6:] = .05 * torch.sin(torch.arange(29) + step)
            pred = torch.linalg.solve(mass, (tau - bias).unsqueeze(-1)).squeeze(-1)
            robot.set_joint_efforts(tau[:, 6:][:, order].numpy())
            world.step(render=False)
            _, v1, _, _, _, _ = read_live_isaac_snapshot(robot, names, "cpu", q.dtype)
            observed = (v1 - v0) / PD_TIMESTEP
            results.append((pred.numpy().copy(), observed.numpy().copy()))
        predicted, observed = np.asarray(results).transpose(1, 0, 2, 3)
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.output_path, predicted=predicted, observed=observed,
            contact_records=np.asarray(contact_records, dtype=str).reshape(-1, 3),
            joint_order=np.asarray(G1_JOINT_NAMES), dt=np.asarray(PD_TIMESTEP),
            self_collisions_disabled=np.asarray(args.disable_self_collisions),
            left_shoulder_roll_offset=np.asarray(args.left_shoulder_roll_offset),
            **passive,
            mass_before=mass_before, mass_after=mass_after,
        )
        print("FREE RESPONSE RMSE", np.sqrt(np.mean((predicted[1:] - observed[1:])**2, axis=(0, 1))), flush=True)
        print("FIRST PREDICTED", predicted[0], "OBSERVED", observed[0], flush=True)
    finally:
        app.close()


if __name__ == "__main__":
    main()
