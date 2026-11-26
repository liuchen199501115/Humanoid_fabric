import unittest
import tempfile
from pathlib import Path

import torch
import warp as wp

from fabrics_sim.fabrics.task_request import TaskRequest, task_objective
from fabrics_sim.fabrics.dynamics_aware import rotation_error, sphere_distance
from fabrics_sim.taskmaps.g1_centroidal_momentum import angular_momentum_matrix
from fabrics_sim.solver.AccelerationQP import G1DynamicsAccelerationQP


def request(A, desired, bias=None, active=None, metric=None):
    return TaskRequest("test", A, torch.zeros_like(desired) if bias is None else bias, desired,
                       torch.diag_embed(torch.ones_like(desired)) if metric is None else metric,
                       torch.ones_like(desired, dtype=torch.bool) if active is None else active)


class TaskAlgebraTest(unittest.TestCase):
    def test_affine_objective_matches_autograd_and_inactive_rows(self):
        torch.manual_seed(4)
        A = torch.randn(2, 3, 35, dtype=torch.float64)
        metric = torch.tensor([[2., .5, 0.], [.5, 3., 0.], [0., 0., 1.]], dtype=A.dtype).expand(2, -1, -1)
        active = torch.tensor([[True, False, True], [False, True, False]])
        task = request(A, torch.randn(2, 3, dtype=A.dtype), torch.randn(2, 3, dtype=A.dtype), active, metric)
        T = torch.eye(35, dtype=A.dtype).expand(2, -1, -1) + .1 * torch.randn(2, 35, 35, dtype=A.dtype)
        offset = torch.randn(2, 35, dtype=A.dtype)
        H, g = task_objective([task], T, offset)
        command = torch.randn(2, 35, dtype=A.dtype, requires_grad=True)
        residual = task.residual((T @ command[..., None]).squeeze(-1) + offset)
        cost = .5 * (residual[..., None, :] @ metric @ residual[..., None]).sum()
        gradient, = torch.autograd.grad(cost, command)
        torch.testing.assert_close(gradient, (H @ command[..., None]).squeeze(-1) + g)
        disabled = request(A, task.desired, active=torch.zeros_like(active))
        self.assertEqual(disabled.natural_form()[0].count_nonzero(), 0)
        self.assertEqual(disabled.natural_form()[1].count_nonzero(), 0)

    def test_invalid_metric_and_nonfinite_data_rejected(self):
        A = torch.zeros(1, 3, 35)
        for metric in (-torch.eye(3)[None], torch.full((1, 3, 3), float("nan"))):
            with self.assertRaises(ValueError):
                request(A, torch.zeros(1, 3), metric=metric)
        with self.assertRaises(ValueError):
            request(A, torch.zeros(1, 3), metric=torch.tensor([[[1., 2., 0.], [2., 1., 0.], [0., 0., 1.]]]))
        with self.assertRaises(ValueError):
            request(A, torch.full((1, 3), float("nan")))

    def test_orientation_restores_both_signs_and_resolves_pi(self):
        for angle in (0., .3, -.3, 3.141592653589793, -3.1415):
            t = torch.tensor(angle, dtype=torch.float64)
            R = torch.tensor([[1., 0., 0.], [0., t.cos(), -t.sin()], [0., t.sin(), t.cos()]], dtype=t.dtype)[None]
            error = rotation_error(R, torch.eye(3, dtype=t.dtype)[None])
            self.assertTrue(torch.isfinite(error).all())
            self.assertAlmostEqual(error[0, 0].item(), angle, places=5)
            self.assertLessEqual((-error[0, 0] * t).item(), 0)

    def test_sphere_separation_direction_and_coincident_fallback(self):
        jac = torch.zeros(1, 2, 3, 35)
        jac[..., :3] = torch.eye(3)
        d, J = sphere_distance(torch.tensor([[[.1, 0., 0.], [0., 0., 0.]]]), jac, .2)
        torch.testing.assert_close(d, torch.tensor([[-.1, -.2]]))
        self.assertTrue(torch.isfinite(J).all())
        self.assertTrue((J[..., 0] > 0).all())

    def test_single_body_momentum_and_translation_invariance(self):
        mass = torch.tensor([2.], dtype=torch.float64)
        pos = torch.tensor([[[7., 3., 1.]]], dtype=mass.dtype)
        R = torch.tensor([[[[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]]], dtype=mass.dtype)
        inertia = torch.diag(torch.tensor([2., 3., 4.], dtype=mass.dtype))[None]
        linear = torch.zeros(1, 1, 3, 35, dtype=mass.dtype)
        angular = torch.zeros_like(linear)
        linear[..., :3] = torch.eye(3, dtype=mass.dtype)
        angular[..., 3:6] = torch.eye(3, dtype=mass.dtype)
        A = angular_momentum_matrix(mass, pos, R, inertia, linear, angular)
        torch.testing.assert_close(A[:, :, 3:6], torch.diag(torch.tensor([3., 2., 4.], dtype=mass.dtype))[None])
        self.assertEqual(A[:, :, :3].count_nonzero(), 0)
        torch.testing.assert_close(A, angular_momentum_matrix(mass, pos + 10, R, inertia, linear, angular))

    def test_importer_rotates_inertial_frame_into_link_axes(self):
        from fabrics_sim.prod.import_urdf import parse_urdf_annotated
        wp.config.kernel_cache_dir = "/tmp/g1_warp_cache"
        wp.init()
        inertia = '<inertial><origin rpy="0 0 1.5707963267948966"/><mass value="1"/><inertia ixx="1" iyy="2" izz="3" ixy="0" ixz="0" iyz="0"/></inertial>'
        urdf = ('<robot name="inertia_test"><link name="root">' + inertia + '</link><link name="child">'
                + inertia + '</link><joint name="joint" type="revolute"><parent link="root"/>'
                '<child link="child"/><axis xyz="0 0 1"/><limit lower="-1" upper="1" effort="1" velocity="1"/></joint>'
                '<link name="sensor"/><joint name="sensor_frame" type="fixed"><parent link="child"/>'
                '<child link="sensor"/></joint></robot>')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.urdf"
            path.write_text(urdf)
            builder = wp.sim.ModelBuilder()
            parse_urdf_annotated(str(path), builder, wp.transform_identity(), floating=True, include_collisions=False)
            model = builder.finalize("cpu")
            actual = torch.tensor(model.body_inertia.numpy())
            torch.testing.assert_close(actual[:2], torch.diag(torch.tensor([2., 1., 3.]))[None].expand(2, -1, -1), atol=1e-6, rtol=0)
            self.assertEqual(actual[2].count_nonzero(), 0)
            self.assertEqual(float(model.body_mass.numpy().sum()), 2.)

    def test_task_wbc_tracks_physical_acceleration_with_servo_and_correction(self):
        dtype = torch.float64
        A = torch.zeros(1, 1, 35, dtype=dtype)
        A[:, :, 6] = 1
        target = torch.tensor([[.4]], dtype=dtype)
        correction = torch.zeros(1, 35, dtype=dtype)
        correction[:, 6] = 2
        solver = G1DynamicsAccelerationQP(max_iter=4000, eps_abs=1e-8, eps_rel=1e-8)
        result = solver.solve(acc_reference=torch.full_like(correction, 100.),
            mass_matrix=torch.eye(35, dtype=dtype)[None], bias_force=torch.zeros_like(correction),
            contact_jacobian=torch.zeros(1, 12, 35, dtype=dtype), jdot_qdot=torch.zeros(1, 12, dtype=dtype),
            contact_mode=torch.zeros(1, dtype=torch.long), command_acc_correction=correction,
            servo_bias=torch.full((1, 29), .1, dtype=dtype), servo_acc_gain=torch.full((1, 29), .2, dtype=dtype),
            task_requests=[request(A, target)], effort_limits=10., acc_lower=-10., acc_upper=10.)
        self.assertTrue(result.validated.all())
        self.assertAlmostEqual(result.predicted_acc[0, 6].item(), .4, places=4)
        self.assertAlmostEqual(result.acc[0, 6].item(), (2 + .4 - .1) / 1.2, places=4)
        self.assertLess(result.task_residuals["test"].abs().max().item(), 1e-4)

    def test_nominal_task_ablation_retains_predicted_feedback_diagnostics(self):
        dtype = torch.float64
        A = torch.zeros(1, 1, 35, dtype=dtype)
        A[:, :, 6] = 1
        correction = torch.zeros(1, 35, dtype=dtype)
        correction[:, 6] = 2
        result = G1DynamicsAccelerationQP(max_iter=4000, eps_abs=1e-8, eps_rel=1e-8).solve(
            acc_reference=torch.zeros_like(correction), mass_matrix=torch.eye(35, dtype=dtype)[None],
            bias_force=torch.zeros_like(correction), contact_jacobian=torch.zeros(1, 12, 35, dtype=dtype),
            jdot_qdot=torch.zeros(1, 12, dtype=dtype), contact_mode=torch.zeros(1, dtype=torch.long),
            command_acc_correction=correction, servo_bias=torch.full((1, 29), .1, dtype=dtype),
            servo_acc_gain=torch.full((1, 29), .2, dtype=dtype), task_tracks_predicted=False,
            task_requests=[request(A, torch.tensor([[.4]], dtype=dtype))], effort_limits=10.)
        self.assertTrue(result.validated.all())
        self.assertAlmostEqual(result.nominal_acc[0, 6].item(), .4, places=4)
        self.assertAlmostEqual(result.predicted_acc[0, 6].item(), .98, places=4)
        self.assertAlmostEqual(result.task_residuals["test"][0, 0].item(), .58, places=4)


class G1PhysicalTasksTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.config.kernel_cache_dir = "/tmp/g1_warp_cache"
        wp.init()
        from fabrics_sim.fabrics.g1_pose_fabric import G1PoseFabric
        cls.fabric = G1PoseFabric(4, "cpu", robot_name="g1_29dof_rev_1_0", task_profile="dynamics-aware")
        cls.q = torch.zeros(4, 36)
        cls.q[:, 2], cls.q[:, 6] = .8, 1.
        torch.manual_seed(5)
        cls.q[:, 7:] = .1 * torch.randn(4, 29)

    def setUp(self):
        f, q = self.fabric, self.q
        f.dynamics_tasks.cache_task_requests = True
        f.dynamics_tasks.priority_mode = 'none'
        f.dynamics_tasks.priority_joint_armature = None
        f.dynamics_tasks.wrist_hip_barrier = False
        f.dynamics_tasks.set_hand_tracking("position")
        f.dynamics_tasks.body_reference = None
        f.dynamics_tasks.collision_enabled = True
        f.dynamics_tasks.momentum_enabled = True
        f.dynamics_tasks.managed_walking = False
        f.dynamics_tasks.source_pelvis_metric_scale = 0.
        f.set_hand_task_active(torch.ones(4, 2, dtype=torch.bool))
        f.set_collision_obstacles(torch.empty(4, 0, 4))
        f.set_contact_state(torch.zeros(4, 2, dtype=torch.bool))
        f.set_centroidal_momentum_target(torch.zeros(4, 3))
        f.set_center_of_mass_target(f.compute_center_of_mass(q))
        f.set_joint_posture_target(q[:, 7:])
        f.set_pelvis_target(q[:, :3], q[:, 3:7])
        for taskmap, setter in ((f.feet_pose_taskmap, f.set_feet_targets), (f.hand_pose_taskmap, f.set_hand_targets)):
            taskmap(q, None)
            transforms = taskmap.robot_kinematics.batch_link_transforms_torch[:, taskmap.link_indices].clone()
            setter(transforms[..., :3], transforms[..., 3:])

    def test_paper_priority_real_g1_preserves_physical_constraint_rows(self):
        f = self.fabric
        velocity = .05 * torch.randn(4, 35)
        f.dynamics_tasks.wrist_hip_barrier = True
        f.compute_acceleration(self.q, velocity)
        constraints = tuple(v.clone() for v in f.dynamics_tasks.last_collision_constraints)
        f.dynamics_tasks.priority_mode = 'paper'
        f.dynamics_tasks.priority_joint_armature = torch.full((4, 29), .01)
        f.dynamics_tasks.cache_task_requests = False
        acceleration = f.compute_acceleration(self.q, velocity)
        state = f.dynamics_tasks.last_snapshot
        self.assertTrue(torch.isfinite(acceleration).all())
        self.assertIsNone(f.dynamics_tasks.last_requests)
        self.assertLess(state['fabric_priority_nullspace_error'].max().item(), 1e-9)
        self.assertLess(state['fabric_priority_idempotence_error'].max().item(), 1e-9)
        self.assertGreater(torch.linalg.eigvalsh(state['fabric_acc_metric']).min().item(), 0.)
        for value, actual in zip(constraints, f.dynamics_tasks.last_collision_constraints):
            torch.testing.assert_close(actual, value)

    def test_acceleration_only_clears_cached_leaves_without_changing_acceleration(self):
        f = self.fabric
        velocity = torch.zeros(4, 35)
        expected = f.compute_acceleration(self.q, velocity)
        self.assertIsNotNone(f.dynamics_tasks.last_requests)
        com = next(r for r in f.dynamics_tasks.last_requests if r.name == 'center_of_mass')
        desired, metric, bias = com.desired.clone(), com.metric.clone(), com.bias.clone()
        expected_metric = f.dynamics_tasks.config['regularization'] * torch.eye(35)[None]
        for request in f.dynamics_tasks.last_requests:
            expected_metric = expected_metric + request.natural_form()[0]
        torch.testing.assert_close(f.dynamics_tasks.last_snapshot['fabric_acc_metric'], expected_metric)
        constraints = f.dynamics_tasks.last_collision_constraints
        f.dynamics_tasks.cache_task_requests = False
        actual = f.compute_acceleration(self.q, velocity)
        self.assertIsNone(f.dynamics_tasks.last_requests)
        self.assertIsNotNone(f.dynamics_tasks.last_snapshot)
        torch.testing.assert_close(f.dynamics_tasks.last_snapshot['com_acc_desired'], desired)
        torch.testing.assert_close(f.dynamics_tasks.last_snapshot['com_acc_metric'], metric)
        torch.testing.assert_close(f.dynamics_tasks.last_snapshot['com_jac_bias'], bias)
        torch.testing.assert_close(f.dynamics_tasks.last_snapshot['fabric_acc_metric'], expected_metric)
        torch.testing.assert_close(actual, expected, rtol=0., atol=0.)
        if constraints is not None:
            for before, after in zip(constraints, f.dynamics_tasks.last_collision_constraints):
                torch.testing.assert_close(before, after, rtol=0., atol=0.)

    def test_managed_walking_changes_soft_metrics_without_changing_contact_rows(self):
        f=self.fabric
        f.set_contact_state(torch.tensor([[True,False]]).expand(4,-1))
        baseline={t.name:t for t in f.compute_task_requests(self.q,torch.zeros(4,35))}
        f.dynamics_tasks.managed_walking=True
        walking={t.name:t for t in f.compute_task_requests(self.q,torch.zeros(4,35))}
        torch.testing.assert_close(walking['feet_pose'].metric,2*baseline['feet_pose'].metric)
        torch.testing.assert_close(walking['joint_posture'].metric,baseline['joint_posture'].metric)
        for name in baseline:
            torch.testing.assert_close(walking[name].active,baseline[name].active)
            torch.testing.assert_close(walking[name].desired,baseline[name].desired)
        self.assertFalse(walking['feet_pose'].active[:,:6].any())

    def test_disabled_posture_stays_disabled_after_reference_updates(self):
        from unittest.mock import patch

        f, q = self.fabric, self.q
        velocity = torch.zeros(4, 35)
        baseline = {task.name: task for task in f.compute_task_requests(q, velocity)}
        with patch.object(f, 'joint_posture_enabled', False):
            for offset in (.1, -.2):
                f.set_joint_posture_target(q[:, 7:] + offset,
                    torch.ones(4, 29), torch.full((4, 29), 2.))
                tasks = {task.name: task for task in f.compute_task_requests(q, velocity)}
                self.assertEqual(set(tasks), set(baseline) - {'joint_posture'})
                for name, task in tasks.items():
                    torch.testing.assert_close(task.desired, baseline[name].desired)
                    torch.testing.assert_close(task.metric, baseline[name].metric)
                # The legacy Fabric term must also stay inactive.
                term = f.joint_posture_term
                error = torch.full((4, 29), offset)
                term.metric_eval(error, velocity[:, 6:], None)
                term.force_eval(error, velocity[:, 6:],
                    f.fabrics_features['joint_posture']['joint_posture_attractor'])
                self.assertEqual(term.metric.count_nonzero(), 0)
                self.assertEqual(term.force.count_nonzero(), 0)

    def test_source_pelvis_target_sees_tilt_hidden_by_waist_compensation(self):
        f = self.fabric
        state = f.dynamics_tasks.model.snapshot(self.q)
        torso_index = f.dynamics_tasks.model.kinematics.get_link_index("torso_link")
        f.set_body_orientation_target(state["transforms"][:, torso_index, 3:7].clone(), link="torso_link")
        f.dynamics_tasks.source_pelvis_metric_scale = 1.
        # Torso target is exactly satisfied; only the pelvis target is tilted.
        target = self.q[:, 3:7].clone()
        angle = torch.tensor(.1)
        target[:, 0], target[:, 3] = torch.sin(angle/2), torch.cos(angle/2)
        f.set_pelvis_target(self.q[:, :3], target)
        tasks = {t.name:t for t in f.compute_task_requests(self.q, torch.zeros(4,35))}
        torso, pelvis = tasks['body_orientation'], tasks['source_pelvis_orientation']
        torch.testing.assert_close(torso.desired, torch.zeros(4,3), atol=1e-5, rtol=0)
        self.assertTrue((pelvis.desired[:, 0] > 0).all())
        # Pelvis feedback cannot be satisfied by moving only waist joints.
        self.assertEqual(pelvis.A[:, :, 6:].count_nonzero(), 0)
        self.assertGreater(torso.A[:, :, 6:].abs().sum(), 0)

    def test_walking_hands_yield_orientation_and_can_be_disabled(self):
        f = self.fabric
        for mode in ("position", "pose", "off"):
            f.dynamics_tasks.set_hand_tracking(mode)
            tasks = {t.name: t for t in f.compute_task_requests(self.q, torch.zeros(4,35))}
            hand = tasks["hand_pose"]
            rows = hand.active.reshape(4,2,6)
            self.assertEqual(bool(rows[:,:,:3].all()), mode != "off")
            self.assertEqual(bool(rows[:,:,3:].any()), mode == "pose")
            if mode == "off":
                H, g = hand.natural_form()
                self.assertEqual(H.count_nonzero(), 0)
                self.assertEqual(g.count_nonzero(), 0)
            # COM translation and body orientation, not a second pelvis XYZ task.
            self.assertNotIn("pelvis_pose", tasks)
            self.assertEqual(tasks["body_orientation"].A.shape[1], 3)
        with self.assertRaises(ValueError):
            f.dynamics_tasks.set_hand_tracking("invalid")

    def test_mixed_contacts_gate_both_metric_and_force(self):
        active = torch.tensor([[False, False], [True, False], [False, True], [True, True]])
        self.fabric.set_contact_state(active)
        tasks = {t.name: t for t in self.fabric.compute_task_requests(self.q, torch.zeros(4, 35))}
        feet = tasks["feet_pose"]
        torch.testing.assert_close(feet.active, (~active).repeat_interleave(6, 1))
        H, force = feet.natural_form()
        self.assertEqual(H[3].count_nonzero(), 0)
        self.assertEqual(force[3].count_nonzero(), 0)
        self.assertGreater(H[0].abs().sum(), 0)
        self.assertEqual(len(tasks), 7)  # Contact is a WBC constraint, not a soft task.

    def test_momentum_matches_link_sum_and_model_mass_matrix(self):
        model = self.fabric.dynamics_tasks.model
        s = model.snapshot(self.q)
        v = torch.randn(4, 35) * .2
        lv = (s["link_com_jac"] @ v[:, None, :, None]).squeeze(-1)
        av = (s["angular_jac"] @ v[:, None, :, None]).squeeze(-1)
        world_inertias = s["rotations"] @ model.inertias[None] @ s["rotations"].transpose(-1, -2)
        masses = model.com_task.link_masses
        direct = ((world_inertias @ av[..., None]).squeeze(-1) + torch.linalg.cross(
            s["link_com"] - s["com"][:, None], masses[None, :, None] * lv)).sum(1)
        torch.testing.assert_close((s["momentum_matrix"] @ v[..., None]).squeeze(-1), direct, atol=2e-6, rtol=2e-5)
        # Float32 mass-weighted cancellation: a few micro kg*m is roundoff.
        torch.testing.assert_close(s["momentum_matrix"][:, :, :3], torch.zeros(4, 3, 3), atol=5e-6, rtol=0)
        translated = self.q.clone()
        translated[:, :3] += torch.tensor([2., -3., 1.])
        torch.testing.assert_close(model.snapshot(translated)["momentum_matrix"], s["momentum_matrix"], atol=1e-5, rtol=1e-4)
        M = (masses[None, :, None, None] * (s["link_com_jac"].transpose(-1, -2) @ s["link_com_jac"])
             + s["angular_jac"].transpose(-1, -2) @ world_inertias @ s["angular_jac"]).sum(1)
        self.assertTrue((torch.linalg.eigvalsh(M) > 0).all())
        torch.testing.assert_close(M[:, :3, :3], masses.sum() * torch.eye(3)[None].expand(4, -1, -1))

    def test_momentum_bias_matches_time_derivative_with_acceleration(self):
        model = self.fabric.dynamics_tasks.model
        v, a = torch.randn(4, 35) * .1, torch.randn(4, 35) * .2
        dt = .002
        s, _, _ = model.evaluate(self.q, v, self.fabric.integrate_root_position)
        qplus = self.fabric.integrate_root_position(self.q, v + .5 * dt * a, dt)
        qminus = self.fabric.integrate_root_position(self.q, v - .5 * dt * a, -dt)
        lp = (model.snapshot(qplus)["momentum_matrix"] @ (v + dt * a)[..., None]).squeeze(-1)
        lm = (model.snapshot(qminus)["momentum_matrix"] @ (v - dt * a)[..., None]).squeeze(-1)
        expected = (s["momentum_matrix"] @ a[..., None]).squeeze(-1) + s["momentum_matrix_bias"]
        torch.testing.assert_close((lp - lm) / (2 * dt), expected, atol=1e-4, rtol=1e-3)

    def test_com_acceleration_only_reference_is_not_discarded(self):
        self.fabric.set_center_of_mass_target(self.fabric.compute_center_of_mass(self.q), acceleration_target=torch.ones(4, 3))
        tasks = {t.name: t for t in self.fabric.compute_task_requests(self.q, torch.zeros(4, 35))}
        torch.testing.assert_close(tasks["center_of_mass"].desired, torch.ones(4, 3), atol=1e-5, rtol=1e-5)

    def test_current_contact_bias_matches_link_acceleration_difference(self):
        model = self.fabric.dynamics_tasks.model
        v, acceleration = torch.randn(4, 35) * .1, torch.randn(4, 35) * .2
        dt = .002
        state, _, _ = model.evaluate(self.q, v, self.fabric.integrate_root_position)
        plus = model.snapshot(self.fabric.integrate_root_position(self.q, v + .5 * dt * acceleration, dt))
        minus = model.snapshot(self.fabric.integrate_root_position(self.q, v - .5 * dt * acceleration, -dt))
        feet = [model.kinematics.get_link_index(name) for name in ("left_ankle_roll_link", "right_ankle_roll_link")]
        for key in ("origin_jac", "angular_jac"):
            next_twist = (plus[key][:, feet] @ (v + dt * acceleration)[:, None, :, None]).squeeze(-1)
            previous_twist = (minus[key][:, feet] @ (v - dt * acceleration)[:, None, :, None]).squeeze(-1)
            requested = (state[key][:, feet] @ acceleration[:, None, :, None]).squeeze(-1) + state[key + "_bias"][:, feet]
            torch.testing.assert_close((next_twist - previous_twist) / (2 * dt), requested, atol=1e-4, rtol=1e-3)

    def test_static_targets_zero_tracking_requests_and_moving_reference(self):
        tasks = self.fabric.compute_task_requests(self.q, torch.zeros(4, 35))
        for task in tasks:
            if task.name != "collision":
                torch.testing.assert_close(task.desired, torch.zeros_like(task.desired), atol=1e-5, rtol=0)
        self.fabric.set_centroidal_momentum_target(torch.ones(4, 3), torch.ones(4, 3))
        momentum = {t.name: t for t in self.fabric.compute_task_requests(self.q, torch.zeros(4, 35))}["centroidal_momentum"]
        torch.testing.assert_close(momentum.desired, torch.tensor([[5., 5., 4.]]).expand(4, -1))

    def test_obstacles_and_optional_torso_target(self):
        f = self.fabric
        obstacle = torch.tensor([[[10., 10., 10., .1]]]).expand(4, -1, -1)
        f.set_collision_obstacles(obstacle)
        torso_index = f.dynamics_tasks.model.kinematics.get_link_index("torso_link")
        snapshot = f.dynamics_tasks.model.snapshot(self.q)
        f.set_body_orientation_target(snapshot["transforms"][:, torso_index, 3:], link="torso_link")
        tasks = {t.name: t for t in f.compute_task_requests(self.q, torch.zeros(4, 35))}
        torch.testing.assert_close(tasks["body_orientation"].desired, torch.zeros(4, 3), atol=1e-5, rtol=0)
        self.assertEqual(tasks["collision"].active[:, 8:].count_nonzero(), 0)
        self.assertEqual(tasks["collision"].A.shape[1], 16)
        obstacle = snapshot["transforms"][:, torso_index, :3].clone()[:, None]
        f.set_collision_obstacles(torch.cat((obstacle, torch.full((4, 1, 1), .5)), -1))
        tasks = {t.name: t for t in f.compute_task_requests(self.q, torch.zeros(4, 35))}
        self.assertGreater(tasks["collision"].desired[:, 8:].max(), 0)
        self.assertTrue(torch.isfinite(f.compute_acceleration(self.q, torch.zeros(4, 35))).all())

    def test_mass_model_mismatch_and_activation(self):
        model = self.fabric.dynamics_tasks.model
        mass = model.mass_matrix(model.snapshot(self.q))
        torch.testing.assert_close(model.validate_mass_matrix(self.q, mass), torch.zeros(4))
        with self.assertRaisesRegex(ValueError, "inertia mismatch"):
            model.validate_mass_matrix(self.q, 2 * mass)
        # Reflected motor inertia belongs on joint diagonals only; it must
        # neither mask a frame mismatch nor be counted twice.
        armature = torch.linspace(.003, .025, 29).expand(4, -1)
        with_motors = mass.clone()
        with_motors[:, 6:, 6:] += torch.diag_embed(armature)
        torch.testing.assert_close(model.validate_mass_matrix(
            self.q, with_motors, joint_armature=armature), torch.zeros(4))
        for bad_mass in (mass, 2 * with_motors):
            with self.assertRaisesRegex(ValueError, "inertia mismatch"):
                model.validate_mass_matrix(self.q, bad_mass, joint_armature=armature)
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            model.validate_mass_matrix(self.q, with_motors, joint_armature=-armature)
        self.fabric.set_hand_task_active(torch.tensor([[True, False]]).expand(4, -1))
        self.fabric.dynamics_tasks.momentum_enabled = False
        tasks = {t.name: t for t in self.fabric.compute_task_requests(self.q, torch.zeros(4, 35))}
        self.assertEqual(tasks["hand_pose"].active[:, 6:].count_nonzero(), 0)
        self.assertEqual(tasks["centroidal_momentum"].natural_form()[0].count_nonzero(), 0)

    def test_motion_reference_drives_momentum_and_posture_feedforward(self):
        import sys
        examples = str(Path(__file__).resolve().parents[1] / "examples")
        if examples not in sys.path:
            sys.path.insert(0, examples)
        from test_g1 import configure_motion_targets
        v = torch.full((4, 35), .05)
        dt = .02
        previous = self.fabric.integrate_root_position(self.q, v, -dt)
        following = self.fabric.integrate_root_position(self.q, v, dt)
        configure_motion_targets(self.fabric, self.q, previous, following, dt)
        state = self.fabric.dynamics_tasks.model.snapshot(self.q)
        expected = (state["momentum_matrix"] @ v[..., None]).squeeze(-1)
        torch.testing.assert_close(self.fabric.dynamics_tasks.momentum_reference, expected, atol=1e-5, rtol=1e-4)
        self.assertGreater(expected.abs().max(), .01)
        features = self.fabric.fabrics_features["joint_posture"]["joint_posture_attractor"]
        torch.testing.assert_close(features["velocity"], v[:, 6:], atol=1e-6, rtol=1e-4)
        self.assertTrue(torch.isfinite(features["acceleration"]).all())

    def test_analytic_reference_translation_has_consistent_link_acceleration(self):
        from test_g1 import configure_motion_targets
        v=torch.zeros(4,35);a=torch.zeros_like(v)
        v[:,:3]=torch.tensor([.3,.2,.1]);a[:,:3]=torch.tensor([.1,.2,.3])
        state=configure_motion_targets(self.fabric,self.q,reference_kinematics=(v,a))
        self.assertIsNotNone(state)
        for task in ('feet_pose','hand_pose'):
            features=self.fabric.fabrics_features[task][task+'_attractor']
            torch.testing.assert_close(features['velocity'].reshape(4,2,6)[:,:,:3],
                v[:,None,:3].expand(-1,2,-1),atol=1e-5,rtol=0)
            torch.testing.assert_close(features['acceleration'].reshape(4,2,6)[:,:,:3],
                a[:,None,:3].expand(-1,2,-1),atol=1e-5,rtol=0)

    def test_managed_reference_skips_unused_momentum_without_changing_tasks(self):
        from unittest.mock import patch
        from test_g1 import configure_motion_targets
        v = torch.full((4, 35), .05)
        dt = .02
        previous = self.fabric.integrate_root_position(self.q, v, -dt)
        following = self.fabric.integrate_root_position(self.q, v, dt)
        configure_motion_targets(self.fabric, self.q, previous, following, dt)
        self.fabric.set_centroidal_momentum_target(torch.zeros(4, 3))
        expected = self.fabric.compute_task_requests(self.q, v)
        with patch.object(self.fabric.dynamics_tasks.model, "evaluate",
                          side_effect=AssertionError("Unused source momentum evaluated")):
            configure_motion_targets(self.fabric, self.q, previous, following, dt,
                                     reference_momentum=False)
        self.fabric.set_centroidal_momentum_target(torch.zeros(4, 3))
        actual = self.fabric.compute_task_requests(self.q, v)
        self.assertEqual([t.name for t in expected], [t.name for t in actual])
        for before, after in zip(expected, actual):
            for field in ("A", "bias", "desired", "metric", "active"):
                torch.testing.assert_close(getattr(before, field), getattr(after, field),
                                           atol=0., rtol=0.)

    def test_wrist_hip_cover_adds_collision_rows_and_preserves_other_requests(self):
        tasks = self.fabric.dynamics_tasks
        velocity = torch.full((4,35),.07)
        before = {t.name:t for t in self.fabric.compute_task_requests(self.q,velocity)}
        try:
            tasks.enable_wrist_hip_avoidance()
            tasks.wrist_hip_barrier = True
            after = {t.name:t for t in self.fabric.compute_task_requests(self.q,velocity)}
            self.assertGreater(after['collision'].A.shape[1],before['collision'].A.shape[1])
            self.assertTrue(any('rubber_hand' in name for name in tasks.sphere_names))
            self.assertTrue(any('hip_yaw' in name for name in tasks.sphere_names))
            C,bound = tasks.last_collision_constraints
            distance = tasks.last_snapshot['collision_distance']
            active = distance < .4
            active[:, :tasks.original_collision_pairs] = False
            rows = active.any(dim=0)
            self.assertTrue(rows.any())
            speed = (after['collision'].A@velocity[...,None]).squeeze(-1)
            expected = after['collision'].bias+16*speed+64*(distance-tasks.config['collision']['safe_distance'])
            expected = torch.where(active,expected,torch.full_like(expected,1e6))
            torch.testing.assert_close(C,-after['collision'].A[:,rows])
            torch.testing.assert_close(bound,expected[:,rows])
            tasks.stable_constraint_rows = True
            self.fabric.compute_task_requests(self.q,velocity)
            stable_C, stable_bound = tasks.last_collision_constraints
            fixed = slice(tasks.original_collision_pairs, len(tasks.pairs))
            self.assertEqual(stable_C.shape[1], len(tasks.pairs)-tasks.original_collision_pairs)
            # Every active material pair keeps the same physical inequality;
            # inactive rows are identities 0 <= 0 at stable pair indices.
            active_fixed = active[:,fixed]
            torch.testing.assert_close(stable_C[active_fixed], -after['collision'].A[:,fixed][active_fixed])
            torch.testing.assert_close(stable_bound[active_fixed], expected[:,fixed][active_fixed])
            self.assertEqual(stable_C[~active_fixed].count_nonzero(), 0)
            self.assertEqual(stable_bound[~active_fixed].count_nonzero(), 0)
            for name in before.keys()-{'collision'}:
                for field in ('A','bias','desired','metric','active'):
                    torch.testing.assert_close(getattr(before[name],field),getattr(after[name],field),rtol=0,atol=0)
        finally:
            tasks.wrist_hip_barrier = False
            tasks.stable_constraint_rows = False
            tasks._set_collision_geometry(tasks.config['collision']['spheres'],tasks.config['collision']['pairs'])

    def test_wrist_body_cover_includes_observed_nonadjacent_collision_pairs(self):
        tasks=self.fabric.dynamics_tasks
        try:
            tasks.enable_wrist_hip_avoidance(include_upper_body=True)
            pairs=[(tasks.sphere_names[a],tasks.sphere_names[b]) for a,b in tasks.pairs.tolist()]
            for side,other in [('left','right'),('right','left')]:
                self.assertTrue(any(a.startswith(side+'_rubber_hand') and b.startswith('torso_link') for a,b in pairs))
                self.assertTrue(any(a.startswith(side+'_wrist_roll') and b.startswith(other+'_shoulder_yaw') for a,b in pairs))
                self.assertFalse(any(a.startswith(side+'_wrist') and b.startswith(side+'_shoulder') for a,b in pairs))
            tasks.wrist_hip_barrier=True
            self.fabric.compute_task_requests(self.q,torch.zeros(4,35))
            C,bound=tasks.last_collision_constraints
            self.assertTrue(torch.isfinite(C).all() and torch.isfinite(bound).all())
        finally:
            tasks.wrist_hip_barrier=False
            tasks._set_collision_geometry(tasks.config['collision']['spheres'],tasks.config['collision']['pairs'])


if __name__ == "__main__":
    unittest.main()
