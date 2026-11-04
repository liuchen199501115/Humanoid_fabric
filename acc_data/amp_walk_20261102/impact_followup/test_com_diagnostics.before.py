import unittest
from pathlib import Path
import yaml
from types import SimpleNamespace
import numpy as np
import torch
from fabrics_sim.solver.com_diagnostics import com_acceleration_snapshot, complete_observed_interval
from fabrics_sim.fabric_terms.center_of_mass import CenterOfMassAttractor


class COMDiagnosticsTest(unittest.TestCase):
    def test_com_has_highest_leaf_motion_weights(self):
        path = Path(__file__).resolve().parents[1] / "src/fabrics_sim/fabric_params/g1_params.yaml"
        params = yaml.safe_load(path.read_text())["fabric_params"]
        others = []
        for name in ("joint_posture_attractor", "pelvis_pose_attractor", "feet_pose_attractor", "hand_pose_attractor"):
            for key, value in params[name].items():
                if key == "metric" or key.endswith("_metric"):
                    others.extend(value if isinstance(value, list) else [value])
        self.assertGreater(min(params["center_of_mass_attractor"]["metric"]), max(others))
        self.assertEqual(params["center_of_mass_attractor"]["stiffness"], [35., 35., 25.])

    def test_mapping_curvature_and_request_do_not_mutate_control_params(self):
        # c(q)=q^2 componentwise: J=2q, Jdot*v=2v^2.
        class Task:
            position_target = torch.zeros(1, 3, dtype=torch.float64)
            def _position_and_jacobian(self, q):
                return q*q, torch.diag_embed(2*q)
        params = dict(metric=[2., 3., 4.], stiffness=[1.]*3,
                      damping=[2.]*3, max_acceleration=[4.]*3, toggle=.5)
        fabric = SimpleNamespace(center_of_mass_taskmap=Task(),
            integrate_root_position=lambda q, v, dt: q + v*dt,
            fabric_params={"center_of_mass_attractor": params},
            fabrics_features={"center_of_mass": {"center_of_mass_attractor": None}})
        q = torch.ones(1, 3, dtype=torch.float64)
        v = torch.full_like(q, .2)
        result = com_acceleration_snapshot(fabric, q, v, {"fabric": torch.ones_like(q)})
        np.testing.assert_allclose(result["com_curvature"], .08, atol=1e-9)
        np.testing.assert_allclose(result["com_acc_fabric"], 2.08, atol=1e-9)
        np.testing.assert_allclose(result["com_acc_requested"], -1.8)
        self.assertEqual(params["toggle"], .5)
        np.testing.assert_allclose(result["com_task_metric"], [1., 1.5, 2.])

    def test_observation_is_assigned_to_previous_prediction(self):
        history = [{"time": 1., "com_velocity": np.array([[1., 2., 3.]])}]
        complete_observed_interval(history, 1.02, np.array([[1.02, 2.04, 3.06]]))
        np.testing.assert_allclose(history[0]["com_acc_observed"], [[1., 2., 3.]])
        self.assertTrue(history[0]["com_observed_valid"])
        with self.assertRaises(ValueError):
            complete_observed_interval(history, 1., np.zeros((1, 3)))

    def test_control_force_uses_same_clipped_request(self):
        params = dict(metric=[2., 3., 4.], stiffness=[1.]*3,
                      damping=[2.]*3, max_acceleration=[4.]*3)
        term = CenterOfMassAttractor(True, params, "cpu", graph_capturable=False)
        x, v = torch.ones(1, 3), torch.zeros(1, 3)
        ref = {"velocity": torch.ones(1, 3), "acceleration": torch.full((1, 3), 10.)}
        term.metric_eval(x, v, ref)
        term.force_eval(x, v, ref)
        torch.testing.assert_close(term.requested_acceleration(x, v, ref), torch.full((1, 3), 4.))
        torch.testing.assert_close(term.force, -torch.tensor([[8., 12., 16.]]))
