import unittest
import numpy as np
from analyze_contact_transfer import summarize_transfer


class TransferAuditTests(unittest.TestCase):
    def data(self):
        t = np.arange(2500)*.002
        feet = np.zeros((len(t), 1, 2, 7)); feet[..., 6] = 1.
        feet[..., 2] = .035; feet[:, :, 1, 2] = .075
        forces = np.zeros((len(t), 2, 3)); forces[:, 0, 2] = 300.
        mode = np.where(t<1., 3, 1)
        return dict(history_time=t, history_commanded_mode=mode, history_frame=t*0+114,
                    history_feet=feet, history_foot_forces=forces,
                    history_com=np.zeros((len(t), 1, 3)), history_com_target=np.zeros((len(t), 1, 3)),
                    motion_tracking_valid=True)

    def test_stable_lift_is_never_full_tracking(self):
        report = summarize_transfer(self.data())
        self.assertTrue(report['transfer_passed'])
        self.assertFalse(report['full_pkl_tracking'])

    def test_no_release_retains_load_and_constraint_diagnostics(self):
        d = self.data()
        d['history_commanded_mode'][:] = 3
        d['history_foot_forces'][:, 0, 2] = 180.
        d['history_foot_forces'][:, 1, 2] = 120.
        d['history_reference_phase'] = np.full(len(d['history_time']), 'PREPARE_STEP')
        d['history_joint_velocity_violation'] = np.zeros((len(d['history_time']), 29))
        d['history_unload_qp_validated'] = np.ones(len(d['history_time']), dtype=bool)
        d['history_unload_qp_validated'][-1] = False
        report = summarize_transfer(d)
        self.assertFalse(report['transfer_passed'])
        self.assertEqual(report['final_reference_phase'], 'PREPARE_STEP')
        np.testing.assert_allclose(report['load_diagnostics']['final_load_n'], [180, 120])
        np.testing.assert_allclose(report['load_diagnostics']['final_second_min_load_fraction'], [.6, .4])
        self.assertFalse(report['constraint_diagnostics']['all_qp_validated'])
        self.assertEqual(report['constraint_diagnostics']['peak_joint_velocity_violation'], 0.)

    def test_zero_load_diagnostic_has_finite_fractions(self):
        d = self.data()
        d['history_commanded_mode'][:] = 3
        d['history_foot_forces'][:] = 0.
        report = summarize_transfer(d)
        self.assertEqual(report['load_diagnostics']['final_second_max_load_fraction'], [0., 0.])
        self.assertFalse(report['transfer_passed'])

    def test_contact_force_clearance_and_controller_are_required(self):
        for key in ('force', 'clearance', 'controller', 'mode'):
            d = self.data()
            if key == 'force': d['history_foot_forces'][-1, 1, 2] = 20.
            if key == 'clearance': d['history_feet'][-1, 0, 1, 2] = .035
            if key == 'controller': d['motion_tracking_valid'] = False
            if key == 'mode': d['history_commanded_mode'][:] = 3
            with self.subTest(key=key):
                self.assertFalse(summarize_transfer(d)['transfer_passed'])


class UnloadingAuditTests(unittest.TestCase):
    def data(self):
        n = 1750
        t = np.arange(n)*.002
        forces = np.zeros((n, 2, 3)); forces[:, 0, 2] = 320.
        return dict(history_time=t, history_unload_elapsed=t,
                    history_unload_support=np.zeros(n), history_commanded_mode=np.full(n, 3),
                    history_foot_forces=forces, history_com=np.zeros((n, 1, 3)),
                    history_com_target=np.zeros((n, 1, 3)), history_unload_com_velocity=np.zeros((n, 1, 3)),
                    history_contact_wrench_world_ankle=np.pad(forces, ((0, 0), (0, 0), (0, 3))),
                    history_unload_effort_ratio=np.full(n, .5),
                    history_unload_qp_validated=np.ones(n, dtype=bool), motion_tracking_valid=True)

    def test_stable_unloading_is_not_a_step(self):
        from analyze_contact_transfer import summarize_unloading
        result = summarize_unloading(self.data())
        self.assertTrue(result['unloading_passed'])
        self.assertFalse(result['full_pkl_tracking'])
        self.assertEqual(result['measured_steps'], 0)
        self.assertEqual(result['final_second']['max_prediction_error_n'], 0)

    def test_rejects_force_spike_instability_and_incomplete_runs(self):
        from analyze_contact_transfer import summarize_unloading
        for case in ('load', 'support', 'speed', 'mode', 'short', 'nan', 'qp', 'missing'):
            data = self.data()
            if case == 'load': data['history_foot_forces'][-1, 1, 2] = 11.
            if case == 'support': data['history_foot_forces'][-1, 0, 2] = 50.
            if case == 'speed': data['history_unload_com_velocity'][-1, 0, 0] = .1
            if case == 'mode': data['history_commanded_mode'][-1] = 1
            if case == 'short': data['history_unload_elapsed'] -= 2.
            if case == 'nan': data['history_foot_forces'][-1, 0, 2] = np.nan
            if case == 'qp': data['history_unload_qp_validated'][0] = False
            if case == 'missing': del data['history_contact_wrench_world_ankle']
            with self.subTest(case=case):
                self.assertFalse(summarize_unloading(data)['unloading_passed'])


class NativeSoleClearanceTests(unittest.TestCase):
    def test_sphere_radius_uses_world_vertical_and_asymmetric_feet(self):
        from analyze_contact_transfer import sole_clearances
        feet = np.zeros((3,2,7));feet[...,2] = .1
        # 90 degrees about Y: a local X offset contributes negative world Z.
        feet[...,4] = np.sqrt(.5);feet[...,6] = np.sqrt(.5)
        points = np.array([[[.02,0,-.03],[0,0,0]],[[.04,0,-.03],[0,0,0]]])
        radii = np.array([[.01,0],[.005,0]])
        np.testing.assert_allclose(sole_clearances(feet,points,radii),
            np.tile([.07,.055],(3,1)),atol=1e-12)

    def test_native_geometry_can_reject_false_proxy_clearance(self):
        from analyze_contact_transfer import sole_clearances
        feet = np.zeros((2,7));feet[:,2] = .06;feet[:,6] = 1.
        self.assertTrue((sole_clearances(feet) >= .02).all())
        points = np.array([[[0,0,-.045]],[[0,0,-.045]]]);radii = np.full((2,1),.005)
        np.testing.assert_allclose(sole_clearances(feet,points,radii),[.01,.01])
