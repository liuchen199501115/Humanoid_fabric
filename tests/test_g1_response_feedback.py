import sys
import unittest
from unittest.mock import patch
import contextlib
import io
from types import SimpleNamespace
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'examples'))
from g1_response_feedback import ResponseFeedback
from fabrics_sim.fabrics.task_request import TaskRequest

class ResponseFeedbackTest(unittest.TestCase):
    def state(self):
        return torch.zeros(1,35,dtype=torch.float64),torch.tensor([1]),torch.full((1,2),100.,dtype=torch.float64)

    def test_exact_model_has_no_correction(self):
        v,m,f=self.state();law=ResponseFeedback();a=torch.ones_like(v)
        law.observe(v,m,f,.002);law.commit(v,a,m,f)
        diag=law.observe(v+.002*a,m,f,.002)
        torch.testing.assert_close(diag['response_estimate'],torch.zeros_like(v))

    def test_constant_disturbance_converges_and_only_tracking_targets_change(self):
        v,m,f=self.state();law=ResponseFeedback(gain=1.,settle_time=0.)
        a=torch.zeros_like(v);disturbance=torch.zeros_like(v);disturbance[:,:3]=2.
        for _ in range(100):
            law.observe(v,m,f,.01);law.commit(v,a,m,f);v+=.01*disturbance
        torch.testing.assert_close(law.estimate,disturbance,atol=1e-8,rtol=1e-8)
        A=torch.eye(35,dtype=v.dtype)[None,:3];z=v[:,:3]*0
        def task(name):return TaskRequest(name,A.clone(),z.clone(),z.clone(),torch.eye(3,dtype=v.dtype)[None],torch.ones_like(z,dtype=torch.bool))
        tracking,contact=task('center_of_mass'),task('support_orientation_0')
        law.apply([tracking,contact])
        torch.testing.assert_close(tracking.desired,-disturbance[:,:3])
        torch.testing.assert_close(contact.desired,z);torch.testing.assert_close(tracking.A,A)

    def test_contact_switch_resets_and_impact_is_not_learned(self):
        v,m,f=self.state();law=ResponseFeedback(settle_time=0.)
        law.observe(v,m,f,.01);law.commit(v,v,m,f)
        law.observe(v+1,m,f,.01)
        self.assertGreater(float(law.estimate[:,:6].abs().sum()),0.)
        law.commit(v,v,m,f)
        law.observe(v,m+1,f,.01)
        torch.testing.assert_close(law.estimate,torch.zeros_like(v))
        law.commit(v,v,m,f)
        diag=law.observe(v+100,m,f+1000,.01)
        self.assertFalse(bool(diag['response_ready'].any()))
        torch.testing.assert_close(law.estimate,torch.zeros_like(v))

    def test_gain_zero_is_exact_and_joint_compensation_defaults_off(self):
        v,m,f=self.state();law=ResponseFeedback(gain=0.,settle_time=0.)
        law.observe(v,m,f,.01);law.commit(v,v,m,f)
        law.observe(v+100,m,f,.01)
        torch.testing.assert_close(law.estimate[:,6:],torch.zeros_like(v[:,6:]))
        request=TaskRequest('joint_posture',torch.ones(1,29,35,dtype=v.dtype),v[:,6:]*0,v[:,6:].clone(),torch.eye(29,dtype=v.dtype)[None],torch.ones(1,29,dtype=torch.bool))
        original=request.desired.clone();law.apply([request]);torch.testing.assert_close(request.desired,original)

    def test_invalid_parameters_fail(self):
        for kwargs in ({'gain':-1},{'gain':2},{'time_constant':0},{'linear_limit':float('nan')}):
            with self.assertRaises(ValueError):ResponseFeedback(**kwargs)

class StoppingDistanceTest(unittest.TestCase):
    def test_threshold_preserves_successful_envelope_and_brakes_outward_motion(self):
        from g1_response_feedback import stopping_distance_request
        q=torch.zeros(1,29,dtype=torch.float64);v=torch.zeros_like(q)
        lo=torch.full_like(q,-.52);hi=-lo
        q[:,13]=.02;v[:,13]=4.5
        request,demand=stopping_distance_request(q,v,lo,hi,(12,13,14),budget=25.)
        self.assertFalse(bool(request.active.any()))
        q[:,13]=.132;v[:,13]=6.12
        request,demand=stopping_distance_request(q,v,lo,hi,(12,13,14),budget=25.)
        self.assertTrue(bool(request.active[0,13]))
        self.assertLess(float(request.desired[0,13]),-50.)
        # Ideal constant-deceleration endpoint reaches the reserved margin.
        acceleration=request.desired[0,13]
        stop_time=-v[0,13]/acceleration
        endpoint=q[0,13]+v[0,13]*stop_time+.5*acceleration*stop_time**2
        self.assertAlmostEqual(float(endpoint),.49,places=9)

    def test_direction_scope_and_acceleration_cap(self):
        from g1_response_feedback import stopping_distance_request
        q=torch.zeros(1,29);v=torch.zeros_like(q);lo=torch.full_like(q,-.52);hi=-lo
        q[:,13]=.50;v[:,13]=-1.
        request,_=stopping_distance_request(q,v,lo,hi,(13,))
        self.assertFalse(bool(request.active.any()))
        v[:,13]=20.;v[:,0]=20.
        request,_=stopping_distance_request(q,v,lo,hi,(13,))
        self.assertEqual(float(request.desired[0,13]),-200.)
        self.assertFalse(bool(request.active[0,0]))
        self.assertGreaterEqual(float(torch.linalg.eigvalsh(request.metric).min()),0.)

class RootRecoveryTest(unittest.TestCase):
    def test_inactive_inside_margin_and_translation_invariant(self):
        from g1_response_feedback import root_recovery_request
        q=torch.zeros(1,36,dtype=torch.float64);q[:,6]=1.
        v=torch.zeros(1,35,dtype=q.dtype);target=q.clone()
        q[:,0]=.05
        task,gate=root_recovery_request(q,v,target,v,v)
        self.assertEqual(float(gate),0.);self.assertFalse(bool(task.active.any()))
        q[:,0]=.13
        task,gate=root_recovery_request(q,v,target,v,v)
        shifted=q.clone();shifted[:,:3]+=10.;shifted_target=target.clone();shifted_target[:,:3]+=10.
        same,same_gate=root_recovery_request(shifted,v,shifted_target,v,v)
        torch.testing.assert_close(task.desired,same.desired)
        torch.testing.assert_close(gate,same_gate)
        self.assertGreater(float(gate),0.);self.assertLess(float(gate),1.)

    def test_feedforward_is_preserved_and_only_root_acceleration_is_observed(self):
        from g1_response_feedback import root_recovery_request
        q=torch.zeros(1,36);target=q.clone();q[:,0]=1.
        v=torch.zeros(1,35);a=torch.full_like(v,80.)
        task,gate=root_recovery_request(q,v,target,v,a)
        torch.testing.assert_close(task.desired,torch.tensor([[50.,80.,80.]]))
        self.assertEqual(float(gate),1.)
        acceleration=torch.ones_like(v);acceleration[:,3:]=1000.
        torch.testing.assert_close(task.residual(acceleration),1.-task.desired)

class ResponseProfileTest(unittest.TestCase):
    def parse(self,*args):
        from test_g1_2 import parse_args
        with patch.object(sys,'argv',['test_g1_2.py',*args]):return parse_args()

    def test_matched_execution_and_original_acceptance(self):
        baseline=vars(self.parse('--tracking-profile','twelve-wbc'))
        response=vars(self.parse('--tracking-profile','response-wbc','--response-gain','0'))
        self.assertEqual({k for k in baseline if baseline[k]!=response[k]},
                         {'tracking_profile','output_path','response_gain'})
        self.assertTrue(response['track_full_motion'] and response['stop_at_fidelity_loss'])
        self.assertTrue(response['direct_wbc'] and response['tracking_servo'])

    def test_invalid_state_control_configuration_is_rejected(self):
        for args in [('--state-braking-budget','25'),('--state-root-metric','-1'),
                     ('--response-error-onset','nan'),('--tracking-steps','12'),
                     ('--state-braking-hard',),('--state-root-angular-scale','5')]:
            with self.subTest(args=args),contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):self.parse('--tracking-profile','response-wbc',*args)

class TrackingTubeRecoveryTest(unittest.TestCase):
    def test_only_auxiliary_rows_are_softened_and_input_constraints_unchanged(self):
        from g1_response_feedback import solve_with_soft_orientation_recovery
        # Import the experiment module to register its historical helper directory.
        import test_g1_2
        C=torch.arange(8*35,dtype=torch.float64).reshape(1,8,35)
        d=torch.arange(8,dtype=torch.float64)[None]
        original_C,original_d=C.clone(),d.clone()
        captured={}
        def solve(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(validated=torch.tensor([True]))
        solver=SimpleNamespace(solve=solve,reset=lambda:None)
        with patch('tube_recovery.solve_with_orientation_tube_recovery',
                   return_value=SimpleNamespace(validated=torch.tensor([False]))):
            result=solve_with_soft_orientation_recovery(solver,tube_C=C[:,-3:],tube_d=d[:,-3:],
                tube_diagnostics=torch.zeros(1,3,3),acceleration_constraints=(C,d),
                task_requests=[],priority_tasks=('base',),effort_limits=torch.tensor([88.]))
        retained_C,retained_d=captured['acceleration_constraints']
        torch.testing.assert_close(retained_C,C[:,[0,1,2,3,4,6]])
        torch.testing.assert_close(retained_d,d[:,[0,1,2,3,4,6]])
        torch.testing.assert_close(C,original_C);torch.testing.assert_close(d,original_d)
        torch.testing.assert_close(captured['effort_limits'],torch.tensor([88.]))
        self.assertEqual([r.name for r in captured['task_requests']],
                         ['joint_tube_recovery','orientation_tube_recovery'])
        self.assertTrue(bool(result.validated.all()) and solver.soft_orientation_recovery_used)

    def test_validated_solution_is_returned_without_relaxation(self):
        from g1_response_feedback import solve_with_soft_orientation_recovery
        import test_g1_2
        result=SimpleNamespace(validated=torch.tensor([True]))
        solver=SimpleNamespace()
        with patch('tube_recovery.solve_with_orientation_tube_recovery',return_value=result):
            actual=solve_with_soft_orientation_recovery(solver,tube_C=None,tube_d=None,tube_diagnostics=None)
        self.assertIs(actual,result)
        self.assertFalse(solver.soft_orientation_recovery_used)

class ComTrackingMetricTest(unittest.TestCase):
    def test_velocity_detects_departure_before_position_threshold(self):
        from g1_response_feedback import com_tracking_metric
        error=torch.tensor([[.02,0.,0.]],dtype=torch.float64)
        stopped=torch.zeros_like(error)
        away=torch.tensor([[.8,0.,0.]],dtype=error.dtype)
        near,_=com_tracking_metric(error,stopped)
        departing,risk=com_tracking_metric(error,away)
        self.assertEqual(float(near),1.)
        self.assertEqual(float(departing),4.)
        self.assertAlmostEqual(float(risk),.1)

    def test_recovery_velocity_reduces_activation_and_vertical_error_is_excluded(self):
        from g1_response_feedback import com_tracking_metric
        error=torch.tensor([[.08,0.,2.]],dtype=torch.float64)
        recovery=torch.tensor([[-.6,0.,10.]],dtype=error.dtype)
        scale,risk=com_tracking_metric(error,recovery)
        self.assertEqual(float(scale),1.)
        self.assertAlmostEqual(float(risk),.02)
        before=error.clone();com_tracking_metric(error,recovery)
        torch.testing.assert_close(error,before)

    def test_bounded_batch_rotation_invariance_and_disable(self):
        from g1_response_feedback import com_tracking_metric
        error=torch.tensor([[.01,0,0],[.06,0,0],[100.,0,0]],dtype=torch.float64)
        v=torch.zeros_like(error)
        scale,risk=com_tracking_metric(error,v,maximum=16.)
        torch.testing.assert_close(scale,torch.tensor([1.,8.5,16.],dtype=error.dtype))
        rotated=error[:,[1,0,2]]
        same,same_risk=com_tracking_metric(rotated,v,maximum=16.)
        torch.testing.assert_close(scale,same);torch.testing.assert_close(risk,same_risk)
        disabled,_=com_tracking_metric(error,v,maximum=1.)
        torch.testing.assert_close(disabled,torch.ones_like(disabled))

    def test_profile_uses_validated_axis_scope_and_keeps_feature_opt_in(self):
        defaults=ResponseProfileTest().parse('--tracking-profile','response-wbc')
        self.assertIsNone(defaults.state_com_maximum)
        self.assertEqual(defaults.state_com_axes,'xyz')
        enabled=ResponseProfileTest().parse('--tracking-profile','response-wbc','--state-com-maximum','4')
        self.assertEqual(enabled.state_com_maximum,4.)
        self.assertTrue(enabled.stop_at_fidelity_loss)

    def test_drift_profile_exactly_matches_the_physical_trial_configuration(self):
        parse=ResponseProfileTest().parse
        tested=vars(parse('--tracking-profile','response-wbc','--response-gain','0',
                          '--state-com-maximum','4','--state-com-axes','xyz'))
        named=vars(parse('--tracking-profile','drift-wbc'))
        self.assertEqual({key for key in tested if tested[key]!=named[key]},
                         {'tracking_profile','output_path'})
        overridden=parse('--tracking-profile','drift-wbc','--state-com-maximum','2')
        self.assertEqual(overridden.state_com_maximum,2.)

    def test_invalid_inputs_and_profile_parameters(self):
        from g1_response_feedback import com_tracking_metric
        zero=torch.zeros(1,3)
        for options in ({'maximum':.5},{'lookahead':-1},{'onset':float('nan')},{'width':0}):
            with self.subTest(options=options),self.assertRaises(ValueError):
                com_tracking_metric(zero,zero,**options)
        with self.assertRaises(ValueError):com_tracking_metric(zero,zero+float('nan'))
        for options in [('--state-com-maximum','.5'),('--state-com-maximum','4','--state-com-width','0')]:
            with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                ResponseProfileTest().parse('--tracking-profile','response-wbc',*options)

if __name__=='__main__':unittest.main()
