import unittest

import torch

from fabrics_sim.solver.joint_control import explicit_pd_gains, integration_acceleration_bounds, integrated_servo_model, servo_health
from fabrics_sim.solver.joint_control import synchronize_command_acceleration
from fabrics_sim.solver.joint_control import bounded_reference_step
from fabrics_sim.solver.joint_control import measured_joint_acceleration_bounds
from fabrics_sim.solver.joint_control import joint_limit_avoidance_request
from fabrics_sim.solver.joint_control import contact_servo_gain_bank
from fabrics_sim.solver.joint_control import implicit_servo_response
from fabrics_sim.taskmaps.joint_posture import JointPostureTaskMap
from fabrics_sim.fabric_terms.joint_posture import JointPostureAttractor


class ExplicitGainTest(unittest.TestCase):
    def test_implicit_feedback_dissipates_free_joint_energy_with_stance_gains(self):
        mass=torch.eye(35,dtype=torch.float64)[None];mass[:,6:,6:]*=.0003
        kp=torch.full((1,29),40.,dtype=mass.dtype);kd=torch.full_like(kp,2.)
        dt=.002;response=implicit_servo_response(mass,kp,kd,dt)
        q=torch.full_like(kp,.1);v=torch.zeros_like(q)
        energy=lambda: (.5*.0003*v.square()+.5*kp*q.square()).sum().item()
        previous=energy()
        for _ in range(600):
            tau=(response@(-kp*q-(kd+dt*kp)*v)[...,None]).squeeze(-1)
            v_next=v+dt*tau/.0003;q_next=q+dt*v_next
            torch.testing.assert_close(tau,-kp*q_next-kd*v_next,atol=1e-12,rtol=1e-10)
            q,v=q_next,v_next
            current=energy();self.assertLessEqual(current,previous+1e-12);previous=current
        self.assertLess(q.abs().max().item(),1e-8)

    def test_contact_gain_bank_respects_free_axes_and_coupled_step_budget(self):
        mass=torch.eye(35,dtype=torch.float64)[None]
        mass[:,6:,6:]*=.001
        J=torch.zeros(1,12,35,dtype=mass.dtype)
        for i in range(6):
            J[0,i,i]=J[0,i,6+i]=1.
            J[0,6+i,i]=J[0,6+i,12+i]=1.
        nominal_kp=torch.full((29,),40.,dtype=mass.dtype)
        nominal_kd=torch.ones(29,dtype=mass.dtype)
        kp,kd=contact_servo_gain_bank(mass,J,None,nominal_kp,nominal_kd,.002)
        free=explicit_pd_gains(mass,nominal_kp,nominal_kd,.002)
        torch.testing.assert_close(kp[0],free[0][0]);torch.testing.assert_close(kd[0],free[1][0])
        self.assertGreater(kd[1,0].item(),5*kd[0,0].item())
        torch.testing.assert_close(kd[1,6:],kd[0,6:])
        for mode in (1,2,3):
            rows=list(range(6)) if mode==1 else list(range(6,12)) if mode==2 else list(range(12))
            jac=J[:,rows];inverse=torch.linalg.inv(mass)
            mobility=inverse-inverse@jac.transpose(-1,-2)@torch.linalg.solve(jac@inverse@jac.transpose(-1,-2),jac@inverse)
            root=(kd[mode]*.002+kp[mode]*.002**2).sqrt()
            budget=root[None,:,None]*mobility[:,6:,6:]*root[None,None,:]
            self.assertLessEqual(torch.linalg.eigvalsh(budget).max().item(),.50000001)
        mask=torch.ones(1,12,dtype=torch.bool);mask[:,3:6]=False
        _,point_kd=contact_servo_gain_bank(mass,J,mask,nominal_kp,nominal_kd,.002)
        torch.testing.assert_close(point_kd[1,3:6],kd[0,3:6])

    def test_soft_limit_braking_anticipates_stops_and_yields_to_effort_limits(self):
        from fabrics_sim.solver.AccelerationQP import G1DynamicsAccelerationQP
        q=torch.full((1,29),.203,dtype=torch.float64)
        v=torch.full_like(q,-1.703)
        lower,upper=torch.full_like(q,-.0873),torch.full_like(q,2.8798)
        request=joint_limit_avoidance_request(q,v,lower,upper,20.)
        self.assertTrue((request.desired > 10.).all())
        idle=joint_limit_avoidance_request(torch.ones_like(q),torch.zeros_like(q),lower,upper,20.)
        self.assertFalse(idle.active.any())
        reverse=joint_limit_avoidance_request(-q,-v,-upper,-lower,20.)
        torch.testing.assert_close(reverse.desired,-request.desired)
        result=G1DynamicsAccelerationQP(max_iter=10000,rho=1.,eps_abs=1e-8,eps_rel=1e-8).solve(
            acc_reference=q.new_zeros(1,35),mass_matrix=torch.eye(35,dtype=q.dtype)[None],
            bias_force=q.new_zeros(1,35),contact_jacobian=q.new_zeros(1,12,35),
            jdot_qdot=q.new_zeros(1,12),contact_mode=torch.zeros(1,dtype=torch.long),
            task_requests=[request],effort_limits=.5,acc_lower=-2.,acc_upper=2.)
        self.assertTrue(result.validated.all())
        self.assertLessEqual(result.total_joint_torque.abs().max().item(),.50001)

    def test_soft_limit_braking_overrides_persistent_extension_before_stop(self):
        q=torch.full((1,29),.95,dtype=torch.float64)
        v=torch.full_like(q,-1.47)
        lower,upper=torch.full_like(q,-.0873),torch.full_like(q,2.8798)
        for _ in range(1500):
            task=joint_limit_avoidance_request(q,v,lower,upper,20.)
            metric=task.metric.diagonal(dim1=-2,dim2=-1)
            acceleration=(metric*task.desired-.5*40)/(metric+.5)
            lo,hi=measured_joint_acceleration_bounds(q,v,lower,upper,.002)
            acceleration=torch.maximum(lo,torch.minimum(hi,acceleration))
            v+=.002*acceleration;q+=.002*v
            self.assertTrue((q>lower+.01).all())
        self.assertLess(v.abs().max().item(),.001)

    def test_distance_barrier_stops_a_persistent_request_before_joint_limit(self):
        q=torch.tensor([[.95]],dtype=torch.float64)
        v=torch.tensor([[-1.47]],dtype=torch.float64)
        lower,upper=torch.full_like(q,-.0873),torch.full_like(q,2.8798)
        strongest=0.
        for _ in range(1500):
            lo,hi=measured_joint_acceleration_bounds(q,v,lower,upper,.002,barrier_bandwidth=8.)
            acceleration=torch.maximum(lo,torch.minimum(hi,torch.full_like(q,-40.)))
            strongest=max(strongest,acceleration.item())
            v=v+.002*acceleration
            q=q+.002*v
            self.assertGreater(q.item(),lower.item()+.01)
        self.assertGreater(strongest,5.)
        self.assertLess(abs(v.item()),.001)

    def test_distance_barrier_brakes_recorded_knee_before_last_interval(self):
        q=torch.tensor([[.203]],dtype=torch.float64);v=torch.tensor([[-1.703]],dtype=torch.float64)
        lower,upper=torch.full_like(q,-.0873),torch.full_like(q,2.8798)
        old_lo,_=measured_joint_acceleration_bounds(q,v,lower,upper,.002)
        lo,_=measured_joint_acceleration_bounds(q,v,lower,upper,.002,barrier_bandwidth=8.)
        self.assertAlmostEqual(old_lo.item(),5.)
        self.assertGreater(lo.item(),10.)

    def test_impact_recovery_brakes_recorded_ankle_without_crossing_margin(self):
        q=torch.tensor([[-.2077,.2077]],dtype=torch.float64)
        v=torch.tensor([[-1.7265,1.7265]],dtype=q.dtype)
        lower,upper=torch.full_like(q,-.2618),torch.full_like(q,.2618)
        old_lo,old_hi=measured_joint_acceleration_bounds(q,v,lower,upper,.002,barrier_bandwidth=8.)
        lo,hi=measured_joint_acceleration_bounds(q,v,lower,upper,.002,
                                              barrier_bandwidth=8.,impact_recovery=True)
        self.assertGreater(lo[0,0].item(),old_lo[0,0].item()+5.)
        self.assertLess(hi[0,1].item(),old_hi[0,1].item()-5.)
        for _ in range(600):
            lo,hi=measured_joint_acceleration_bounds(q,v,lower,upper,.002,
                                                  barrier_bandwidth=8.,impact_recovery=True)
            a=torch.maximum(lo,torch.minimum(hi,torch.zeros_like(q)))
            v=v+.002*a;q=q+.002*v
            self.assertTrue((q>lower+.01).all())
            self.assertTrue((q<upper-.01).all())
        self.assertGreater(v[0,0].item(),0.)
        self.assertLess(v[0,1].item(),0.)

    def test_impact_recovery_does_not_hide_unavailable_braking(self):
        q=torch.tensor([[-.250]],dtype=torch.float64)
        v=torch.tensor([[-2.]],dtype=q.dtype)
        lo,_=measured_joint_acceleration_bounds(q,v,torch.full_like(q,-.2618),
             torch.full_like(q,.2618),.002,impact_recovery=True)
        self.assertGreater(lo.item(),150.)

    def test_impact_recovery_leaves_admissible_motion_unchanged(self):
        q=torch.zeros(1,2,dtype=torch.float64)
        v=torch.tensor([[-.2,.2]],dtype=q.dtype)
        lower,upper=torch.full_like(q,-1.),torch.full_like(q,1.)
        expected=measured_joint_acceleration_bounds(q,v,lower,upper,.002)
        actual=measured_joint_acceleration_bounds(q,v,lower,upper,.002,impact_recovery=True)
        for a,b in zip(actual,expected):torch.testing.assert_close(a,b)

    def test_measured_limit_requires_braking_before_command_reaches_stop(self):
        q = torch.tensor([[-.240, .240]])
        v = torch.tensor([[-.5, .5]])
        lo, hi = measured_joint_acceleration_bounds(q, v, torch.full_like(q, -.262),
                                                  torch.full_like(q, .262), .02)
        self.assertGreater(lo[0, 0].item(), 0.)
        self.assertLess(hi[0, 1].item(), 0.)
        # Already outside the conservative envelope: recover at 5 rad/s²,
        # rather than demanding a one-tick jump back into that envelope.
        self.assertAlmostEqual(lo[0,0].item(), 5., places=5)
        self.assertAlmostEqual(hi[0,1].item(), -5., places=5)
        next_lo = q+.02*(v+.02*lo)
        next_hi = q+.02*(v+.02*hi)
        self.assertTrue((next_lo >= -.252-1e-6).all())
        self.assertTrue((next_hi <= .252+1e-6).all())

    def test_physical_position_limit_overrides_gradual_recovery(self):
        q = torch.tensor([[.251]])
        v = torch.tensor([[2.]])
        lo, hi = measured_joint_acceleration_bounds(q,v,torch.full_like(q,-.262),torch.full_like(q,.262),.02)
        self.assertLess(hi.item(),-5.)
        self.assertLessEqual((q+.02*(v+.02*hi)).item(), .252001)
        self.assertLessEqual(lo.item(),hi.item())

    def test_balance_reference_is_bounded_and_integrally_consistent(self):
        p, v = torch.zeros(1, 3), torch.zeros(1, 3)
        for step in range(300):
            target = torch.full_like(p, .1 if step < 100 else -.1)
            new_p, new_v, acceleration = bounded_reference_step(p, v, target, .02, .15)
            self.assertLessEqual(acceleration.abs().max().item(), .500001)
            self.assertLessEqual(new_v.abs().max().item(), .200001)
            torch.testing.assert_close(new_p, p + .02 * new_v)
            torch.testing.assert_close(new_v, v + .02 * acceleration)
            p, v = new_p, new_v
        torch.testing.assert_close(p, torch.full_like(p, -.1), atol=1e-4, rtol=0.)
        with self.assertRaises(ValueError):
            bounded_reference_step(p, v, p, 0., .15)
        with self.assertRaises(ValueError):
            bounded_reference_step(p, torch.ones_like(v), p, .02, .15)

    def test_velocity_damper_brakes_before_the_last_position_interval(self):
        q=torch.tensor([[.17]],dtype=torch.float64)
        v=torch.tensor([[2.46]],dtype=torch.float64)
        lower,upper=torch.full_like(q,-.52),torch.full_like(q,.52)
        old_lo,old_hi=measured_joint_acceleration_bounds(q,v,lower,upper,.02)
        lo,hi=measured_joint_acceleration_bounds(q,v,lower,upper,.02,velocity_damper_time=.2)
        self.assertGreaterEqual(lo.item(),old_lo.item())
        self.assertLess(hi.item(),old_hi.item())
        self.assertLess(hi.item(),-30.)
        self.assertLessEqual((v+.02*hi).item(),(upper-q-.01).item()/.2+1e-8)
        with self.assertRaises(ValueError):
            measured_joint_acceleration_bounds(q,v,lower,upper,.02,velocity_damper_time=0)

    def test_continuous_sync_restores_command_without_reset(self):
        p, v = torch.ones((1,29)), torch.zeros((1,29))
        reference = torch.zeros((1,35))
        corrected = synchronize_command_acceleration(reference, p, v, v, v)
        torch.testing.assert_close(corrected[:,6:], -p)
        torch.testing.assert_close(corrected[:,:6], reference[:,:6])
        torch.testing.assert_close(p, torch.ones_like(p))
        torch.testing.assert_close(synchronize_command_acceleration(reference, p, v, p, v), reference)
    def test_servo_health_rejects_hidden_limit_cycle(self):
        v = torch.tensor([37., -37.] * 10).reshape(-1, 1, 1)
        torque = torch.zeros_like(v)
        valid, oscillating, _ = servo_health(v, torque, torque)
        self.assertFalse(valid)
        self.assertTrue(oscillating)
        # One landing transient must not be confused with a limit cycle.
        v[2:] = 0
        valid, oscillating, _ = servo_health(v, torque, torque)
        self.assertTrue(valid)
        self.assertFalse(oscillating)
        valid, _, fraction = servo_health(v, torque, torque + 1)
        self.assertFalse(valid)
        self.assertEqual(fraction.item(), 1.)
    def test_servo_model_matches_integrated_target_without_canceling_error(self):
        p = torch.tensor([[.1, .9]])
        v = torch.tensor([[.2, 2.]])
        actual_p, actual_v = torch.zeros_like(p), torch.ones_like(v)
        kp, kd = torch.tensor([10., 20.]), torch.tensor([1., 2.])
        a = torch.tensor([[3., -4.]])
        b, g = integrated_servo_model(p, v, actual_p, actual_v, kp, kd, .02)
        new_v = v + .02 * a
        new_p = p + .02 * new_v
        torch.testing.assert_close(b + g * a, kp * (new_p - actual_p) + kd * (new_v - actual_v))
        self.assertGreater(b[0, 1].item(), 0.)

    def test_integrator_brakes_before_joint_limit(self):
        q = torch.tensor([[0.0]], dtype=torch.float64)
        v = torch.zeros_like(q)
        for _ in range(200):
            lo, hi = integration_acceleration_bounds(q, v, -torch.ones_like(q), torch.ones_like(q),
                                                     torch.ones_like(q), 10 * torch.ones_like(q), .02)
            self.assertTrue((lo <= hi).all())
            v = v + .02 * hi
            q = q + .02 * v
            self.assertLessEqual(q.item(), 1.0 + 1e-10)
            self.assertLessEqual(abs(v.item()), 1.0 + 1e-10)
        self.assertGreater(q.item(), .99)

    def test_posture_restores_joint_without_direct_base_force(self):
        task = JointPostureTaskMap(1, "cpu")
        target = torch.full((1, 29), .1)
        task.set_target(target)
        q = torch.zeros((1, 36)); q[:, 6] = 1
        x, jac = task(q, None)
        torch.testing.assert_close(jac[:, :, :6], torch.zeros((1, 29, 6)))
        term = JointPostureAttractor(True, {"metric": .5, "stiffness": 20., "damping": 8., "max_acceleration": 30.}, "cpu", False)
        metric, force = term(x, torch.zeros_like(x), None)
        acc = torch.linalg.solve(metric, -force.unsqueeze(-1)).squeeze(-1)
        torch.testing.assert_close(acc, torch.full_like(x, 2.))
        _, damped_force = term(x, torch.ones_like(x), None)
        self.assertTrue((damped_force > 0).all())
        task.set_target(q[:, 7:])
        x, _ = task(q, None)
        _, force = term(x, torch.zeros_like(x), None)
        torch.testing.assert_close(force, torch.zeros_like(x))

    def test_low_inertia_servo_discrete_stability(self):
        # Previously unstable wrist-like inertia, plus higher-inertia joints.
        inertia = torch.logspace(-4, 0, 29, dtype=torch.float64)
        mass = torch.diag(torch.cat((torch.ones(6), inertia))).unsqueeze(0)
        kp, kd = explicit_pd_gains(mass, torch.full((29,), 40.), torch.ones(29), .002)
        dt = .002
        transition = torch.zeros((29, 2, 2), dtype=torch.float64)
        transition[:, 0, 0] = 1 - dt**2 * kp[0] / inertia
        transition[:, 0, 1] = dt * (1 - dt * kd[0] / inertia)
        transition[:, 1, 0] = -dt * kp[0] / inertia
        transition[:, 1, 1] = 1 - dt * kd[0] / inertia
        self.assertTrue((torch.linalg.eigvals(transition).abs() < 1).all())
        self.assertTrue((kp > 0).all())
        self.assertTrue((kd > 0).all())

    def test_coupled_step_budget(self):
        torch.manual_seed(0)
        a = torch.randn((35, 35), dtype=torch.float64)
        mass = (a @ a.T + .001 * torch.eye(35)).unsqueeze(0)
        kp, kd = explicit_pd_gains(mass, torch.full((29,), 100.), torch.full((29,), 5.), .002)
        root = torch.sqrt(kd * .002 + kp * .002**2)
        budget = root[:, :, None] * torch.linalg.inv(mass)[:, 6:, 6:] * root[:, None, :]
        self.assertLessEqual(torch.linalg.eigvalsh(budget).max().item(), .500001)
