import unittest
import torch

from fabrics_sim.solver.toe_contact import ToeContact, has_loaded_toe_line, has_touching_toe_line
from fabrics_sim.solver.AccelerationQP import G1DynamicsAccelerationQP
from fabrics_sim.solver.dynamic_consistent_gpu import G1FabricContactProjector, ProjectionConfig


class ToeContactTests(unittest.TestCase):
    def test_source_toe_rotation_changes_only_soft_angular_request(self):
        p=torch.tensor([[0.,0.,.035]],dtype=torch.float64)
        r=torch.eye(3,dtype=p.dtype)[None];c=ToeContact(p,r,0.)
        jac=p.new_zeros(1,6,35);jac[0,:,:6]=torch.eye(6,dtype=p.dtype)
        bias=p.new_zeros(1,6);v=p.new_zeros(1,35)
        angle=p.new_tensor(.2);target=r.clone()
        target[:,0,0]=target[:,2,2]=angle.cos()
        target[:,0,2]=angle.sin();target[:,2,0]=-angle.sin()
        reference=(target,p.new_tensor([[.5,.1,2.]]),p.new_tensor([[1.,.3,5.]]))
        for point in (False,True):
            c.point_support=point
            baseline=c.evaluate(p,r,jac,bias,v,.1)
            tracked=c.evaluate(p,r,jac,bias,v,.1,orientation_reference=reference)
            for i in range(3):torch.testing.assert_close(baseline[i],tracked[i])
            pitch=1 if point else 0
            self.assertAlmostEqual(tracked[3].desired[0,pitch].item(),9.5,places=9)
            self.assertEqual(tracked[3].active.shape,(1,3 if point else 1))
            if point:
                torch.testing.assert_close(tracked[3].desired[:,[0,2]],p.new_zeros(1,2))
            # Activation starts from the existing bounded request.
            a=c.evaluate(p,r,jac,bias,v,0.)[3]
            b=c.evaluate(p,r,jac,bias,v,0.,orientation_reference=reference)[3]
            torch.testing.assert_close(a.desired,b.desired)

    def test_outgoing_point_contact_damps_newly_free_roll_and_yaw(self):
        p = torch.tensor([[0., 0., .035]], dtype=torch.float64)
        rotation = torch.eye(3, dtype=p.dtype)[None]
        contact = ToeContact(p, rotation, 0.)
        contact.update_support(p, rotation, [[.12, .03, 0.]], [1.], [0.])
        jac = p.new_zeros(1, 6, 35); jac[0, :, :6] = torch.eye(6, dtype=p.dtype)
        velocity = p.new_zeros(1, 35); velocity[0, [3, 5]] = p.new_tensor([.4, -.3])
        _, _, _, task = contact.evaluate(p, rotation, jac, p.new_zeros(1, 6), velocity, 0.)
        self.assertEqual(task.name, 'toe_orientation')
        self.assertTrue(task.active.all())
        expected = p.new_tensor([[-4.8, 0., 3.6]])
        torch.testing.assert_close(task.desired, expected)
        # It is a soft task in the free angular directions, not a support moment.
        self.assertTrue(contact.point_support)
        torch.testing.assert_close(task.A, jac[:, 3:])

    def test_unloaded_touching_toe_can_pivot_after_load_transfer(self):
        points = [[.12, -.03, -.035], [.12, .03, -.035]]
        self.assertFalse(has_loaded_toe_line(points, [0., .1], [0., 0.]))
        self.assertTrue(has_touching_toe_line(points, [0., 0.]))
        self.assertFalse(has_touching_toe_line(points, [.003, 0.]))
        self.assertFalse(has_touching_toe_line(points, [-.003, 0.]))
        self.assertFalse(has_touching_toe_line(points[:1], [0.]))
        self.assertFalse(has_touching_toe_line([[-.05, -.03, -.035], [-.05, .03, -.035]], [0., 0.]))
        self.assertFalse(has_touching_toe_line([[.12, -.03, -.01], [.12, .03, -.01]], [0., 0.]))

    def test_loaded_heel_edge_retains_line_support_at_actual_heel(self):
        p=torch.tensor([[0.,0.,.06]],dtype=torch.float64)
        r=torch.eye(3,dtype=p.dtype)[None]
        points=[[-.05,-.025,0.],[-.05,.025,0.]]
        c=ToeContact.from_lifted_sole(p,r,0.,.03,points,[40.,30.],[0.,0.],flatten=True)
        self.assertFalse(c.point_support)
        torch.testing.assert_close(c.anchor,p.new_tensor([[-.05,0.,0.]]))
        torch.testing.assert_close(c.local_point,p.new_tensor([[-.05,0.,-.06]]))
        wrench=c.wrench_to_world_ankle(p.new_tensor([[0.,0.,100.,1.,0.,0.]]),r)
        torch.testing.assert_close(wrench,p.new_tensor([[0.,0.,100.,1.,5.,0.]]))
        self.assertFalse(c.update_support(p,r,points,[40.,30.],[0.,0.]))
        self.assertTrue(c.update_support(p,r,points,[40.,0.],[0.,0.]))
        self.assertTrue(c.point_support)

    def test_observed_stance_edge_requests_flattening_instead_of_toe_off(self):
        dtype=torch.float64
        p=torch.tensor([[0.,0.,.06]],dtype=dtype)
        angle=torch.tensor(.2,dtype=dtype);c,s=angle.cos(),angle.sin()
        r=torch.tensor([[[c,0.,s],[0.,1.,0.],[-s,0.,c]]],dtype=dtype)
        contact=ToeContact.from_lifted_sole(p,r,0.,.03,[[.11,.02,0.]],[100.],[0.],flatten=True)
        jac=torch.zeros(1,6,35,dtype=dtype);jac[0,:,:6]=torch.eye(6,dtype=dtype)
        _,_,_,task=contact.evaluate(p,r,jac,torch.zeros(1,6,dtype=dtype),torch.zeros(1,35,dtype=dtype),.5)
        self.assertTrue(contact.point_support)
        self.assertTrue(task.active.all())
        self.assertEqual(task.name,'stance_flatten')
        self.assertLess(task.desired[0,1].item(),0.)
        torch.testing.assert_close(task.desired[0,[0,2]],torch.zeros(2,dtype=dtype),atol=1e-12,rtol=0)

    def test_flat_sole_can_change_directly_to_a_measured_heel_point(self):
        p=torch.tensor([[0.,0.,.05]],dtype=torch.float64)
        r=torch.eye(3,dtype=p.dtype)[None]
        points=[[-.05,-.025,0.]]
        contact=ToeContact.from_lifted_sole(p,r,6.2,.03,points,[1.],[0.])
        self.assertIsNotNone(contact)
        self.assertTrue(contact.point_support)
        torch.testing.assert_close(contact.anchor,p.new_tensor(points))
        torch.testing.assert_close(contact.local_point,p.new_tensor([[-.05,-.025,-.05]]))
        self.assertIsNone(ToeContact.from_lifted_sole(p,r,6.2,.001,points,[1.],[0.]))
        self.assertIsNone(ToeContact.from_lifted_sole(p,r,6.2,.03,points,[0.],[0.]))
        # A genuine remaining toe line still retains its roll coordinate.
        line=ToeContact.from_lifted_sole(p,r,6.2,.03,
            [[.12,-.03,.015],[.12,.03,.015]],[10.,20.],[0.,0.])
        self.assertIsNotNone(line)
        self.assertFalse(line.point_support)

    def test_line_downgrades_to_actual_loaded_material_point(self):
        p = torch.tensor([[0.,0.,.035]], dtype=torch.float64)
        r = torch.eye(3,dtype=p.dtype)[None]
        contact = ToeContact(p,r,0.)
        points = [[.12,-.03,0.],[.12,.03,0.]]
        self.assertFalse(contact.update_support(p,r,points,[10.,20.],[0.,0.]))
        self.assertTrue(contact.update_support(p,r,points,[0.,20.],[0.,0.]))
        self.assertTrue(contact.point_support)
        torch.testing.assert_close(contact.local_point,p.new_tensor([[.12,.03,-.035]]))
        torch.testing.assert_close(contact.anchor,p.new_tensor([[.12,.03,0.]]))
        # Force changes at the same single loaded corner retain its anchor.
        # A second loaded corner is tested separately as a line promotion.
        self.assertFalse(contact.update_support(p,r,points,[0.,20.],[0.,0.]))
        # A rigid-body rotation about the chosen point preserves its position.
        from fabrics_sim.taskmaps.g1_end_effector_pose_taskmap import _quat_xyzw_to_matrix
        angle = torch.tensor(.2,dtype=p.dtype)
        rotated = _quat_xyzw_to_matrix(torch.stack([angle.sin(),angle*0,angle*0,angle.cos()])[None])
        new_position=contact.anchor-(rotated@contact.local_point[...,None]).squeeze(-1)
        jac=p.new_zeros(1,6,35);jac[0,:,:6]=torch.eye(6,dtype=p.dtype)
        _,_,error,_=contact.evaluate(new_position,rotated,jac,p.new_zeros(1,6),p.new_zeros(1,35),.5)
        torch.testing.assert_close(error[:,:3],p.new_zeros(1,3),atol=1e-12,rtol=0)

    def test_point_contact_cannot_supply_any_moment(self):
        dtype=torch.float64
        jac=torch.zeros(1,12,35,dtype=dtype);jac[0,:6,6:12]=torch.eye(6,dtype=dtype)
        desired=torch.zeros(1,35,dtype=dtype);desired[:,9:12]=2.
        mask=torch.ones(1,12,dtype=torch.bool);mask[:,3:6]=False
        result=G1DynamicsAccelerationQP(max_iter=4000).solve(
            acc_reference=desired,mass_matrix=torch.eye(35,dtype=dtype)[None],
            bias_force=torch.zeros_like(desired),contact_jacobian=jac,
            jdot_qdot=torch.zeros(1,12,dtype=dtype),contact_mode=torch.tensor([1]),
            contact_row_mask=mask)
        self.assertTrue(result.validated.all())
        self.assertGreater(result.acc[0,9:12].min().item(),1.9)
        self.assertLess(result.contact_wrench[0,3:6].abs().max().item(),1e-6)

    def test_new_line_anchors_rotation_after_free_point_roll(self):
        from scipy.spatial.transform import Rotation
        p=torch.tensor([[0.,0.,.035]],dtype=torch.float64)
        old_r=torch.tensor(Rotation.from_euler('x',.12).as_matrix()[None],dtype=p.dtype)
        contact=ToeContact(p,old_r,0.)
        contact.update_support(p,old_r,[[.12,.03,0.]],[100.],[0.])
        self.assertTrue(contact.point_support)
        flat=torch.eye(3,dtype=p.dtype)[None]
        contact.update_support(p,flat,[[.12,-.03,0.],[.12,.03,0.]],[40.,60.],[0.,0.])
        self.assertFalse(contact.point_support)
        jac=p.new_zeros(1,6,35);jac[0,:,:6]=torch.eye(6,dtype=p.dtype)
        _,_,error,_=contact.evaluate(p,flat,jac,p.new_zeros(1,6),p.new_zeros(1,35),.1)
        torch.testing.assert_close(error,p.new_zeros(1,6),atol=1e-12,rtol=0)

    def test_airborne_material_point_is_replaced_by_observed_loaded_point(self):
        dtype=torch.float64
        p=torch.tensor([[0.,0.,.035]],dtype=dtype);r=torch.eye(3,dtype=dtype)[None]
        contact=ToeContact(p,r,0.)
        contact.update_support(p,r,[[.12,.03,0.]],[100.],[0.])
        self.assertTrue(contact.point_support)
        angle=torch.tensor(-.2,dtype=dtype);c,s=angle.cos(),angle.sin()
        rotated=torch.tensor([[[c,0.,s],[0.,1.,0.],[-s,0.,c]]],dtype=dtype)
        heel=p.new_tensor([[-.05,.03,-.035]])
        heel_world=p.new_tensor([[-.05,.03,0.]])
        new_position=heel_world-(rotated@heel[...,None]).squeeze(-1)
        self.assertTrue(contact.update_support(new_position,rotated,heel_world.numpy(),[100.],[0.]))
        torch.testing.assert_close(contact.local_point,heel,atol=1e-12,rtol=0)
        torch.testing.assert_close(contact.anchor,heel_world)

    def test_a_single_toe_corner_is_not_a_line_contact(self):
        points=[[.12,-.03,-.035],[.12,.03,-.035]]
        self.assertTrue(has_loaded_toe_line(points,[1.,20.],[0.,0.]))
        self.assertFalse(has_loaded_toe_line(points,[0.,20.],[0.,0.]))
        self.assertFalse(has_loaded_toe_line(points,[1.,20.],[.003,0.]))
        self.assertFalse(has_loaded_toe_line([[-.05,-.025,-.035],[-.05,.025,-.035]], [10.,10.],[0.,0.]))
        self.assertFalse(has_loaded_toe_line(torch.empty(0,3),[],[]))

    def test_heel_can_lift_while_toe_line_stays_fixed(self):
        dtype=torch.float64
        position=torch.tensor([[0.,0.,.035]],dtype=dtype)
        initial=torch.eye(3,dtype=dtype)[None]
        contact=ToeContact(position,initial,0.)
        angle=torch.tensor(.25,dtype=dtype);c,s=angle.cos(),angle.sin()
        rotation=torch.stack((torch.stack((c,c*0,s)),torch.tensor([0.,1.,0.],dtype=dtype),
                              torch.stack((-s,s*0,c))))[None]
        lifted=contact.anchor-(rotation@contact.local_point[...,None]).squeeze(-1)
        jac=torch.zeros(1,6,35,dtype=dtype);jac[0,:,:6]=torch.eye(6,dtype=dtype)
        _,_,error,_=contact.evaluate(lifted,rotation,jac,torch.zeros(1,6,dtype=dtype),
                                     torch.zeros(1,35,dtype=dtype),1.)
        torch.testing.assert_close(error[:,[0,1,2,3,5]],torch.zeros(1,5,dtype=dtype),atol=1e-12,rtol=0)
        self.assertAlmostEqual(error[0,4].item(),.25)
        self.assertGreater(lifted[0,2].item(),position[0,2].item())

    def test_point_jacobian_bias_and_wrench_preserve_rigid_body_mechanics(self):
        dtype=torch.float64
        position=torch.tensor([[.2,.3,.035]],dtype=dtype)
        rotation=torch.tensor([[[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]]],dtype=dtype)
        jac=torch.zeros(1,6,35,dtype=dtype);jac[0,:,:6]=torch.eye(6,dtype=dtype)
        velocity=torch.zeros(1,35,dtype=dtype);velocity[0,:6]=torch.tensor([.1,.2,.3,.4,.5,.6])
        contact=ToeContact(position,rotation,0.)
        mapped,bias,error,_=contact.evaluate(position,rotation,jac,torch.zeros(1,6,dtype=dtype),velocity,0.)
        offset=(rotation@contact.local_point[...,None]).squeeze(-1)
        world_velocity=velocity[:,:3]+torch.linalg.cross(velocity[:,3:6],offset)
        torch.testing.assert_close((mapped[:,:3]@velocity[...,None]).squeeze(-1),
                                  (contact.frame.transpose(-1,-2)@world_velocity[...,None]).squeeze(-1))
        world_bias=torch.linalg.cross(velocity[:,3:6],torch.linalg.cross(velocity[:,3:6],offset))
        torch.testing.assert_close(bias[:,:3],(contact.frame.transpose(-1,-2)@world_bias[...,None]).squeeze(-1))
        torch.testing.assert_close(error,torch.zeros_like(error))
        wrench=torch.tensor([[3.,4.,100.,.1,0.,.2]],dtype=dtype)
        ankle=contact.wrench_to_world_ankle(wrench,rotation)
        torch.testing.assert_close(mapped.transpose(-1,-2)@wrench[...,None],jac.transpose(-1,-2)@ankle[...,None])

    def test_released_pitch_row_is_free_and_cannot_apply_pitch_moment(self):
        dtype=torch.float64
        jac=torch.zeros(1,12,35,dtype=dtype);jac[0,:6,6:12]=torch.eye(6,dtype=dtype)
        desired=torch.zeros(1,35,dtype=dtype);desired[:,10]=2.
        mask=torch.ones(1,12,dtype=torch.bool);mask[:,4]=False
        kwargs=dict(acc_reference=desired,mass_matrix=torch.eye(35,dtype=dtype)[None],
                    bias_force=torch.zeros_like(desired),contact_jacobian=jac,
                    jdot_qdot=torch.zeros(1,12,dtype=dtype),contact_mode=torch.tensor([1]))
        solver=G1DynamicsAccelerationQP(max_iter=4000)
        locked=solver.solve(**kwargs)
        free=solver.solve(**kwargs,contact_row_mask=mask)
        self.assertTrue(free.validated.all())
        self.assertLess(abs(locked.acc[0,10].item()),1e-6)
        self.assertGreater(free.acc[0,10].item(),1.9)
        self.assertLess(abs(free.contact_wrench[0,4].item()),1e-6)
        self.assertLess(free.contact_wrench[0,6:].abs().max().item(),1e-6)
        with self.assertRaises(ValueError):solver.solve(**kwargs,contact_row_mask=mask.float())

    def test_projector_uses_the_same_released_pitch_coordinate(self):
        dtype=torch.float64
        jac=torch.zeros(1,6,35,dtype=dtype);jac[0,:,6:12]=torch.eye(6,dtype=dtype)
        acc=torch.zeros(1,35,dtype=dtype);acc[:,10]=2.
        mask=torch.ones(1,12,dtype=torch.bool);mask[:,4]=False
        projector=G1FabricContactProjector(1,'cpu',dtype,projection_config=ProjectionConfig(contact_kp=0.,contact_kd=0.))
        result=projector.step(acc,torch.eye(35,dtype=dtype)[None],torch.zeros_like(acc),jac,torch.zeros_like(jac),
                              torch.tensor([[100.,0.]],dtype=dtype),left_jdot_v=torch.zeros(1,6,dtype=dtype),
                              right_jdot_v=torch.zeros(1,6,dtype=dtype),contact_mode_override=torch.tensor([1]),
                              contact_row_mask=mask)
        torch.testing.assert_close(result.acc_dc,acc)
        self.assertEqual(result.active_row_mask[0,4].item(),0.)
