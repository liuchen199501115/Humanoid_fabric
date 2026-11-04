import unittest
import importlib.util
from fabrics_sim.reference.source_priority import source_swing_priority


class SourcePriorityTests(unittest.TestCase):
    def test_only_airborne_position_after_balance(self):
        for mode, rows in ((1,(6,7,8)),(2,(0,1,2))):
            tasks, selected, levels=source_swing_priority(True,True,False,False,mode)
            self.assertEqual(tasks,('center_of_mass','feet_pose'))
            self.assertEqual(selected,{'feet_pose':rows})
            self.assertEqual(levels,(('center_of_mass',),('feet_pose',)))

    def test_pose_selects_only_airborne_position_and_orientation(self):
        for mode, rows in ((1,tuple(range(6,12))),(2,tuple(range(6)))):
            self.assertEqual(source_swing_priority(True,True,False,False,mode,pose=True)[1],
                             {'feet_pose':rows})

    def test_existing_body_priority_is_preserved(self):
        body=('body_orientation','source_pelvis_orientation')
        for stage in ('balance','swing'):
            tasks,rows,levels=source_swing_priority(True,True,False,False,1,
                                                  pose=True,body_priority=stage)
            self.assertEqual(set(tasks),{'center_of_mass','feet_pose',*body})
            self.assertEqual(levels[0],('center_of_mass',)+ (body if stage=='balance' else ()))
            self.assertEqual(levels[1],('feet_pose',)+ (body if stage=='swing' else ()))

    def test_does_not_change_transfer_stop_or_disabled_controller(self):
        for args in ((False,True,False,False,1),(True,False,False,False,1),
                     (True,True,True,False,1),(True,True,False,True,1),
                     (True,True,False,False,3),(True,True,False,False,0)):
            self.assertIsNone(source_swing_priority(*args))
        with self.assertRaises(ValueError):
            source_swing_priority(True,True,False,False,4)

    @unittest.skipUnless(importlib.util.find_spec('daqp'), 'Optional daqp dependency')
    def test_selected_swing_beats_posture_but_respects_motor_limit(self):
        import torch
        from fabrics_sim.fabrics.task_request import TaskRequest
        from fabrics_sim.solver.AccelerationQP import G1DynamicsAccelerationQP
        dtype=torch.float64
        balance=torch.zeros(1,1,35,dtype=dtype);balance[0,0,6]=1.
        feet=torch.zeros(1,12,35,dtype=dtype)
        feet[0,6,7]=1.;feet[0,9,7]=1.
        desired=torch.zeros(1,12,dtype=dtype);desired[0,6]=2.;desired[0,9]=-3.
        posture=feet[:,6:7]
        tasks=[TaskRequest(name,A,torch.zeros(1,A.shape[1],dtype=dtype),target,
            weight*torch.eye(A.shape[1],dtype=dtype)[None],torch.ones(1,A.shape[1],dtype=torch.bool))
            for name,A,target,weight in (
                ('center_of_mass',balance,torch.tensor([[1.]],dtype=dtype),1.),
                ('feet_pose',feet,desired,1.),
                ('joint_posture',posture,torch.tensor([[-3.]],dtype=dtype),10000.))]
        names,rows,levels=source_swing_priority(True,True,False,False,1)
        solver=G1DynamicsAccelerationQP(solver_backend='daqp')
        result=solver.solve(acc_reference=torch.zeros(1,35,dtype=dtype),
            mass_matrix=torch.eye(35,dtype=dtype)[None],bias_force=torch.zeros(1,35,dtype=dtype),
            contact_jacobian=torch.zeros(1,12,35,dtype=dtype),jdot_qdot=torch.zeros(1,12,dtype=dtype),
            contact_mode=torch.zeros(1,dtype=torch.long),task_requests=tasks,
            priority_tasks=names,priority_rows=rows,priority_levels=levels,
            effort_limits=1.5,acc_lower=-10.,acc_upper=10.,priority_tolerance=.001)
        self.assertTrue(result.validated.all())
        torch.testing.assert_close(result.predicted_acc[0,6:8],torch.tensor([1.,1.5],dtype=dtype),
                                   atol=.00102,rtol=0.)
        self.assertLessEqual(float(result.total_joint_torque.abs().max()),1.5+1e-6)
