import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from fabrics_sim.reference.convex_collision import ConvexLinkCollision


@unittest.skipUnless(importlib.util.find_spec('mujoco'), 'Optional convex geometry backend')
class ConvexCollisionTests(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory()
        path=Path(self.directory.name)/'robot.urdf'
        path.write_text('<robot name="test">'+''.join(
            f'<link name="{name}"><collision><geometry><box size=".2 .2 .2"/></geometry></collision></link>'
            for name in ('a','b'))+'</robot>')
        self.geometry=ConvexLinkCollision(path,SimpleNamespace(get_link_index=lambda n:('a','b').index(n)),[('a','b')])

    def tearDown(self):self.directory.cleanup()

    def state(self,x=0.,angle=0.,other=(.3,0.,0.)):
        poses=torch.zeros(1,2,7,dtype=torch.float64);poses[:,:,6]=1.
        poses[0,0,0]=x;poses[0,0,3:]=torch.from_numpy(Rotation.from_rotvec([0.,0.,angle]).as_quat())
        poses[0,1,:3]=torch.tensor(other,dtype=torch.float64)
        origin=torch.zeros(1,2,3,35,dtype=torch.float64);origin[0,0,0,6]=1.
        angular=torch.zeros_like(origin);angular[0,0,2,7]=1.
        return dict(transforms=poses,origin_jac=origin,angular_jac=angular)

    def test_signed_distance_translation_and_cutoff(self):
        for separation,expected in ((.3,.1),(.15,-.05)):
            distance,rows=self.geometry.evaluate(self.state(other=(separation,0.,0.)))
            self.assertAlmostEqual(float(distance[0,0]),expected,places=7)
            self.assertAlmostEqual(float(rows[0,0,6]),-1.,places=6)
        distance,rows=self.geometry.evaluate(self.state(other=(2.,0.,0.)))
        self.assertEqual(float(distance[0,0]),1.)
        torch.testing.assert_close(rows,torch.zeros_like(rows))

    def test_barrier_activation_does_not_create_a_jacobian_discontinuity(self):
        rows=[]
        for x in (-.0001,0.,.0001):
            distance,jacobian=self.geometry.evaluate(self.state(x=x,other=(.6,0.,0.)))
            self.assertAlmostEqual(float(distance[0,0]),.4-x,places=7)
            self.assertAlmostEqual(float(jacobian[0,0,6]),-1.,places=6)
            rows.append(jacobian)
        torch.testing.assert_close((rows[2]-rows[0])/.002,torch.zeros_like(rows[0]),atol=1e-6,rtol=0.)

    def test_witness_jacobian_matches_translation_and_rotation(self):
        epsilon=1e-5;angle=.23
        _,rows=self.geometry.evaluate(self.state(angle=angle,other=(.32,.08,0.)))
        for column in (6,7):
            values=[]
            for sign in (-1.,1.):
                state=self.state(x=sign*epsilon if column==6 else 0.,
                    angle=angle+(sign*epsilon if column==7 else 0.),other=(.32,.08,0.))
                values.append(float(self.geometry.evaluate(state)[0][0,0]))
            self.assertAlmostEqual(float(rows[0,0,column]),(values[1]-values[0])/(2*epsilon),places=5)

    def test_g1_runtime_and_audit_geometry_row_layout(self):
        import warp as wp
        from fabrics_sim.fabrics.g1_pose_fabric import G1PoseFabric
        from fabrics_sim.reference.whole_body import WholeBodyReference
        from fabrics_sim.reference.collision_audit import collision_reference_check
        wp.config.kernel_cache_dir='/tmp/g1_warp_cache';wp.init()
        fabric=G1PoseFabric(1,'cpu',robot_name='g1_29dof_rev_1_0',task_profile='dynamics-aware')
        tasks=fabric.dynamics_tasks
        tasks.enable_wrist_hip_avoidance(include_upper_body=True,convex=True)
        tasks.wrist_hip_barrier=True
        q=torch.zeros(1,36);q[:,2]=.8;q[:,6]=1.
        reference=WholeBodyReference(q,torch.zeros(1,35),torch.zeros(1,35))
        reference.apply(fabric)
        fabric.set_collision_obstacles(torch.tensor([[[2.,2.,2.,.1]]]))
        requests={r.name:r for r in fabric.compute_task_requests(q,reference.v)}
        expected=tasks.barrier_pair_count+len(tasks.sphere_indices)
        self.assertEqual(requests['collision'].A.shape,(1,expected,35))
        self.assertEqual(requests['collision'].metric.shape,(1,expected,expected))
        state=tasks.model.snapshot(q)
        old,_=tasks._sphere_geometry(state);new,_=tasks.collision_geometry(state)
        torch.testing.assert_close(new[:,:tasks.original_collision_pairs],old[:,:tasks.original_collision_pairs])
        torch.testing.assert_close(new[:,tasks.barrier_pair_count:],old[:,len(tasks.pairs):])
        self.assertTrue(collision_reference_check(fabric,reference)['checked'])
        self.assertTrue(torch.isfinite(tasks.last_collision_constraints[1]).all())
