import unittest
from pathlib import Path

import numpy as np

from fabrics_sim.reference.collision_geometry import box_sphere_cover, collision_mesh_spheres


class CollisionCoverTests(unittest.TestCase):
    def test_each_box_slab_is_enclosed_including_boundaries(self):
        lower, upper = np.array([-.0263, -.0491, -.1241]), np.array([.0802, .0488, .0465])
        centers, radius = box_sphere_cover(lower, upper)
        points = np.stack(np.meshgrid(*[np.linspace(a,b,19) for a,b in zip(lower,upper)]),-1).reshape(-1,3)
        distances = np.linalg.norm(points[:,None]-centers[None],axis=-1).min(1)
        self.assertLessEqual(distances.max(), radius+1e-12)
        self.assertLess(radius, np.linalg.norm((upper-lower)/2))

    def test_actual_wrist_and_hip_mesh_vertices_are_covered(self):
        root=Path(__file__).resolve().parents[1]/'src/fabrics_sim/models/robots/urdf/g1'
        dtype=np.dtype([('normal','<f4',(3,)),('vertices','<f4',(3,3)),('attribute','<u2')])
        for side in ('left','right'):
            for joint in ('wrist_roll','wrist_pitch','wrist_yaw','hip_pitch','hip_roll','hip_yaw','rubber_hand'):
                name=f'{side}_{joint}' if joint == 'rubber_hand' else f'{side}_{joint}_link'
                with self.subTest(link=name):
                    cover=collision_mesh_spheres(root/'g1_29dof_rev_1_0.urdf',name,
                        geometry_kind='visual' if joint == 'rubber_hand' else 'collision')
                    data=(root/'meshes'/f'{name}.STL').read_bytes()
                    points=np.frombuffer(data,offset=84,dtype=dtype)['vertices'].reshape(-1,3)
                    gaps=np.stack([np.linalg.norm(points-np.asarray(center),axis=1)-radius
                                   for center,radius in cover],axis=1).min(1)
                    self.assertLessEqual(float(gaps.max()),1e-8)

    def test_rotated_cylinder_surface_is_covered(self):
        import tempfile
        from scipy.spatial.transform import Rotation
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'robot.urdf'
            path.write_text('<robot name="test"><link name="shoulder"><collision>'
                            '<origin xyz="0.1 0.2 0.3" rpy="0.2 0.4 0.7"/>'
                            '<geometry><cylinder radius="0.04" length="0.15"/></geometry>'
                            '</collision></link></robot>')
            cover=collision_mesh_spheres(path,'shoulder')
            angle,z=np.meshgrid(np.linspace(0,2*np.pi,41),np.linspace(-.075,.075,61))
            points=np.c_[.04*np.cos(angle.ravel()),.04*np.sin(angle.ravel()),z.ravel()]
            points=Rotation.from_euler('xyz',[.2,.4,.7]).apply(points)+[.1,.2,.3]
            gaps=np.stack([np.linalg.norm(points-center,axis=1)-radius for center,radius in cover],axis=1).min(1)
            self.assertLessEqual(float(gaps.max()),1e-12)


if __name__ == '__main__':
    unittest.main()
