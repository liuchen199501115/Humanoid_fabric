import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from preview_surface_kinematics import surface_jacobian, surface_bias

class SurfaceKinematicsTests(unittest.TestCase):
    def test_derivative_matches_changing_world_contact(self):
        R = Rotation.from_rotvec([.3,-.4,.1]).as_matrix()
        J = np.eye(6); v = np.array([.2,.3,-.1,3.,-5.,2.])
        points=np.array([[.1,.02,-.03],[-.05,-.03,-.03]])
        radii=np.array([.005,0.]); h=1e-6
        plus=Rotation.from_rotvec(h*v[3:]).as_matrix()@R
        minus=Rotation.from_rotvec(-h*v[3:]).as_matrix()@R
        finite=(surface_jacobian(plus,J,points,radii)@v-surface_jacobian(minus,J,points,radii)@v)/(2*h)
        np.testing.assert_allclose(surface_bias(R,J,np.zeros(6),v,points,radii),finite,atol=1e-9)
    def test_rolling_sphere_has_constant_zero_contact_velocity(self):
        J=np.eye(6); radius=.1; omega=3.;v=np.array([radius*omega,0.,0.,0.,omega,0.])
        points=np.zeros((1,3));radii=np.array([radius])
        for angle in [0.,.5,1.]:
            R=Rotation.from_rotvec([0.,angle,0.]).as_matrix()
            np.testing.assert_allclose(surface_jacobian(R,J,points,radii)@v,0.,atol=1e-12)
            np.testing.assert_allclose(surface_bias(R,J,np.zeros(6),v,points,radii),0.,atol=1e-12)
    def test_zero_radius_is_material_point(self):
        R=np.eye(3);J=np.eye(6);v=np.array([0.,0.,0.,1.,2.,3.]);p=np.array([[.1,-.2,.3]])
        expected=np.cross(v[3:],np.cross(v[3:],p[0]))
        np.testing.assert_allclose(surface_bias(R,J,np.zeros(6),v,p,np.zeros(1)),expected)
if __name__=='__main__':unittest.main()
