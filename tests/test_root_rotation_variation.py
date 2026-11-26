import sys
import unittest
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'acc_data/g1_tracking_complete_20261123'))
from root_rotation_variation import rotation_variation_maps,skew


class RotationVariationTests(unittest.TestCase):
    def test_world_rotation_velocity_and_acceleration_by_finite_difference(self):
        w1=np.array([.3,-.8,.2]);w2=np.array([-.5,.1,.7])
        def data(t):
            R=Rotation.from_rotvec(t*w1).as_matrix()
            return w1+R@w2,np.cross(w1,R@w2),np.array([t*t,.2*t**3,np.sin(t)]),np.array([2*t,.6*t*t,np.cos(t)]),np.array([2.,1.2*t,-np.sin(t)])
        def corrected_velocity(t,eps):
            w,_,d,dv,_=data(t);r=eps*d;theta=np.linalg.norm(r);S=skew(r)
            J=np.eye(3)+(1-np.cos(theta))/theta**2*S+(theta-np.sin(theta))/theta**3*(S@S)
            return J@(eps*dv)+Rotation.from_rotvec(r).apply(w)
        t=.7;w,a,d,dv,da=data(t);maps=[]
        for x in (d,dv,da):
            M=np.zeros((35,1));M[3:6,0]=x;maps.append(M)
        _,V,A=rotation_variation_maps(*maps,w,a)
        eps=1e-3;dt=1e-4
        measured_v=(corrected_velocity(t,eps)-corrected_velocity(t,-eps))/(2*eps)
        def acceleration(e):return (corrected_velocity(t+dt,e)-corrected_velocity(t-dt,e))/(2*dt)
        measured_a=(acceleration(eps)-acceleration(-eps))/(2*eps)
        np.testing.assert_allclose(V[3:6,0],measured_v,atol=5e-7)
        np.testing.assert_allclose(A[3:6,0],measured_a,atol=2e-6)
        for old,x in zip(maps,(d,dv,da)):np.testing.assert_array_equal(old[3:6,0],x)

    def test_nonangular_rows_are_unchanged(self):
        maps=[np.arange(70.).reshape(35,2)+i for i in range(3)]
        changed=rotation_variation_maps(*maps,np.ones(3),np.ones(3)*2)
        rows=np.r_[0:3,6:35]
        for old,new in zip(maps,changed):np.testing.assert_array_equal(old[rows],new[rows])
