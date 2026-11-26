import unittest
import numpy as np
from scipy.optimize import linprog
from inscribed_patch_wrench import inscribed_wrench_cone

class InscribedPatchWrenchTests(unittest.TestCase):
    def test_physical_generators_and_original_four_ray_cones_are_retained(self):
        rng=np.random.default_rng(29)
        for points in [np.zeros((1,3)),np.array([[0,-.03,0],[0,.03,0]]),
                       np.array([[-.05,-.03,0],[-.05,.03,0],[.12,-.03,0],[.12,.03,0]])]:
            G=inscribed_wrench_cone(points,.6)
            for offset in [0.,np.pi/4]:
                angles=np.arange(4)*np.pi/2+offset
                F=np.tile(np.c_[.6*np.cos(angles),.6*np.sin(angles),np.ones(4)],(len(points),1))
                W=np.c_[F,np.cross(np.repeat(points,4,axis=0),F)]
                loads=rng.uniform(0,100,(len(W),20));w=W.T@loads
                self.assertLess(float(np.max(G@w)),1e-8)
    def test_no_direction_exceeds_circular_friction(self):
        G=inscribed_wrench_cone(np.zeros((1,3)),.6)
        for angle in np.linspace(0,2*np.pi,33):
            direction=np.array([np.cos(angle),np.sin(angle),0,0,0,0])
            result=linprog(-direction,A_ub=G,b_ub=np.zeros(len(G)),bounds=[(None,None)]*2+[(1,1)]+[(None,None)]*3,method='highs')
            self.assertTrue(result.success)
            self.assertLessEqual(-result.fun,.6+1e-9)
        self.assertGreater(float((G@np.array([.6,.6,1,0,0,0])).max()),.1)
    def test_accepted_extreme_wrenches_decompose_into_physical_vertex_forces(self):
        rng=np.random.default_rng(41)
        angles=np.arange(8)*np.pi/4
        for points in [np.zeros((1,3)),np.array([[0,-.03,0],[0,.03,0]]),
                       np.array([[-.05,-.03,0],[-.05,.03,0],[.12,.03,0]])]:
            G=inscribed_wrench_cone(points,.6)
            F=np.tile(np.c_[.6*np.cos(angles),.6*np.sin(angles),np.ones(8)],(len(points),1))
            generators=np.c_[F,np.cross(np.repeat(points,8,axis=0),F)].T
            for _ in range(5):
                optimum=linprog(rng.normal(size=6),A_ub=G,b_ub=np.zeros(len(G)),
                    bounds=[(None,None)]*2+[(1,1)]+[(None,None)]*3,method='highs')
                self.assertTrue(optimum.success)
                decomposition=linprog(np.zeros(generators.shape[1]),A_eq=generators,b_eq=optimum.x,
                    bounds=(0,None),method='highs')
                self.assertTrue(decomposition.success)
                np.testing.assert_allclose(generators@decomposition.x,optimum.x,atol=1e-8)

    def test_point_and_line_do_not_gain_absent_moments(self):
        point=inscribed_wrench_cone(np.zeros((1,3)),.6)
        for axis in [3,4,5]:
            w=np.array([0.,0.,100.,0.,0.,0.]);w[axis]=1
            self.assertGreater(float((point@w).max()),.9)
        line=inscribed_wrench_cone(np.array([[0.,-.03,0.],[0.,.03,0.]]),.6)
        self.assertGreater(float((line@np.array([0,0,100,0,1,0])).max()),.9)
        # At the edge of the line, normal force and yaw are coupled.
        self.assertGreater(float((line@np.array([0,0,100,3,0,1])).max()),0.)
    def test_observed_patch_keeps_load_bounds_and_zero_point_moments(self):
        from types import SimpleNamespace
        import torch
        from inscribed_patch_wrench import observed_patch_constraints
        patch=SimpleNamespace(vertices=torch.zeros(1,1,3,dtype=torch.float64),
            local_point=torch.zeros(1,3,dtype=torch.float64),
            frame=torch.eye(3,dtype=torch.float64)[None],point_support=True,polygon_support=False)
        G,b=observed_patch_constraints(patch,patch.frame,.6,800.,8,minimum_normal_force=20.,yaw_limit=.1)
        for force,valid in [(0.,False),(20.,True),(800.,True),(801.,False)]:
            w=torch.tensor([0.,0.,force,0.,0.,0.],dtype=torch.float64)
            self.assertEqual(bool((G[0]@w-b[0]).max()<=1e-9),valid)
        w=torch.tensor([0.,0.,100.,0.,.01,0.],dtype=torch.float64)
        self.assertGreater(float((G[0]@w-b[0]).max()),.009)

    def test_invalid_geometry_rejected(self):
        for points,mu,rays in [(np.zeros((0,3)),.6,8),(np.zeros((1,3)),-.6,8),(np.zeros((1,3)),.6,6)]:
            with self.assertRaises(ValueError):inscribed_wrench_cone(points,mu,rays)

if __name__=='__main__':unittest.main()
