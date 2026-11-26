"""Force realizability of finite Coulomb rays on degenerate and polygon supports."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'examples'))
from g1_contact_rays import contact_ray_cone, patch_ray_constraints


class ContactRayTest(unittest.TestCase):
    def test_all_generated_point_line_polygon_wrenches_are_admissible(self):
        random=np.random.default_rng(24)
        supports=[np.zeros((1,3)),np.array([[0.,-.02,0.],[0.,.02,0.]]),
                  np.array([[-.05,-.02,0.],[.08,-.02,0.],[.08,.02,0.],[-.05,.02,0.]])]
        for points in supports:
            G=contact_ray_cone(points,.6)
            angle=np.arange(8)*np.pi/4;directions=np.c_[.6*np.cos(angle),.6*np.sin(angle),np.ones(8)]
            for _ in range(30):
                weights=random.uniform(0,10,(len(points),8));forces=weights@directions
                wrench=np.r_[forces.sum(0),np.cross(points,forces).sum(0)]
                self.assertLessEqual(float((G@wrench).max()),1e-8)
                self.assertTrue((np.linalg.norm(forces[:,:2],axis=1)<=.6*forces[:,2]+1e-9).all())

    def test_refinement_stays_inside_coulomb_and_does_not_create_moments(self):
        G=contact_ray_cone(np.zeros((1,3)),.6)
        for angle in np.arange(8)*np.pi/4:
            valid=np.array([60*np.cos(angle),60*np.sin(angle),100.,0.,0.,0.])
            self.assertLessEqual(float((G@valid).max()),1e-8)
            invalid=valid.copy();invalid[:2]*=1.01
            self.assertGreater(float((G@invalid).max()),.1)
        for axis in (3,4,5):
            invalid=np.array([0.,0.,100.,0.,0.,0.]);invalid[axis]=1.
            self.assertGreater(float((G@invalid).max()),.1)
        self.assertGreater(float((G@np.array([0.,0.,-1.,0.,0.,0.])).max()),.1)

    def test_line_cannot_supply_free_axis_moment_or_excess_cop(self):
        G=contact_ray_cone(np.array([[0.,-.02,0.],[0.,.02,0.]]),.6)
        for moment in ([0.,1.,0.],[2.1,0.,0.],[0.,0.,1.3]):
            self.assertGreater(float((G@np.r_[0.,0.,100.,moment]).max()),1e-4)

    def test_patch_bounds_and_geometry_cache(self):
        patch=SimpleNamespace(polygon_support=False,point_support=False,half_width=.02)
        rotation=torch.eye(3,dtype=torch.float64)[None]
        G,b=patch_ray_constraints(patch,rotation,.6,800.,20.)
        G2,b2=patch_ray_constraints(patch,rotation,.6,800.,20.);self.assertIs(G,G2);self.assertIs(b,b2)
        for fz in [19.,801.]:
            wrench=torch.tensor([[[0.],[0.],[fz],[0.],[0.],[0.]]],dtype=torch.float64)
            self.assertGreater(float((G@wrench).squeeze(-1).sub(b).max()),.9)
        patch.half_width=.01;G3,_=patch_ray_constraints(patch,rotation,.6,800.,20.);self.assertIsNot(G,G3)

    def test_invalid_input_rejected(self):
        for pts,mu,n in [(np.zeros((0,3)),.6,8),(np.zeros((1,3)),-1.,8),(np.zeros((1,3)),.6,5)]:
            with self.assertRaises(ValueError):contact_ray_cone(pts,mu,n)

if __name__=='__main__':unittest.main()
