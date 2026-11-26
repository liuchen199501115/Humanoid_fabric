import unittest
import numpy as np
import torch

from fabrics_sim.solver.observed_sole_contact import ObservedSoleContact, polygon_wrench_cone


class ObservedSoleTests(unittest.TestCase):
    def contact(self):
        p=torch.tensor([[0.,0.,.035]],dtype=torch.float64)
        R=torch.eye(3,dtype=p.dtype)[None]
        c=ObservedSoleContact(p,R,0.);c.flatten=True
        return c,p,R

    def test_side_edge_uses_its_actual_free_rotation_axis(self):
        c,p,R=self.contact()
        points=np.array([[-.05,.025,0.],[.12,.025,0.]])
        c.update_support(p,R,points,[40.,60.],[0.,0.])
        self.assertFalse(c.point_support);self.assertFalse(c.polygon_support)
        np.testing.assert_allclose(abs(c.frame[0,:,1].numpy()),[1.,0.,0.],atol=1e-12)
        self.assertAlmostEqual(c.half_width,.085)
        jac=p.new_zeros(1,6,35);jac[0,:,:6]=torch.eye(6,dtype=p.dtype)
        _,_,error,task=c.evaluate(p,R,jac,p.new_zeros(1,6),p.new_zeros(1,35),.1)
        torch.testing.assert_close(error,p.new_zeros(1,6))
        torch.testing.assert_close(task.desired,p.new_zeros(1,3))

    def test_third_loaded_point_promotes_line_to_polygon_and_can_unload(self):
        c,p,R=self.contact()
        points=np.array([[-.05,-.025,0.],[-.05,.025,0.],[.12,.025,0.]])
        c.update_support(p,R,points,[40.,60.,0.],[0.,0.,0.])
        self.assertFalse(c.polygon_support)
        c.update_support(p,R,points,[40.,60.,20.],[0.,0.,0.])
        self.assertTrue(c.polygon_support)
        jac=p.new_zeros(1,6,35)
        task=c.evaluate(p,R,jac,p.new_zeros(1,6),p.new_zeros(1,35),.1)[3]
        self.assertFalse(task.active.any())
        G,b=c.wrench_constraints(R,.6,800.)
        self.assertEqual(G.shape[2],6)
        self.assertTrue(torch.isfinite(G).all());self.assertEqual(float(b[0,-1]),800.)
        c.update_support(p,R,points,[40.,0.,0.],[0.,0.,0.])
        self.assertTrue(c.point_support);self.assertFalse(c.polygon_support)

    def test_triangle_wrench_cone_preserves_coupled_support_region(self):
        points=np.array([[-.05,-.025,0.],[-.05,.025,0.],[.12,.025,0.]])
        G=polygon_wrench_cone(points,.6)
        for r in points:
            for f in np.array([[.6,0,1],[-.6,0,1],[0,.6,1],[0,-.6,1]]):
                self.assertLessEqual(float((G@np.r_[f,np.cross(r,f)]).max()),1e-10)
        # The unobserved fourth rectangle corner is outside this triangle.
        f=np.array([0.,0.,1.]);r=np.array([.12,-.025,0.])
        self.assertGreater(float((G@np.r_[f,np.cross(r,f)]).max()),1e-3)
        self.assertGreater(float((G@np.array([0,0,-1,0,0,0])).max()),.9)

    def test_outgoing_toe_tracks_loaded_tip_then_oblique_edge(self):
        c,p,R=self.contact();c.flatten=False
        # Recorded toe-off switches between the outer toe corner and a toe
        # tip 3.6 cm away. Both can be near the floor; only loaded geometry is
        # evidence of the current force application point.
        points=np.array([[.1206,.0301,0.],[.1407,.0001,0.]])
        c.update_support(p,R,points,[90.,0.],[0.,0.])
        np.testing.assert_allclose(c.anchor[0],points[0],atol=1e-12)
        c.update_support(p,R,points,[0.,180.],[0.,0.])
        self.assertTrue(c.point_support)
        np.testing.assert_allclose(c.anchor[0],points[1],atol=1e-12)
        c.update_support(p,R,points,[93.,93.],[0.,0.])
        self.assertFalse(c.point_support);self.assertFalse(c.polygon_support)
        direction=(points[1]-points[0])/np.linalg.norm(points[1]-points[0])
        self.assertAlmostEqual(abs(float(c.frame[0,:,1]@p.new_tensor(direction))),1.)
        np.testing.assert_allclose(c.anchor[0],points.mean(0),atol=1e-12)
        self.assertAlmostEqual(c.half_width,np.linalg.norm(points[1]-points[0])/2)

    def test_outgoing_polygon_retains_all_loaded_points(self):
        c,p,R=self.contact();c.flatten=False
        points=np.array([[-.05,-.025,0.],[-.05,.025,0.],[.12,.025,0.]])
        c.update_support(p,R,points,[40.,60.,20.],[0.,0.,0.])
        self.assertTrue(c.polygon_support)
        task=c.evaluate(p,R,p.new_zeros(1,6,35),p.new_zeros(1,6),p.new_zeros(1,35),.1)[3]
        self.assertFalse(task.active.any())
        G,b=c.wrench_constraints(R,.6,800.)
        self.assertTrue(torch.isfinite(G).all())

    def test_optional_touching_points_need_a_bearing_foot_and_small_separation(self):
        points=np.array([[-.05,-.025,0.],[-.05,.025,0.],[.12,.025,0.]])
        c,p,R=self.contact()
        c.update_support(p,R,points,[100.,0.,0.],[0.,1e-5,2e-5])
        self.assertTrue(c.point_support)
        c.update_support(p,R,points,[100.,0.,0.],[0.,1e-5,2e-5],touching_tolerance=5e-5)
        self.assertTrue(c.polygon_support)
        c.update_support(p,R,points,[100.,0.,0.],[0.,1e-3,2e-3],touching_tolerance=5e-5)
        self.assertTrue(c.point_support)
        for load in (0.,10.):
            c,p,R=self.contact()
            c.update_support(p,R,points,[load,0.,0.],[0.,0.,0.],touching_tolerance=5e-5)
            self.assertFalse(c.polygon_support)

    def test_touching_point_cone_still_excludes_unobserved_corner(self):
        c,p,R=self.contact()
        points=np.array([[-.05,-.025,0.],[-.05,.025,0.],[.12,.025,0.]])
        c.update_support(p,R,points,[100.,0.,0.],[0.,0.,0.],touching_tolerance=5e-5)
        G,b=c.wrench_constraints(R,.6,800.)
        force=np.array([0.,0.,100.])
        for point in points:
            wrench=np.r_[force,np.cross(point-p[0].numpy()-c.local_point[0].numpy(),force)]
            self.assertLessEqual(float((G[0]@p.new_tensor(wrench)-b[0]).max()),1e-8)
        outside=np.array([.12,-.025,0.])-p[0].numpy()-c.local_point[0].numpy()
        wrench=np.r_[force,np.cross(outside,force)]
        self.assertGreater(float((G[0]@p.new_tensor(wrench)-b[0]).max()),1e-3)


if __name__=='__main__':unittest.main()
