import unittest
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from fabrics_sim.solver.contact_patches import ContactPatchManager, loaded_sole_contacts
from fabrics_sim.solver.toe_contact import ToeContact


class ContactPatchTests(unittest.TestCase):
    def scene(self):
        rotations=Rotation.from_euler('y',[.3,-.25])
        local=np.array([[[.12,-.025,-.035],[.12,.025,-.035]],
                        [[-.05,-.025,-.035],[-.05,.025,-.035]]])
        poses=np.zeros((2,7));poses[:,3:]=rotations.as_quat()
        poses[1,0]=.3
        poses[:,2]=-np.einsum('bij,bj->bi',rotations.as_matrix(),local[:,0])[:,2]
        points=np.einsum('bij,bkj->bki',rotations.as_matrix(),local)+poses[:,:3,None].transpose(0,2,1)
        data=(np.array([100.,100.,40.,40.]),points.reshape(-1,3),np.tile([0.,0.,1.],(4,1)),
              np.zeros(4),np.array([2,2]),np.array([0,2]))
        return torch.tensor(poses[None],dtype=torch.float64),data

    def test_toe_and_opposite_heel_keep_separate_material_anchors(self):
        poses,data=self.scene();manager=ContactPatchManager()
        rotation=torch.tensor(Rotation.from_quat(poses[0,0,3:]).as_matrix()[None],dtype=poses.dtype)
        primary=ToeContact(poses[:,0,:3],rotation,0.)
        patches,_=manager.update(torch.tensor([[True,True]]),poses,[0.,0.],data,0.,.002,0,primary)
        self.assertEqual(set(patches),{0,1})
        self.assertIs(patches[0],primary)
        self.assertFalse(patches[0].flatten)
        self.assertTrue(patches[1].flatten)
        self.assertFalse(patches[1].point_support)
        self.assertAlmostEqual(float(patches[1].local_point[0,0]),-.05)
        anchor=patches[1].anchor.clone();heel=patches[1]
        patches,_=manager.update(torch.tensor([[False,True]]),poses,[0.,0.],data,.002,.002)
        self.assertEqual(set(patches),{1});self.assertIs(patches[1],heel)
        torch.testing.assert_close(patches[1].anchor,anchor)

    def test_unloaded_sole_geometry_cannot_create_support(self):
        poses,data=self.scene()
        self.assertTrue(loaded_sole_contacts(poses,data).all())
        data=(np.zeros(4),*data[1:])
        self.assertFalse(loaded_sole_contacts(poses,data).any())
        patches,_=ContactPatchManager().update(torch.tensor([[True,True]]),poses,[0.,0.],data,0.,.002)
        self.assertFalse(patches)

    def test_loaded_surface_of_actual_toe_tip_sphere_is_recognized(self):
        poses=torch.tensor([[[0.,0.,.033,0.,0.,0.,1.],
                             [.3,0.,.033,0.,0.,0.,1.]]],dtype=torch.float64)
        points=np.array([[.1407,.0001,0.],[.5,0.,0.]])
        data=(np.array([30.,30.]),points,np.tile([0.,0.,1.],(2,1)),
              np.zeros(2),np.ones(2,dtype=int),np.array([0,1]))
        self.assertEqual(loaded_sole_contacts(poses,data).tolist(),[True,False])
        self.assertFalse(loaded_sole_contacts(poses,(np.zeros(2),*data[1:])).any())

    def test_single_heel_corner_is_point_not_line(self):
        poses,data=self.scene();forces=data[0].copy();forces[3]=0.
        data=(forces,*data[1:])
        patches,_=ContactPatchManager().update(torch.tensor([[True,True]]),poses,[0.,0.],data,0.,.002)
        self.assertTrue(patches[1].point_support)

    def test_second_loaded_heel_corner_promotes_existing_point_to_line(self):
        poses,data=self.scene();forces=data[0].copy();forces[3]=0.
        manager=ContactPatchManager();active=torch.tensor([[True,True]])
        patches,_=manager.update(active,poses,[0.,0.],(forces,*data[1:]),0.,.002)
        heel=patches[1]
        self.assertTrue(heel.point_support)
        # Original corner still touches; the new loaded corner adds a real
        # constraint. Keeping the old point would omit that constraint.
        patches,_=manager.update(active,poses,[0.,0.],data,.002,.002)
        self.assertIs(patches[1],heel)
        self.assertFalse(heel.point_support)
        np.testing.assert_allclose(heel.local_point.numpy(),[[-.05,0.,-.035]],atol=1e-10)
        self.assertAlmostEqual(heel.half_width,.025)

    def test_flat_loaded_quiet_sole_requires_confirmation_before_restoring(self):
        poses,data=self.scene();manager=ContactPatchManager();active=torch.tensor([[True,True]])
        manager.update(active,poses,[0.,0.],data,0.,.002)
        poses[0,:,2]=.035;poses[0,:,3:]=torch.tensor([0.,0.,0.,1.])
        local=np.array([[-.05,-.025,-.035],[-.05,.025,-.035],[.12,-.025,-.035],[.12,.025,-.035]])
        points=(poses[0,:,:3].numpy()[:,None]+local[None]).reshape(-1,3)
        data=(np.full(8,50.),points,np.tile([0.,0.,1.],(8,1)),np.zeros(8),np.array([4,4]),np.array([0,4]))
        for i in range(29):
            patches,restored=manager.update(active,poses,[0.,0.],data,(i+1)*.002,.002)
            self.assertEqual(len(patches),2);self.assertFalse(restored)
        patches,restored=manager.update(active,poses,[0.,0.],data,.06,.002)
        self.assertFalse(patches);self.assertEqual(restored,[0,1])


if __name__=='__main__':unittest.main()
