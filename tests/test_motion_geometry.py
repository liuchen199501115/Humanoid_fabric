import unittest
from pathlib import Path
import numpy as np
import torch
from motion_generation.data import joint_spec, rotation6d, quat_matrix, MotionWindows
from motion_generation.kinematics import G1FeetFK
from motion_generation.losses import MotionGeometryLoss
from motion_generation.model import MotionDenoiser, sample, schedule


class GeometryTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_fk_matches_mujoco_and_has_gradients(self):
        import mujoco
        names, _ = joint_spec()
        model = mujoco.MjModel.from_xml_path(str(Path(__file__).resolve().parents[1]/'retarget/assets/g1_description/g1_29dof.xml'))
        data = mujoco.MjData(model)
        rng = np.random.default_rng(7)
        state = np.zeros((4,38), dtype=np.float32)
        state[:, :3] = [.1,.2,.8]
        q = rng.normal(size=(4,4));q /= np.linalg.norm(q,axis=-1,keepdims=True)
        state[:, 3:9] = rotation6d(quat_matrix(q))
        state[:, 9:] = rng.uniform(-.2,.2,(4,29))
        expected = []
        for row, quat in zip(state,q):
            data.qpos[:3] = row[:3];data.qpos[3:7] = quat[[3,0,1,2]]
            for i,name in enumerate(names):
                data.qpos[model.jnt_qposadr[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,name)]] = row[9+i]
            mujoco.mj_forward(model,data)
            feet=[]
            for side in ('left','right'):
                bid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,f'{side}_ankle_roll_link')
                ids=np.flatnonzero((model.geom_bodyid==bid)&(model.geom_type==mujoco.mjtGeom.mjGEOM_SPHERE))
                feet.append(data.geom_xpos[ids].copy())
            expected.append(feet)
        x=torch.tensor(state,requires_grad=True)
        actual=G1FeetFK(names)(x)
        np.testing.assert_allclose(actual.detach(),expected,atol=2e-5)
        actual.square().sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(x.grad[:,9:].abs().sum().item(),0)

    def test_contact_losses_detect_sliding_and_penetration(self):
        names,limits=joint_spec()
        geometry=MotionGeometryLoss(names,limits,50.,'cpu')
        h=torch.zeros(1,2,38);h[:,:,2]=.79;h[:,:,3]=1;h[:,:,7]=1
        reference=h[:,-1:].repeat(1,5,1)
        moving=reference.clone();moving[:,:,0]=torch.arange(1,6)*.01
        c=torch.ones(1,7,2)
        still=geometry(reference,reference,h,c)
        slide=geometry(moving,reference,h,c)
        flight=geometry(moving,reference,h,torch.zeros_like(c))
        self.assertGreater(slide['skating'].item(),still['skating'].item())
        self.assertEqual(flight['skating'].item(),0)
        below=reference.clone();below[:,:,2]-=.4;below.requires_grad_()
        losses=geometry(below,reference,h,c)
        self.assertGreater(losses['penetration'].item(),still['penetration'].item())
        sum(losses.values()).backward()
        self.assertTrue(torch.isfinite(below.grad).all())

    def test_inpainting_preserves_prefix_and_conditions_network(self):
        torch.manual_seed(5)
        model=MotionDenoiser(width=32,layers=1,horizon=5,inpainting=True)
        h=torch.randn(2,2,38);d=torch.tensor([[1.,0.],[1.,0.]])
        known=torch.randn(2,5,38);mask=torch.zeros(2,5,1);mask[:,:2]=1
        result=sample(model,h,d,schedule(20,'cpu'),5,known,mask)
        torch.testing.assert_close(result[:,:2],known[:,:2])
        self.assertTrue(torch.isfinite(result).all())
        out=model(torch.randn_like(known),torch.ones(2),h,d,known,mask)
        out[:,2:].square().mean().backward()
        self.assertGreater(model.known_input.weight.grad.abs().sum().item(),0)


    def test_contact_windows_remain_aligned_and_missing_labels_fail(self):
        x=np.zeros((60,38),dtype=np.float32);x[:,3]=1;x[:,7]=1
        c=np.zeros((60,2),dtype=np.float32);c[:30,0]=1;c[30:,1]=1
        archive=dict(states=x,offsets=np.array([0,30,60]),split=np.array([0,1]),contacts=c)
        window=MotionWindows(archive,0,with_contacts=True)[0]
        np.testing.assert_array_equal(window[3],c[:27])
        del archive['contacts']
        with self.assertRaises(ValueError):MotionWindows(archive,0,with_contacts=True)

    def test_prefix_noise_matches_diffusion_timestep(self):
        class Recorder(torch.nn.Module):
            inpainting=True
            horizon=5
            def __init__(self):
                super().__init__();self.calls=[]
            def forward(self,x,t,h,d,known=None,mask=None):
                self.calls.append((x.clone(),t.clone()))
                return torch.zeros_like(x)
        model=Recorder();alpha=schedule(20,'cpu')
        known=torch.randn(1,5,38);mask=torch.zeros(1,5,1);mask[:,:2]=1
        sample(model,torch.randn(1,2,38),torch.ones(1,2),alpha,5,known,mask)
        recovered=[]
        for x,t in model.calls:
            a=alpha[t[0]]
            recovered.append((x[:,:2]-a.sqrt()*known[:,:2])/(1-a).sqrt())
        for noise in recovered[1:]:torch.testing.assert_close(noise,recovered[0],atol=2e-6,rtol=2e-6)


if __name__=='__main__':unittest.main()
