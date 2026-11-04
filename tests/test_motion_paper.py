import unittest
import numpy as np
import torch
from motion_generation.paper.kinematics import PaperFK
from motion_generation.paper.data import localize, worldize, PaperWindows
from motion_generation.paper.losses import PaperMotionLoss
from motion_generation.model import MotionDenoiser, schedule
from motion_generation.train import objective


class PaperTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.fk=PaperFK()
        self.x=torch.zeros(54,99);self.x[:,2]=.8;self.x[:,6]=1
        self.x[:,0]=torch.arange(54)*.01
        self.x[:,30:]=self.fk(self.x).flatten(-2)

    def test_coordinates_and_clip_boundaries(self):
        x=self.x.numpy();local,*frame=localize(x)
        np.testing.assert_allclose(worldize(local,*frame),x,atol=1e-6)
        a=dict(states=x,fps=50.,representation='g1-paper99',offsets=[0,27,54],split=[0,0])
        data=PaperWindows(a,0)
        self.assertEqual(len(data),2)
        past,future,heading=data[1]
        self.assertEqual(future.shape,(25,99));self.assertEqual(past.shape,(2,99))
        np.testing.assert_allclose(heading,[1,0],atol=1e-6)

    def test_consistency_and_penetration_gradients(self):
        loss=PaperMotionLoss('cpu')
        self.assertLess(loss(self.x)['fk'].item(),1e-10)
        altered=self.x.clone();altered[:,30:]+=.1;altered.requires_grad_()
        terms=loss(altered);terms['fk'].backward()
        self.assertTrue(torch.isfinite(altered.grad).all())
        self.assertGreater(altered.grad[:,:30].abs().sum().item(),0)
        self.assertGreater(altered.grad[:,30:].abs().sum().item(),0)
        low=self.x.clone();low[:,2]-=1
        self.assertGreater(loss(low)['penetration'],loss(self.x)['penetration'])

    def test_fk_against_mujoco(self):
        import mujoco
        from motion_generation.paper.kinematics import URDF
        model=mujoco.MjModel.from_xml_path(str(URDF.with_suffix('.xml')))
        data=mujoco.MjData(model)
        x=self.x[:3].clone()
        generator=torch.Generator().manual_seed(9)
        x[:,3:7]=torch.randn(3,4,generator=generator)
        x[:,3:7]/=x[:,3:7].norm(dim=-1,keepdim=True)
        x[:,7:30]=torch.randn(3,23,generator=generator)*.15
        expected=[]
        for row in x.numpy():
            data.qpos[:3]=row[:3];data.qpos[3:7]=row[[6,3,4,5]]
            for i,name in enumerate(self.fk.names):
                jid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,name)
                self.assertGreaterEqual(jid,0)
                data.qpos[model.jnt_qposadr[jid]]=row[7+i]
            mujoco.mj_forward(model,data)
            ids=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,n) for n in self.fk.links]
            self.assertTrue(all(i>=0 for i in ids))
            expected.append(data.xpos[ids].copy())
        np.testing.assert_allclose(self.fk(x).numpy(),expected,atol=2e-5)

    def test_batch_fetch_matches_reference(self):
        x=self.x.numpy().copy()
        rng=np.random.default_rng(12)
        q=rng.normal(size=(54,4)).astype(np.float32)
        q/=np.linalg.norm(q,axis=-1,keepdims=True)
        x[:,3:7]=q
        # Include static-root windows and nontrivial quaternion signs.
        x[27:,:2]=0
        a=dict(states=x,fps=50.,representation='g1-paper99',offsets=[0,27,54],split=[0,0])
        data=PaperWindows(a,0)
        for index,actual in zip([1,0],data.__getitems__([1,0])):
            for got,expected in zip(actual,data[index]):
                np.testing.assert_allclose(got.numpy(),expected,atol=2e-6)
        from torch.utils.data import DataLoader, Subset
        for batch in DataLoader(Subset(data,[1,0]),batch_size=2):
            self.assertEqual(batch[0].shape,(2,2,99))

    def test_training_backward(self):
        model=MotionDenoiser(dim=99,width=32,layers=1,heads=4)
        batch=(self.x[:2][None],self.x[2:27][None],torch.tensor([[1.,0.]]))
        total,*_=objective(model,batch,torch.zeros(99),torch.ones(99),schedule(10,'cpu'),.1,.01,
                          geometry=PaperMotionLoss('cpu'),geometry_weights={'fk':1.,'penetration':10.})
        total.backward()
        self.assertTrue(torch.isfinite(total))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

if __name__=='__main__':unittest.main()
