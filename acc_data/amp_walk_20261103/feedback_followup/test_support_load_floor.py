import unittest,torch
from support_load_floor import support_load_floor
class SupportLoadFloorTests(unittest.TestCase):
    def test_only_observed_single_support_is_loaded(self):
        mode=torch.tensor([1,2,3,0,1]);force=torch.tensor([[120.,0.],[0.,120.],[120.,120.],[0.,0.],[0.,0.]])
        G,b=support_load_floor(mode,force,100.)
        wrench=torch.zeros(5,12);wrench[0,2]=99.;wrench[1,8]=101.
        residual=(G@wrench[:,:,None]).squeeze(-1)-b
        self.assertGreater(residual[0,0],0);self.assertLess(residual[1,0],0)
        torch.testing.assert_close(G[2:],torch.zeros_like(G[2:]));torch.testing.assert_close(b[2:],torch.zeros_like(b[2:]))
        self.assertEqual(torch.count_nonzero(G[0,:,6:]),0);self.assertEqual(torch.count_nonzero(G[1,:,:6]),0)
    def test_floor_is_vertical_and_does_not_add_friction_capacity(self):
        G,b=support_load_floor(torch.tensor([1]),torch.tensor([[120.,0.]]),100.)
        a=torch.randn(1,12)*100.;a[0,2]=100.;bigger=a.clone();bigger[0,0]+=10000.;bigger[0,3:]+=10000.
        torch.testing.assert_close(G@a[:,:,None],G@bigger[:,:,None])
    def test_transfer_only_protects_observed_incoming_contact(self):
        mode=torch.tensor([3,3,3,1,0]);forces=torch.tensor([[180.,200.],[180.,0.],[180.,200.],[180.,0.],[0.,0.]])
        intended=torch.tensor([2,2,3,2,2])
        G,b=support_load_floor(mode,forces,100.,preferred_mode=intended)
        self.assertEqual(float(G[0,0,8]),-1.);self.assertEqual(float(G[0,0,2]),0.)
        torch.testing.assert_close(G[1:3],torch.zeros_like(G[1:3]))
        self.assertEqual(float(G[3,0,2]),-1.);self.assertEqual(float(G[3,0,8]),0.)
        torch.testing.assert_close(G[4],torch.zeros_like(G[4]))
    def test_invalid_load_is_rejected(self):
        for value in [-1.,800.,float('inf'),float('nan')]:
            with self.assertRaises(ValueError):support_load_floor(torch.tensor([1]),torch.tensor([[100.,0.]]),value)
if __name__=='__main__':unittest.main()
