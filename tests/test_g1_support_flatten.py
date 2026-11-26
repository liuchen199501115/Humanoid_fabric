import sys
import unittest
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'examples'))
from g1_support_flatten import flat_support_target


class SupportFlattenTests(unittest.TestCase):
    def test_sole_heading_is_distinct_from_contact_edge_frame(self):
        sole=Rotation.from_euler('z',.4);edge=Rotation.from_euler('z',1.2)
        tilt=np.array([.12,-.08,0.]);actual=Rotation.from_rotvec(tilt)*sole
        R=lambda x:torch.tensor(x.as_matrix(),dtype=torch.float64)[None]
        J=torch.zeros(1,3,35,dtype=torch.float64);J[0,:,3:6]=R(edge)[0].T
        v=torch.zeros(1,35,dtype=torch.float64);v[0,3:6]=torch.tensor([.3,-.1,.2])
        desired=flat_support_target(R(actual),R(sole),R(edge),J,v,gain_scale=4,feedback_limit=100)
        expected=edge.inv().apply(-200*tilt-28*v[0,3:6].numpy())
        np.testing.assert_allclose(desired[0],expected,atol=1e-10)

    def test_default_gain_component_cap_and_validation(self):
        R=torch.eye(3,dtype=torch.float64)[None];J=torch.zeros(1,3,35,dtype=torch.float64);J[0,:,:3]=torch.eye(3)
        v=torch.zeros(1,35,dtype=torch.float64);v[0,:3]=torch.tensor([10.,-10.,1.])
        torch.testing.assert_close(flat_support_target(R,R,R,J,v),torch.tensor([[-25.,25.,-14.]],dtype=torch.float64))
        for gain,limit in [(0,25),(1,-1),(float('nan'),25),(1,float('inf'))]:
            with self.assertRaises(ValueError):flat_support_target(R,R,R,J,v,gain_scale=gain,feedback_limit=limit)
