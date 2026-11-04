import unittest
import torch
from fabrics_sim.solver.com_height import com_height_constraint


class ComHeightTests(unittest.TestCase):
    def inputs(self):
        dtype=torch.float64
        J=torch.zeros(1,3,35,dtype=dtype);J[:,:,:3]=torch.eye(3,dtype=dtype)
        return dict(position=torch.tensor([[0.,0.,.63]],dtype=dtype), jacobian=J,
                    bias=torch.tensor([[0.,0.,.2]],dtype=dtype), velocity=torch.zeros(1,35,dtype=dtype),
                    reference_position=torch.tensor([[0.,0.,.65]],dtype=dtype),
                    reference_velocity=torch.zeros(1,3,dtype=dtype),
                    reference_acceleration=torch.tensor([[0.,0.,.3]],dtype=dtype))

    def test_moving_envelope_and_bias_in_physical_coordinates(self):
        data=self.inputs();data['velocity'][:,2]=-.1;data['reference_velocity'][:,2]=-.1
        C,d=com_height_constraint(**data)
        # At the floor with equal vertical speed, actual zdd must be at
        # least the reference acceleration, after adding kinematic bias.
        self.assertAlmostEqual(d.item(), -.1)
        self.assertEqual(float(C[0,0,2]), -1.)
        data['velocity'][:,2]=-.2
        _,d=com_height_constraint(**data)
        self.assertAlmostEqual(d.item(), -2.1)

    def test_closed_loop_minimum_acceleration_preserves_admissible_floor(self):
        data=self.inputs();data['bias'].zero_();data['reference_acceleration'].zero_()
        data['position'][:,2]=.66;data['velocity'][:,2]=-.1
        minimum_gap=1.
        for _ in range(2000):
            C,d=com_height_constraint(**data)
            acc=-d.item()  # Saturate the bound, the least upward acceleration.
            data['position'][:,2]+=.001*data['velocity'][:,2]+.5e-6*acc
            data['velocity'][:,2]+=.001*acc
            minimum_gap=min(minimum_gap,float(data['position'][0,2]-.63))
        self.assertGreaterEqual(minimum_gap,-1e-7)
        self.assertLess(float(abs(data['position'][0,2]-.63)),1e-6)

    def test_invalid_margin_rejected(self):
        for margin in (0.,-1.,float('nan')):
            with self.assertRaises(ValueError):com_height_constraint(**self.inputs(),margin=margin)


if __name__=='__main__':unittest.main()
