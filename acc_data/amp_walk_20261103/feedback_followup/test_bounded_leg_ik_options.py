import unittest
from pathlib import Path
import numpy as np
from bounded_leg_ik import solve_leg_pose
from numpy_leg_kinematics import LegKinematics
class BoundedIKOptionsTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  p=Path(__file__).parent;cls.q=np.load(p/'tenth_landing_forward15_right20.npz')['control_q'][800].astype(float);cls.model=LegKinematics('src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf',np.load(p/'wrench_fixed_ten.npz')['final_joint_order'])
 def test_default_preserves_previous_weight(self):
  pos,R,_=self.model.evaluate(self.q,1);target=pos+np.array([.001,0,0]);a=solve_leg_pose(self.model,self.q,1,target,R);b=solve_leg_pose(self.model,self.q,1,target,R,orientation_weight=.2);np.testing.assert_array_equal(a[0],b[0]);self.assertEqual(a[1:],b[1:])
 def test_position_weight_keeps_input_and_joint_bounds(self):
  original=self.q.copy();pos,R,_=self.model.evaluate(self.q,1);result,pe,re=solve_leg_pose(self.model,self.q,1,pos+[0,0,.001],R,orientation_weight=.05);np.testing.assert_array_equal(self.q,original);self.assertLess(pe,2e-6);self.assertTrue(np.all(result[13:19]>=self.model.lower[6:]));self.assertTrue(np.all(result[13:19]<=self.model.upper[6:]));np.testing.assert_array_equal(result[:13],original[:13])
 def test_invalid_orientation_weights_rejected(self):
  pos,R,_=self.model.evaluate(self.q,1)
  for weight in [0.,-1.,float('inf'),float('nan')]:
   with self.assertRaises(ValueError):solve_leg_pose(self.model,self.q,1,pos,R,orientation_weight=weight)
if __name__=='__main__':unittest.main()
