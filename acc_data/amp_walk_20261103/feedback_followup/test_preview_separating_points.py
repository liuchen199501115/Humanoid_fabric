import unittest
import numpy as np
from preview_separating_points import separating_points
class SeparatingPointTests(unittest.TestCase):
 def setUp(self):
  self.state=dict(feet=np.tile([0.,0.,0.,0.,0.,0.,1.],(2,1)),jacobian=np.zeros((2,6,35)))
  self.state['jacobian'][:,:6,:6]=np.eye(6)
  self.geometry=[(np.array([[-.1,0.,0.],[.1,0.,0.]]),np.zeros(2))]*2
 def test_separating_heel_released_while_loaded_toe_remains(self):
  v=np.zeros(35);v[2]=.015;v[4]=.15;initial={1:[0,1]}
  kept,rows=separating_points(self.state,self.geometry,v,initial)
  self.assertEqual(kept,{1:[1]});self.assertEqual(rows[0]['point'],0);self.assertAlmostEqual(rows[0]['normal_speed'],.03)
  self.assertEqual(initial,{1:[0,1]})
 def test_static_contacts_preserved(self):
  self.assertEqual(separating_points(self.state,self.geometry,np.zeros(35),{1:[0,1]}),({},[]))
 def test_no_stable_support_does_not_invent_one(self):
  v=np.zeros(35);v[2]=.03
  self.assertEqual(separating_points(self.state,self.geometry,v,{1:[0,1]}),({},[]))
 def test_last_observed_point_is_not_removed(self):
  v=np.zeros(35);v[2]=.03
  self.assertEqual(separating_points(self.state,self.geometry,v,{1:[0]}),({},[]))
if __name__=='__main__':unittest.main()
