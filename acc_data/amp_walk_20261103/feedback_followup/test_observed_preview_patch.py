import unittest,numpy as np
from observed_preview_patch import observed_native_patch_indices
class ObservedPatchTests(unittest.TestCase):
 def setUp(self):
  self.poses=np.tile([0.,0.,.035,0.,0.,0.,1.],(2,1))
  self.local=np.array([[-.05,-.025,-.03],[-.05,.025,-.03],[.12,-.025,-.03],[.12,.025,-.03]])
  self.geometry=[(self.local,np.full(4,.005))]*2
  self.points=np.vstack([np.c_[self.local[:,:2],np.zeros(4)]]*2)
  self.normals=np.tile([0.,0.,1.],(8,1));self.distances=np.zeros(8)
 def get(self,forces):return observed_native_patch_indices(self.poses,self.geometry,(np.asarray(forces),self.points,self.normals,self.distances,[4,4],[0,4]),2)
 def test_geometrically_near_unloaded_corners_not_in_initial_patch(self):
  self.assertEqual(self.get([0,0,0,0,30,0,20,0]),{1:[0,2]})
 def test_loaded_flat_patch_contains_all_four_points(self):
  self.assertEqual(self.get([0,0,0,0,30,10,20,10]),{1:[0,1,2,3]})
 def test_no_measured_load_does_not_create_support(self):
  self.assertEqual(self.get(np.zeros(8)),{1:[]})
 def test_unmatched_loaded_point_is_rejected(self):
  self.points[4,0]+=.02
  with self.assertRaises(ValueError):self.get([0,0,0,0,30,0,0,0])
if __name__=='__main__':unittest.main()
