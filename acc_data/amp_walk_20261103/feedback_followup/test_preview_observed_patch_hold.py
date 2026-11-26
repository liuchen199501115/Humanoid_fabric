import unittest
from preview_observed_patch_hold import hold_observed_patches
class ObservedPatchHoldTests(unittest.TestCase):
 def test_airborne_toe_not_added_over_observed_heel(self):
  patches=[[{8},{9}]]+[[{10},{11}] for _ in range(3)]
  hold_observed_patches(patches,[0,1,2,3],[0,.002,.04,.064],{1:[9]},lambda f:2,.04)
  self.assertEqual([p[1] for p in patches],[{9},{9},{9},{11}])
  self.assertEqual([p[0] for p in patches],[{8},{10},{10},{10}])
 def test_original_patch_not_reintroduced_after_takeoff_and_relanding(self):
  patches=[[{8},{9}]]+[[{10},{11}] for _ in range(3)]
  hold_observed_patches(patches,[0,1,2,3],[0,.002,.004,.006],{1:[9]},lambda f:[2,2,1,2][f],.04)
  self.assertEqual([p[1] for p in patches],[{9},{9},{11},{11}])
 def test_disabling_hold_preserves_candidate_geometry(self):
  patches=[[{8},{9}],[{10},{11}]]
  hold_observed_patches(patches,[0,1],[0,.002],{1:[9]},lambda f:2,0.)
  self.assertEqual(patches[1][1],{11})
 def test_release_of_one_predicted_point_does_not_mutate_observation(self):
  initial={1:[9,11]};patches=[[set(),{9,11}],[set(),{11}]]
  hold_observed_patches(patches,[0,1],[0,.002],initial,lambda f:2,.04)
  patches[1][1].remove(11);self.assertEqual(initial,{1:[9,11]});self.assertEqual(patches[0][1],{9,11})
if __name__=='__main__':unittest.main()
