import unittest
from types import SimpleNamespace
import numpy as np
from preview_contact_release import released_indices,release_future_points
from test_full_body_preview_audit import CartesianModel,StaticReference
class ContactReleaseTests(unittest.TestCase):
 def test_only_unloaded_lifted_point_is_released(self):
  self.assertEqual(released_indices([9,11],[[0.,0.,0.],[0.,0.,25.]],[.0002,0.]),{9})
 def test_loaded_point_is_retained(self):
  self.assertEqual(released_indices([9,11],[[0.,0.,30.],[0.,0.,25.]],[.0002,0.]),set())
 def test_grounded_or_penetrating_point_is_not_released(self):
  for gap in [0.,-.001]:self.assertEqual(released_indices([9,11],[[0.,0.,0.],[0.,0.,25.]],[gap,0.]),set())
 def test_last_point_or_unloaded_patch_is_not_removed(self):
  self.assertEqual(released_indices([9],[[0.,0.,0.]],[.001]),set())
  self.assertEqual(released_indices([9,11],[[0.,0.,0.],[0.,0.,0.]],[.001,0.]),set())
 def test_current_observed_patch_is_unchanged(self):
  geometry=[(np.array([[0.,0.,.0002],[0.,0.,0.]]),np.zeros(2))]*2
  planner=SimpleNamespace(reference=StaticReference(),model=CartesianModel(),geometry=geometry,_fixed_patches=[[{0,1},{0,1}],[{0,1},{0,1}]])
  block=dict(indices=[0,1],forces=[[0.,0.,0.],[0.,0.,25.]])
  result=dict(frames=np.array([0.,1.]),states=np.zeros((2,70)),point_forces={0:{1:block},1:{1:block}})
  changes=release_future_points(planner,result,1.)
  self.assertEqual(planner._fixed_patches[0][1],{0,1});self.assertEqual(planner._fixed_patches[1][1],{1});self.assertEqual(changes,[dict(stage=1,side=1,released=[0])])
if __name__=='__main__':unittest.main()
