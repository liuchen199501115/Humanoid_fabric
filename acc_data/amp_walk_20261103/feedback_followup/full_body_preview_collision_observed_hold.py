"""Offline candidate: do not invent support-point gains in the next 40 ms."""
from full_body_preview_collision_fine_release import FullBodyPreview as BasePreview
class FullBodyPreview(BasePreview):
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs);self.initial_patch_hold_seconds=.04
