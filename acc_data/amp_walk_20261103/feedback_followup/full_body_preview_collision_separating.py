"""Offline candidate: first-control damping and observed point separation."""
from full_body_preview_collision_first_damped import FullBodyPreview as BasePreview
class FullBodyPreview(BasePreview):
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs);self.separation_aware_patches=True
