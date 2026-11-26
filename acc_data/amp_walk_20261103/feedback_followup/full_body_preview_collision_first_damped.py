"""Offline candidate: apply the same SQP acceleration damping at stage zero."""
from full_body_preview_collision_recover_margin import FullBodyPreview as BasePreview
class FullBodyPreview(BasePreview):
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs);self.damp_first_control=True
