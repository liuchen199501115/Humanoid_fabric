"""Offline candidate: exact rotational integration linearization and audit."""
from full_body_preview_collision_fine_release import FullBodyPreview as BasePreview
class FullBodyPreview(BasePreview):
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs);self.exact_rotation_transition=True
 def nonlinear_audit(self,*args,**kwargs):
  result=super().nonlinear_audit(*args,**kwargs)
  result['nonlinear_verified']=result['nonlinear_verified'] and result['nonlinear_integration_orientation']<1e-4
  return result
