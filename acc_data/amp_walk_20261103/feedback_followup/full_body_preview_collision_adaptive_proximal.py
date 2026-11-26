"""Offline candidate: reduce SQP proximal damping after three audited iterates."""
from full_body_preview_collision_fine_release import FullBodyPreview as BasePreview
class FullBodyPreview(BasePreview):
 def refine(self,*args,**kwargs):
  self._audited_iterations=0
  try:return super().refine(*args,**kwargs)
  finally:self.proximal_scale=1.
 def solve(self,*args,**kwargs):
  self.proximal_scale=1. if self._audited_iterations<3 else .1
  result=super().solve(*args,**kwargs);result['proximal_scale']=self.proximal_scale
  return result
 def nonlinear_audit(self,*args,**kwargs):
  result=super().nonlinear_audit(*args,**kwargs);self._audited_iterations+=1
  return result
