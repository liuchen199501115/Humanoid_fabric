"""Offline candidate: reduce proximal damping only near nonlinear feasibility."""
from full_body_preview_collision_fine_release import FullBodyPreview as BasePreview
class FullBodyPreview(BasePreview):
 def refine(self,*args,**kwargs):
  self.proximal_scale=1.
  try:return super().refine(*args,**kwargs)
  finally:self.proximal_scale=1.
 def solve(self,*args,**kwargs):
  result=super().solve(*args,**kwargs);result['proximal_scale']=self.proximal_scale
  return result
 def nonlinear_audit(self,*args,**kwargs):
  result=super().nonlinear_audit(*args,**kwargs)
  close=(result['nonlinear_collision_barrier']<.2 and result['nonlinear_support_gap']<.0002 and result['nonlinear_base_force_residual']<5.)
  self.proximal_scale=.2 if close else 1.
  return result
