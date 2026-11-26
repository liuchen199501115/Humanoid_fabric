"""Offline candidate: exact rotation and feasible initial joint reserve."""
import numpy as np
from full_body_preview_collision_exact_rotation import FullBodyPreview as BasePreview
class FullBodyPreview(BasePreview):
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs);self.recover_initial_joint_reserve=True
 def nonlinear_audit(self,result,rate):
  audit=super().nonlinear_audit(result,rate);violation=0.
  for k,frame in enumerate(result['frames']):
   reference=self.reference.kinematics(float(frame),rate)[0].numpy().ravel()
   position=reference[7:]+result['states'][k,6:35]
   violation=max(violation,float(np.max(self.limits[:,0]-position)),float(np.max(position-self.limits[:,1])))
  audit['nonlinear_joint_position_limit_violation']=violation
  audit['nonlinear_verified']=audit['nonlinear_verified'] and violation<1e-6
  return audit
