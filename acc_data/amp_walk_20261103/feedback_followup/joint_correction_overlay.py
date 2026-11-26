"""Finite-support C2 joint reference correction with analytic retiming."""
import hashlib
import numpy as np
import torch
from scipy.interpolate import make_interp_spline

class JointCorrectionOverlay:
 def __init__(self, original, trajectory, source, path):
  self.original=original
  with np.load(path) as data:
   for key,array in [('baseline_sha256',trajectory),('source_sha256',source)]:
    if data[key].item()!=hashlib.sha256(array.detach().cpu().numpy().tobytes()).hexdigest():
     raise ValueError('Joint correction input mismatch: '+key)
   self.frames=data['frames'].copy();correction=data['correction'].copy()
  if correction.shape!=(len(self.frames),29) or not np.isfinite(correction).all() or not np.all(np.diff(self.frames)>0):
   raise ValueError('Invalid correction samples')
  if np.max(np.abs(correction[[0,-1]]))>1e-12:raise ValueError('Correction endpoints must vanish')
  z=np.zeros(29);self.spline=make_interp_spline(self.frames,correction,k=5,bc_type=([(1,z),(2,z)],[(1,z),(2,z)]))
 def __call__(self,frame):return self.kinematics(frame,0.)[0]
 def kinematics(self,frame,frame_rate,frame_acceleration=0.):
  result=self.original.kinematics(frame,frame_rate,frame_acceleration)
  if not self.frames[0]<frame<self.frames[-1]:return result
  q,v,a=[x.clone() for x in result];d=[self.spline(frame,k) for k in range(3)]
  for value,offset,change in [(q,7,d[0]),(v,6,d[1]*frame_rate),(a,6,d[2]*frame_rate**2+d[1]*frame_acceleration)]:
   value[:,offset:]+=torch.as_tensor(change,device=value.device,dtype=value.dtype)[None]
  return q,v,a
