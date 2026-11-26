"""Execute the exact C2 curve used in the offline collocation experiment."""
import numpy as np
import torch
from scipy.interpolate import BSpline
class SplineReference:
 def __init__(self,path,device,dtype):
  with np.load(path) as data:
   self.spline=BSpline(data['knots'],data['coefficients'],int(data['degree']))
   self.wrenches=data['wrench'].copy();self.support_points=data['support_points'].copy() if 'support_points' in data else None
   self.last_frame=int(data['frames'][-1]);self.trajectory=torch.as_tensor(data['q'],device=device,dtype=dtype)
  self.device,self.dtype=device,dtype
 def wrench_at(self,frame):
  from planned_wrench import interpolate_wrenches
  if self.support_points is None:raise ValueError('Reference wrench has no physical support origin')
  return interpolate_wrenches(frame,self.wrenches,self.support_points)
 def __call__(self,frame):return self.kinematics(frame,0.)[0]
 def kinematics(self,frame,frame_rate,frame_acceleration=0.):
  if not 0<=frame<=self.last_frame:raise ValueError('Experimental spline does not cover this source frame')
  value=self.spline(frame);first=self.spline(frame,1);second=self.spline(frame,2)
  velocity=first*frame_rate;acceleration=second*frame_rate**2+first*frame_acceleration
  u=value[3:7];du=velocity[3:7];ddu=acceleration[3:7];norm=np.linalg.norm(u);quat=u/norm
  nv=quat@du;qv=(du-quat*nv)/norm;na=(du@du+u@ddu-nv*nv)/norm;qa=(ddu-quat*na-2*qv*nv)/norm
  def angular(d):return 2*(quat[3]*d[:3]-d[3]*quat[:3]+np.cross(quat[:3],d[:3]))
  value[3:7]=quat
  return tuple(torch.as_tensor(a,device=self.device,dtype=self.dtype)[None] for a in (value,np.r_[velocity[:3],angular(qv),velocity[7:]],np.r_[acceleration[:3],angular(qa),acceleration[7:]]))
