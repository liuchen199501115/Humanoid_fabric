"""Blend into a smoother pose curve; all q/v/a come from that same curve."""
import numpy as np,torch
from scipy.interpolate import BSpline,CubicSpline,PchipInterpolator
class BlendedSplineReference:
 def __init__(self,original,trajectory,*,stride,start_frame,ramp_frames):
  if stride<2 or start_frame<0 or ramp_frames<=0:raise ValueError('Invalid smoothing parameters')
  self.original=original;self.trajectory=trajectory;self.start=start_frame;self.ramp=ramp_frames
  samples=trajectory.detach().cpu().double().numpy().copy()
  samples[:,3:7]/=np.linalg.norm(samples[:,3:7],axis=1,keepdims=True)
  for k in range(1,len(samples)):
   if samples[k-1,3:7]@samples[k,3:7]<0:samples[k,3:7]*=-1
  nodes=np.unique(np.r_[np.arange(0,len(samples),stride),len(samples)-1]);degree=min(3,len(nodes)-1)
  if degree!=3:raise ValueError('Smoothing requires at least four control poses')
  knots=np.r_[np.full(4,nodes[0]),nodes[2:-2],np.full(4,nodes[-1])]
  self.raw=(PchipInterpolator if original.method=='pchip' else CubicSpline)(np.arange(len(samples)),samples)
  # Clamped boundary knots have nonuniform coefficient times. In particular,
  # the final source interval may be shorter than stride. Sampling coefficients
  # at source node times spuriously brakes a moving endpoint. Greville times
  # reproduce linear motion exactly and equal the old times in the interior.
  greville=np.array([np.mean(knots[i+1:i+4]) for i in range(len(nodes))])
  self.smooth=BSpline(knots,self.raw(greville),3)
 def __call__(self,frame):return self.kinematics(frame,0.)[0]
 def kinematics(self,frame,frame_rate,frame_acceleration=0.):
  if frame<=self.start:return self.original.kinematics(frame,frame_rate,frame_acceleration)
  if not 0<=frame<=len(self.trajectory)-1:raise ValueError('Reference frame outside source coverage')
  u=min(1.,(frame-self.start)/self.ramp)
  w=u**3*(10-15*u+6*u*u);dw=30*u*u*(1-u)**2/self.ramp;ddw=60*u*(1-u)*(1-2*u)/self.ramp**2
  r=[self.raw(frame,k) for k in range(3)];s=[self.smooth(frame,k) for k in range(3)]
  value=r[0]+w*(s[0]-r[0]);first=r[1]+w*(s[1]-r[1])+dw*(s[0]-r[0]);second=r[2]+w*(s[2]-r[2])+2*dw*(s[1]-r[1])+ddw*(s[0]-r[0])
  velocity=first*frame_rate;acceleration=second*frame_rate**2+first*frame_acceleration
  quat=value[3:7];norm=np.linalg.norm(quat);quat=quat/norm;du=velocity[3:7];ddu=acceleration[3:7]
  nv=quat@du;qv=(du-quat*nv)/norm;na=(du@du+value[3:7]@ddu-nv*nv)/norm;qa=(ddu-quat*na-2*qv*nv)/norm
  def angular(d):return 2*(quat[3]*d[:3]-d[3]*quat[:3]+np.cross(quat[:3],d[:3]))
  value[3:7]=quat
  return tuple(torch.as_tensor(x,device=self.trajectory.device,dtype=self.trajectory.dtype)[None] for x in (value,np.r_[velocity[:3],angular(qv),velocity[7:]],np.r_[acceleration[:3],angular(qa),acceleration[7:]]))
