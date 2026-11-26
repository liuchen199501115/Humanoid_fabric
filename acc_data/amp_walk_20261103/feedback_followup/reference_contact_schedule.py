"""Candidate prediction schedule using prepared native sole arrival times.

Only future touchdown times change. Source poses and actual landing acceptance
are not changed. Existing support is retained if delayed arrival would create
an unsupported interval; a schedule without a later arrival is rejected.
"""
import numpy as np

def delayed_touchdowns(modes,clearance,start_frame,threshold=.002):
 modes=np.asarray(modes,dtype=int);clearance=np.asarray(clearance,dtype=float)
 if modes.ndim!=1 or clearance.shape!=(len(modes),2) or not np.isfinite(clearance).all():raise ValueError('Invalid contact schedule inputs')
 result=modes.copy();changes=[]
 for side in [0,1]:
  bit=1<<side;active=(modes&bit)!=0
  starts=np.flatnonzero(active & ~np.r_[active[0],active[:-1]])
  for onset in starts:
   if onset<start_frame:continue
   end=onset+1
   while end<len(modes) and active[end]:end+=1
   candidates=np.flatnonzero(clearance[onset:end,side]<=threshold)
   if not len(candidates):raise ValueError(f'No ground arrival for foot {side} in source stance {onset}:{end}')
   arrival=int(onset)+int(candidates[0]);result[onset:arrival]&=~bit
   if arrival!=onset:changes.append(dict(side=side,source_onset=int(onset),predicted_onset=arrival))
 for frame in range(max(1,int(start_frame)),len(result)):
  if result[frame]!=0:continue
  later=np.flatnonzero(result[frame+1:]!=0)
  if not len(later):raise ValueError('Delayed touchdown leaves an unsupported terminal interval')
  result[frame]=result[frame-1]
 return result,changes

def native_reference_clearance(planner,rate):
 from scipy.spatial.transform import Rotation
 result=[]
 for frame in range(len(planner.modes)):
  q,v,_=[a.numpy().ravel() for a in planner.reference.kinematics(float(frame),rate)]
  state=planner.model.evaluate(q,v);heights=[]
  for side in [0,1]:
   points,radii=planner.geometry[side];foot=state['feet'][side]
   heights.append(foot[2]+np.min((points@Rotation.from_quat(foot[3:]).as_matrix().T)[:,2]-radii))
  result.append(heights)
 return np.asarray(result)
