"""Map a reference world wrench to the currently measured contact coordinates."""
import numpy as np

def shift_wrench(wrench,source_point,target_point,target_axes):
 force=np.asarray(wrench[:3]);moment=np.asarray(wrench[3:])+np.cross(np.asarray(source_point)-np.asarray(target_point),force)
 return np.r_[np.asarray(target_axes).T@force,np.asarray(target_axes).T@moment]

def interpolate_wrenches(frame,wrench,points):
 low=min(int(frame),len(wrench)-1);high=min(low+1,len(wrench)-1);u=min(1.,max(0.,frame-low))
 force=(1-u)*wrench[low,:,:3]+u*wrench[high,:,:3]
 origin_moment=(1-u)*(wrench[low,:,3:]+np.cross(points[low],wrench[low,:,:3]))+u*(wrench[high,:,3:]+np.cross(points[high],wrench[high,:,:3]))
 point=(1-u)*points[low]+u*points[high]
 return np.c_[force,origin_moment-np.cross(point,force)],point
