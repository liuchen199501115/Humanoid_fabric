"""Truncate lookahead at the moving source endpoint without adding a stop."""
import numpy as np

def remaining_intervals(frame,rate,last_frame,intervals):
 intervals=np.asarray(intervals,dtype=float)
 if rate<=0 or frame<0 or frame>=last_frame or np.any(intervals<=0):raise ValueError('Positive intervals and remaining source coverage required')
 remaining=(last_frame-frame)/rate
 offsets=np.r_[0.,np.cumsum(intervals)]
 if remaining>=offsets[-1]:return intervals.copy()
 interior=offsets[(offsets>0)&(offsets<remaining-1e-12)]
 return np.diff(np.r_[0.,interior,remaining])
