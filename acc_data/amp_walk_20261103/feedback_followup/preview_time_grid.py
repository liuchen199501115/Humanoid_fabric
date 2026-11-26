"""Resolve future source contact transitions and stop the horizon at the clip end."""
import numpy as np

def prediction_grid(frame,rate,modes,dt,horizon):
 if rate<=0 or dt<=0 or horizon<1:raise ValueError('Positive rate, step, and horizon required')
 end=float(len(modes)-1)
 if not 0<=frame<end:raise ValueError('Prediction requires remaining source frames')
 intervals=np.full(horizon,dt);intervals[:min(2,horizon)]=min(.01,dt)
 offsets=np.r_[0.,np.cumsum(intervals)]
 finish=min(end,frame+offsets[-1]*rate)
 base=frame+offsets*rate;base=base[base<finish-1e-9]
 # The fine first two intervals match the online planning timescale.
 # Later changes are represented exactly, including brief double support.
 changes=np.flatnonzero(np.asarray(modes)[1:]!=np.asarray(modes)[:-1])+1
 events=changes[(changes>frame+offsets[min(2,horizon)]*rate+1e-9)&(changes<finish-1e-9)]
 values=np.sort(np.r_[base,events,finish]);near=np.abs(values-np.round(values))<1e-8;values[near]=np.round(values[near])
 frames=values[np.r_[True,np.diff(values)>1e-8]]
 return frames,np.diff(frames)/rate
