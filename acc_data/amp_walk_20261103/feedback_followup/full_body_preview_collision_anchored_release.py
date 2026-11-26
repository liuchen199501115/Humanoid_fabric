"""Unpromoted stable-tail grid with 500 Hz near controls and contact events."""
import numpy as np
from full_body_preview_collision_fine_release import FullBodyPreview as BasePreview
from anchored_preview_grid import anchored_grid

class FullBodyPreview(BasePreview):
 def prediction_intervals(self,frame,rate):
  if not hasattr(self,'grid_origin'):self.grid_origin=frame
  frames,_=anchored_grid(frame,rate,self.modes,self.grid_origin,dt=self.dt,horizon=self.N-4)
  fine=frame+np.arange(1,5)*.002*rate
  changes=np.flatnonzero(self.modes[1:]!=self.modes[:-1])+1
  tick=.002*rate;events=self.grid_origin+np.ceil((changes-self.grid_origin)/tick-1e-10)*tick
  values=np.sort(np.r_[frames,fine[(fine<frames[-1]-1e-8)],events[(events>frame+1e-8)&(events<frames[-1]-1e-8)]])
  unique=values[np.r_[True,np.diff(values)>1e-8]]
  return np.diff(unique)/rate
 def planned_mode(self,frame):
  return int(self.modes[int(np.clip(frame-self.contact_delay_frames+1e-8,0,len(self.modes)-1))])
