"""Unpromoted candidate: resolve all 500 Hz controls until the next replan."""
from full_body_preview_collision_fine_release import FullBodyPreview as BasePreview

class FullBodyPreview(BasePreview):
 def prediction_intervals(self,frame,rate):
  from anchored_preview_grid import anchored_control_grid
  if not hasattr(self,'grid_origin'):self.grid_origin=frame
  return anchored_control_grid(frame,rate,self.modes,self.grid_origin,dt=self.dt,horizon=self.N)[1]
 def planned_mode(self,frame):
  import numpy as np
  return int(self.modes[int(np.clip(frame-self.contact_delay_frames+1e-8,0,len(self.modes)-1))])
