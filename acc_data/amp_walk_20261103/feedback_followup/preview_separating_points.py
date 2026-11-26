"""Conservative future patch seed for observed points already separating.

Stage-zero measurements are not changed. A stable loaded point must remain;
the complete candidate still needs the unchanged force and geometry audits.
"""
import numpy as np
from scipy.spatial.transform import Rotation
from centroidal_pose_linearization import skew

def separating_points(state,geometry,velocity,initial,dt=.002,tolerance=2e-5):
 kept={};diagnostics=[]
 for side,indices in initial.items():
  side=int(side);indices=np.asarray(indices,dtype=int)
  if len(indices)<2:continue
  foot=state['feet'][side];R=Rotation.from_quat(foot[3:]).as_matrix();points,radii=geometry[side];J=state['jacobian'][side]
  offset=points[indices]@R.T-radii[indices,None]*np.array([0.,0.,1.])
  height=foot[2]+offset[:,2]
  speed=np.array([(J[:3]-skew(x)@J[3:])[2]@velocity for x in offset])
  departing=(height>=-5e-5)&(speed>2*tolerance/dt)
  stable=(np.abs(height)<=5e-5)&(np.abs(speed)<=tolerance/dt)
  if not departing.any() or not stable.any():continue
  kept[side]=indices[~departing].tolist()
  diagnostics.extend(dict(side=side,point=int(indices[i]),height=float(height[i]),normal_speed=float(speed[i])) for i in np.flatnonzero(departing))
 return kept,diagnostics
