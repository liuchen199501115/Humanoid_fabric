"""Remove a predicted unloaded, separating point from a future contact patch."""
import numpy as np
from scipy.spatial.transform import Rotation
from preview_manifold import integrate

def released_indices(indices,forces,heights):
 indices=np.asarray(indices,dtype=int);forces=np.asarray(forces);heights=np.asarray(heights)
 if len(indices)<2 or np.max(forces[:,2],initial=0.)<20.:return set()
 return set(indices[(forces[:,2]<.5)&(heights>.00005)].tolist())

def release_future_points(planner,result,rate):
 changes=[]
 for stage,sides in result.get('point_forces',{}).items():
  if stage==0:continue
  rq,rv,_=[x.numpy().ravel() for x in planner.reference.kinematics(float(result['frames'][stage]),rate)]
  state=planner.model.evaluate(integrate(rq,result['states'][stage,:35]),rv+result['states'][stage,35:])
  for side,block in sides.items():
   foot=state['feet'][side];R=Rotation.from_quat(foot[3:]).as_matrix();points,radii=planner.geometry[side]
   heights=foot[2]+(points@R.T)[:,2]-radii;indices=block['indices']
   remove=released_indices(indices,block['forces'],heights[indices])
   if remove:
    planner._fixed_patches[stage][side]-=remove
    changes.append(dict(stage=int(stage),side=int(side),released=sorted(remove)))
 return changes
