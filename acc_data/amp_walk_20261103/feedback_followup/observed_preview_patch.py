"""Match genuinely loaded ground points to the unchanged native sole geometry."""
import numpy as np
from scipy.spatial.transform import Rotation

def observed_native_patch_indices(poses,geometry,contact_data,mode):
 poses=np.asarray(poses).reshape(2,7)
 forces,points,normals,distances,counts,starts=(np.asarray(x) for x in contact_data)
 forces=forces.ravel();distances=distances.ravel();counts=counts.ravel();starts=starts.ravel()
 result={}
 for side in [0,1]:
  if not int(mode)&(1<<side):continue
  start,count=int(starts[side]),int(counts[side]);sl=slice(start,start+count)
  loaded=(forces[sl]>.5)&(distances[sl]<=.001)&(np.abs(normals[sl,2])>.9)
  measured=points[sl][loaded]
  if not len(measured):result[side]=[];continue
  local,radii=geometry[side];R=Rotation.from_quat(poses[side,3:]).as_matrix()
  native=poses[side,:3]+np.asarray(local)@R.T-np.asarray(radii)[:,None]*np.array([0.,0.,1.])
  distance=np.linalg.norm(measured[:,None,:]-native[None,:,:],axis=2)
  nearest=distance.argmin(axis=1)
  if np.any(distance[np.arange(len(nearest)),nearest]>.003):
   raise ValueError('Loaded ground point does not match native sole geometry within 3 mm')
  result[side]=sorted(set(nearest.tolist()))
 return result
