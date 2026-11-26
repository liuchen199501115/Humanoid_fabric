"""Measured metadata for the actual prepared control poses, not template claims."""
import numpy as np
from scipy.spatial.transform import Rotation
from numpy_leg_kinematics import LegKinematics

def reference_pose_metadata(control,source,names,events):
 control=np.asarray(control);source=np.asarray(source)
 if control.shape!=source.shape or control.ndim!=2 or control.shape[1]!=36:raise ValueError('Matching full-body trajectories required')
 position=np.linalg.norm(control[:,:3]-source[:,:3],axis=1)
 angle=(Rotation.from_quat(control[:,3:7])*Rotation.from_quat(source[:,3:7]).inv()).magnitude()
 model=LegKinematics('src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf',names);errors=[]
 for event in events:
  f=event['end_frame'];side=1-event['support'];a,R,_=model.evaluate(control[f],side);b,S,_=model.evaluate(source[f],side)
  errors.append([float(np.linalg.norm(a-b)),float((Rotation.from_matrix(R@S.T)).magnitude())])
 peak=np.max(errors,axis=0) if errors else np.zeros(2)
 return dict(pelvis_preserved=bool(position.max()<2e-5 and angle.max()<2e-5),
             pelvis_peak_position_change_m=float(position.max()),pelvis_peak_orientation_change_rad=float(angle.max()),
             landing_endpoints_preserved=bool((peak<2e-5).all()),
             peak_landing_endpoint_position_change_m=float(peak[0]),peak_landing_endpoint_orientation_change_rad=float(peak[1]),
             preservation_scope='Prepared control poses at integer source frames, before runtime interpolation or optional overlays')
