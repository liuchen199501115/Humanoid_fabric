"""Audit torso angular feedback against the saved URDF Jacobian and live history."""
from pathlib import Path
import json,numpy as np
from scipy.spatial.transform import Rotation
p=Path(__file__).parent;d=np.load(p/'fixed_clock120_body10_turn_recovery40.npz');snapshot=np.load(p/'fixed_clock120_body10_turn_recovery40_comfirst.snapshot_3.800.npz')
def body(q,v,a=None):
 root=Rotation.from_quat(q[3:7]).as_matrix();rz=Rotation.from_euler('z',q[19]).as_matrix();rx=Rotation.from_euler('x',q[20]).as_matrix();rot=root@rz@rx@Rotation.from_euler('y',q[21]).as_matrix();axes=[root[:,2],(root@rz)[:,0],(root@rz@rx)[:,1]];w=v[3:6].copy();alpha=np.zeros(3) if a is None else a[3:6].copy()
 for j,axis in enumerate(axes):
  if a is not None:alpha+=axis*a[18+j]+np.cross(w,axis)*v[18+j]
  w+=axis*v[18+j]
 return rot,w,alpha
_,velocity,_=body(snapshot['q'][0],snapshot['v'][0]);error=np.linalg.norm(velocity-snapshot['task_body_orientation_A'][0]@snapshot['v'][0]);assert error<2e-5,error
options=json.loads(str(d['controller_options']));feedback_scale=float(options.get('motion_feedback_scale',1.));frames=d['history_frame'].reshape(-1);samples=[]
for f in [870,880,890,900,910,920,930,938]:
 i=abs(frames-f).argmin();rot,w,_=body(d['history_q'][i,0],d['history_qd'][i,0]);rr,rw,aa=body(d['history_whole_body_reference_q'][i].reshape(-1,36)[0],d['history_whole_body_reference_v'][i].reshape(-1,35)[0],d['history_whole_body_reference_a'][i].reshape(-1,35)[0]);e=Rotation.from_matrix(rot@rr.T).as_rotvec();feedback=-feedback_scale*np.array([50,50,25])*e-np.sqrt(feedback_scale)*np.array([14,14,10])*(w-rw);samples.append(dict(source_frame=float(frames[i]),torso_orientation_error=e.tolist(),reference_angular_acceleration=aa.tolist(),unclipped_feedback=feedback.tolist(),baseline_requested_acceleration=(aa+np.clip(feedback,[-25,-25,-20],[25,25,20])).tolist(),feedback75_requested_acceleration=(aa+np.clip(feedback,-75,75)).tolist()))
out=dict(urdf_angular_velocity_crosscheck_error=float(error),baseline_feedback_limits=[25,25,20],experimental_feedback_limit=75,samples=samples,physical_tracking_certified=False);(p/'body_feedback_saturation_audit.json').write_text(json.dumps(out,indent=2)+'\n');print('URDF angular velocity crosscheck error',error);print(samples[-3])
