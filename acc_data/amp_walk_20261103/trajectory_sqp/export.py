from pathlib import Path
import json,numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation
p=Path(__file__).parent;d=np.load(p/'torque_candidate.npz');frames=d['frames'];q=d['q'].astype(float)
for k in range(1,len(q)):
 if q[k-1,3:7]@q[k,3:7]<0:q[k,3:7]*=-1
sp=CubicSpline(frames,q,bc_type=((1,(q[1]-q[0])/(frames[1]-frames[0])),(2,np.zeros(36))));out=sp(np.arange(2000));out[:,3:7]/=np.linalg.norm(out[:,3:7],axis=1,keepdims=True)
np.savez_compressed(p/'control.npz',q=out.astype(np.float32),source_frames=2000,optimized_frames=frames)
v=sp(np.arange(2000),1)*200;a=sp(np.arange(2000),2)*200**2
report=dict(source_frames=2000,optimized_last_frame=int(frames[-1]),interpolation='C2 cubic; final source frame extrapolated by one 5 ms interval',peak_joint_speed=float(abs(v[:,7:]).max()),peak_joint_acceleration=float(abs(a[:,7:]).max()),initial_pose_max_difference=float(abs(out[0]-d['source'][0]).max()),full_tracking_passed=False)
(p/'export.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
