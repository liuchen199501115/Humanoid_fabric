from pathlib import Path
import torch,numpy as np,json
from scipy.spatial.transform import Rotation as R
from blended_spline import BlendedSplineReference
from fabrics_sim.reference.interpolation import MotionInterpolator
p=Path(__file__).parent;z=np.load(p/'fixed_clock120_body10_overdue.control_reference.npz');q=torch.tensor(z['source_q']);curves={s:BlendedSplineReference(MotionInterpolator(q),q,stride=s,start_frame=130,ramp_frames=20) for s in [2,6,10]};rows=[]
for stride in [6,10]:
 pose=[];joint=[];acc_old=[];acc_new=[]
 for f in np.arange(780.,980.,.48):
  a=[x.numpy().ravel() for x in curves[2].kinematics(f,240.)];b=[x.numpy().ravel() for x in curves[stride].kinematics(f,240.)]
  pose.append(np.linalg.norm((R.from_quat(a[0][3:7])*R.from_quat(b[0][3:7]).inv()).as_rotvec()));joint.append(np.sqrt(np.mean((a[0][7:]-b[0][7:])**2)));acc_old.append(a[2][3:6]);acc_new.append(b[2][3:6])
 rows.append(dict(stride=stride,maximum_root_orientation_change=float(max(pose)),maximum_joint_rmse_change=float(max(joint)),old_peak_angular_acceleration=float(np.abs(acc_old).max()),new_peak_angular_acceleration=float(np.abs(acc_new).max())))
(p/'wbc_tube_smoothing_audit.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows,indent=2))
