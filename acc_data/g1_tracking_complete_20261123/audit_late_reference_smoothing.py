"""Check later spline geometry and acceleration before closed-loop experiments."""
import sys,json
from pathlib import Path
import numpy as np,torch
from fabrics_sim.reference.interpolation import MotionInterpolator
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'acc_data/amp_walk_20261103/feedback_followup'))
from blended_spline import BlendedSplineReference
from late_tube_smoothing import transition_tube_spline
from numpy_leg_kinematics import LegKinematics
q=torch.from_numpy(np.load(OUT/'prepared_combined_root_y50.npz')['control_q'])
source=np.load(ROOT/'acc_data/g1_tracking_contacts_20261123/eight_verify.control_reference.npz')['source_q']
names=np.load(OUT/'landing9_load.npz')['final_joint_order'].tolist();model=LegKinematics(str(ROOT/'src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf'),names)
base=BlendedSplineReference(MotionInterpolator(q),q,stride=6,start_frame=130,ramp_frames=20);records=[]
for stride in [6,12,20]:
 sampler=base if stride==6 else transition_tube_spline(base,q,stride=stride,start_frame=740,ramp_frames=30)
 peak_joint_error=peak_pose_change=peak_foot_change=peak_acc=0.;landings={}
 for k in range(740,851):
  pose,vel,acc=[x.numpy()[0] for x in sampler.kinematics(float(k),240.,0.)];old=base(float(k)).numpy()[0]
  peak_joint_error=max(peak_joint_error,float(np.sqrt(np.mean((pose[7:]-source[k,7:])**2))));peak_pose_change=max(peak_pose_change,float(np.sqrt(np.mean((pose[7:]-old[7:])**2))));peak_acc=max(peak_acc,float(abs(acc[6:]).max()))
  if k>=802:
   error=float(np.linalg.norm(model.evaluate(pose,1)[0]-model.evaluate(source[k],1)[0]));peak_foot_change=max(peak_foot_change,error)
  if k in [762,780,790,802,810,820,840]:landings[k]=[float(np.linalg.norm(model.evaluate(pose,s)[0]-model.evaluate(source[k],s)[0])) for s in [0,1]]
 records.append(dict(stride=stride,peak_joint_rmse_rad=peak_joint_error,peak_joint_change_from_stride6=peak_pose_change,peak_joint_acceleration=peak_acc,peak_right_source_foot_error_after_frame802=peak_foot_change,foot_errors_at_frames=landings))
(OUT/'late_reference_smoothing_audit.json').write_text(json.dumps(records,indent=2));print(json.dumps(records,indent=2))
