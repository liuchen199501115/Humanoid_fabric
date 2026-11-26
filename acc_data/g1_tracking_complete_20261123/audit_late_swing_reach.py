"""Local bounded IK reach diagnostic at recorded roots; not a global feasibility proof."""
import json,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'acc_data/amp_walk_20261103/feedback_followup'))
from numpy_leg_kinematics import LegKinematics
from bounded_leg_ik import solve_leg_pose
name='ten_anchor_support1_commax3_816';data=np.load(OUT/(name+'.npz'));frames=data['history_frame'].ravel()
legs=LegKinematics(str(ROOT/'src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf'),data['final_joint_order']);rows=[]
for frame in [842,850,858,866,874,878]:
 i=int(np.argmin(abs(frames-frame)));measured=data['history_q'][i,0].astype(float);ref=data['history_whole_body_reference_q'][i,0].astype(float);p,R,_=legs.evaluate(ref,0)
 for shift in [np.zeros(3),np.array([0.,0.,-.05]),np.array([.05,.05,0.]),np.array([.05,.05,-.05])]:
  state=measured.copy();state[:3]+=shift
  candidates=[]
  for seed in [measured[7:13],ref[7:13]]:
   state[7:13]=seed;out,ep,er=solve_leg_pose(legs,state,0,p,R,max_iterations=300)
   candidates.append((ep+.2*er,ep,er,out))
  _,ep,er,out=min(candidates,key=lambda x:x[0]);rows.append(dict(frame=float(frames[i]),root_shift=shift.tolist(),position_residual_m=ep,orientation_residual_rad=er,solved_knee=float(out[10]),measured_knee=float(measured[10]),reference_knee=float(ref[10])))
report=dict(scope=__doc__,trial=name,rows=rows)
(OUT/'late_swing_reach_audit.json').write_text(json.dumps(report,indent=2)+'\n')
for row in rows:print(row,flush=True)
