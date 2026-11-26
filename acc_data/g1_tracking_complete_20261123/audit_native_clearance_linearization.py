"""Finite-difference check of native foot-bottom height constraint rows."""
import json,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'acc_data/amp_walk_20261103/feedback_followup'))
from numpy_leg_kinematics import LegKinematics
from root_rotation_variation import skew
names=np.load(OUT/'ten_verify.npz')['final_joint_order'].tolist()
legs=LegKinematics(str(ROOT/'src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf'),names)
q=np.load(OUT/'prepared_margin30_turn_com_y15.npz')['control_q']
plan=json.loads((OUT/'ten_verify.steps_plan.json').read_text());points=np.array(plan['collision_points'][1]);radii=np.array(plan['collision_radii'][1])
rng=np.random.default_rng(192);rows=[]
for k in [782,790,800]:
 pose=q[k].astype(float);foot,R,J=legs.evaluate(pose,1);A=np.zeros((6,35));A[:3,:3]=np.eye(3);A[:3,3:6]=-skew(foot-pose[:3]);A[3:,3:6]=np.eye(3);A[:,12:18]=J
 offsets=points@R.T;C=A[2]+offsets[:,1,None]*A[3]-offsets[:,0,None]*A[4]
 direction=rng.normal(size=35);eps=1e-5
 def heights(sign):
  qq=pose.copy();qq[:3]+=sign*eps*direction[:3];qq[3:7]=(Rotation.from_rotvec(sign*eps*direction[3:6])*Rotation.from_quat(pose[3:7])).as_quat();qq[7:]+=sign*eps*direction[6:]
  p,r,_=legs.evaluate(qq,1);return (points@r.T)[:,2]+p[2]-radii
 error=float(np.max(abs((heights(1)-heights(-1))/(2*eps)-C@direction)))
 assert error<1e-7,(k,error)
 rows.append(dict(source_frame=k,maximum_native_height_derivative_error=error))
report=dict(scope=__doc__,passed=True,cases=rows)
(OUT/'native_clearance_linearization_verification.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
