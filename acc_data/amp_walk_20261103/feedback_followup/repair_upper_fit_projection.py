"""Jointly project COM and both feet; retain all original physical bounds."""
from pathlib import Path
import json
import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from numpy_g1_com import CenterOfMassModel
from numpy_leg_kinematics import LegKinematics
p=Path(__file__).parent
parent=np.load(p/'wbc_tenth_held_out33.npz');base=parent['control_q']
fit=np.load(p/'wbc_turn_upper_centroidal_fit.npz');out=fit['control_q'].copy();c=np.load(p/'wbc_turn_upper_centroidal_fit.centroidal.npz')
d=np.load(p/'wrench_fixed_ten.npz');limits=d['joint_position_limits'][0];urdf='src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf'
com=CenterOfMassModel(urdf,d['final_joint_order']);leg=LegKinematics(urdf,d['final_joint_order']);curve=BSpline(c['knots'],c['coefficients'],3);records=[]
for f in range(781,1100):
 q=out[f].astype(float);target=com.evaluate(base[f])+curve(f)[:3];feet=[leg.evaluate(base[f],side)[:2] for side in [0,1]]
 def residual(x,weighted=True):
  state=q.copy();state[:3]=x[:3];state[7:19]=x[3:];r=[com.evaluate(state)-target]
  for side,(position,rotation) in enumerate(feet):
   actual,R,_=leg.evaluate(state,side);r.extend([actual-position,Rotation.from_matrix(R@rotation.T).as_rotvec()])
  r=np.concatenate(r)
  if weighted:r*=np.r_[[3.]*3,[5.]*3,[1.]*3,[5.]*3,[1.]*3]
  return r
 x=np.r_[q[:3],q[7:19]];before=residual(x,False)
 if max(np.linalg.norm(before[6:9]),np.linalg.norm(before[12:15]))>.002 or max(np.linalg.norm(before[3:6]),np.linalg.norm(before[9:12]),np.linalg.norm(before[:3]))>.0001:
  lo=np.r_[q[:3]-.03,limits[:12,0]+1e-6];hi=np.r_[q[:3]+.03,limits[:12,1]-1e-6]
  answer=least_squares(residual,np.clip(x,lo,hi),bounds=(lo,hi),ftol=1e-11,xtol=1e-11,gtol=1e-11,max_nfev=100)
  x=answer.x;out[f,:3]=x[:3];out[f,7:19]=x[3:]
  after=residual(x,False);records.append(dict(frame=f,com=float(np.linalg.norm(after[:3])),position=float(max(np.linalg.norm(after[3:6]),np.linalg.norm(after[9:12]))),orientation=float(max(np.linalg.norm(after[6:9]),np.linalg.norm(after[12:15]))),evaluations=answer.nfev))
  print(records[-1],flush=True)
report=dict(repaired_frames=records,physical_tracking_certified=False,preflight_passed=False,requires_nonlinear_dynamic_audit=True)
np.savez_compressed(p/'wbc_upper_fit_projected.npz',control_q=out,fps=200.,baseline_sha256=parent['baseline_sha256'],source_sha256=parent['source_sha256'])
(p/'wbc_upper_fit_projected.json').write_text(json.dumps(report,indent=2)+'\n')
