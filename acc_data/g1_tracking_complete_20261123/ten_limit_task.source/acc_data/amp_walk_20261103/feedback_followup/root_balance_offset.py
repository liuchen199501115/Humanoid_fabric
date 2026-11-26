"""Reference-only root displacement while preserving both world foot poses."""
import numpy as np
from scipy.spatial.transform import Rotation
from numpy_leg_kinematics import LegKinematics

def root_balance_offset(trajectory,joint_names,urdf,*,fps=200.,offset=(0,.1,-.025),start=3.35,rise=.4,hold_end=4.25,fall=.4):
 model=LegKinematics(urdf,joint_names);result=trajectory.copy();records=[]
 def smooth(x):x=np.clip(x,0,1);return x**3*(10-15*x+6*x*x)
 for k,original in enumerate(trajectory):
  t=k/fps;weight=smooth((t-start)/rise)*(1-smooth((t-hold_end)/fall))
  if weight<1e-12:continue
  q=original.astype(float).copy();q[:3]+=weight*np.asarray(offset);peakp=peakr=0.
  for side in range(2):
   target,R,_=model.evaluate(original,side);sl=slice(7+6*side,13+6*side)
   for iteration in range(80):
    pos,actual,J=model.evaluate(q,side);pe=target-pos;re=Rotation.from_matrix(R@actual.T).as_rotvec()
    if np.linalg.norm(pe)<1e-6 and np.linalg.norm(re)<1e-5:break
    error=np.r_[pe,re];delta=J.T@np.linalg.solve(J@J.T+1e-7*np.eye(6),error);delta*=min(1,.06/max(abs(delta).max(),1e-12));q[sl]=np.clip(q[sl]+delta,model.lower[6*side:6*side+6],model.upper[6*side:6*side+6])
   peakp=max(peakp,float(np.linalg.norm(pe)));peakr=max(peakr,float(np.linalg.norm(re)))
  result[k]=q
  records.append(dict(frame=k,position_error=peakp,rotation_error=peakr))
 report=dict(offset_m=list(offset),start_s=start,rise_s=rise,hold_end_s=hold_end,fall_s=fall,
   peak_foot_position_residual=max((r['position_error'] for r in records),default=0.),peak_foot_rotation_residual=max((r['rotation_error'] for r in records),default=0.),
   peak_joint_correction_rmse=float(np.mean((result[:,7:]-trajectory[:,7:])**2,axis=1).max()**.5),
   worst_frames=sorted(records,key=lambda r:r['position_error'],reverse=True)[:3],physical_tracking_certified=False)
 return result,report
