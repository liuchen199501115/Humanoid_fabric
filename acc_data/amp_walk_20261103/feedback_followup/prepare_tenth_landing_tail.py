from pathlib import Path
import hashlib,json
import numpy as np,torch
from scipy.spatial.transform import Rotation
from fabrics_sim.reference.interpolation import MotionInterpolator
from numpy_leg_kinematics import LegKinematics
from blended_spline import BlendedSplineReference
p=Path(__file__).parent
with np.load(p/'eight_smooth6_tube2.control_reference.npz') as data:control=data['control_q'];source=data['source_q'];fps=float(data['fps'])
names=np.load(p/'eight_smooth6_tube2.failure.npz')['joint_names'];model=LegKinematics('src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf',names)
def smooth(x):x=np.clip(x,0,1);return x**3*(10-15*x+6*x*x)
for key in ['004']:
 trajectory=np.load(p/('balance_y'+key+'.npz'))['control_q'];tensor=torch.tensor(trajectory);base=BlendedSplineReference(MotionInterpolator(tensor),tensor,stride=6,start_frame=564,ramp_frames=16)
 anchor_q=base.kinematics(780,200)[0].numpy()[0];anchor,Ranchor,_=model.evaluate(anchor_q,0)
 for flatten,extra_height in [(.5,.05)]:
  name='tenth_tail'+str(int(100*extra_height))+'_y'+key;frames=np.arange(760,871,dtype=float);correction=[];records=[]
  for f in frames:
   q=base.kinematics(f,200)[0].numpy()[0].astype(float);initial=q.copy();err=[]
   for side in [0,1]:
    pos,R,_=model.evaluate(q,side)
    if side==0:
     w=smooth((f-770)/15)*(1-smooth((f-825)/45));target=pos+w*(anchor-pos);targetR=Rotation.from_rotvec(w*Rotation.from_matrix(Ranchor@R.T).as_rotvec()).as_matrix()@R
    else:
     w=smooth((f-760)/20)*(1-smooth((f-809)/21));angles=Rotation.from_matrix(R).as_euler('xyz');flat=Rotation.from_euler('z',angles[2]).as_matrix();target=pos.copy();target[2]+=extra_height*smooth((f-765)/15)*(1-smooth((f-809)/20));targetR=Rotation.from_rotvec(w*flatten*Rotation.from_matrix(flat@R.T).as_rotvec()).as_matrix()@R
    sl=slice(7+6*side,13+6*side)
    for iteration in range(150):
     actual,actualR,J=model.evaluate(q,side);pe=target-actual;re=Rotation.from_matrix(targetR@actualR.T).as_rotvec()
     if np.linalg.norm(pe)<1e-7 and np.linalg.norm(re)<1e-6:break
     error=np.r_[pe,re];delta=J.T@np.linalg.solve(J@J.T+1e-7*np.eye(6),error);delta*=min(1,.05/max(abs(delta).max(),1e-12));q[sl]=np.clip(q[sl]+delta,model.lower[6*side:6*side+6],model.upper[6*side:6*side+6])
    err.extend([float(np.linalg.norm(pe)),float(np.linalg.norm(re))])
   correction.append(q[7:]-initial[7:]);records.append([f,*err,np.mean((q[7:]-source[int(f),7:])**2)**.5])
  correction=np.array(correction);correction[[0,-1]]=0;records=np.array(records)
  report=dict(peak_foot_position_residual=float(records[:,[1,3]].max()),peak_foot_rotation_residual=float(records[:,[2,4]].max()),peak_source_joint_reference_rmse=float(records[:,-1].max()),flatten_fraction=flatten,extra_height_m=extra_height,physical_tracking_certified=False)
  report['preflight_passed']=report['peak_foot_position_residual']<.002 and report['peak_foot_rotation_residual']<.01 and report['peak_source_joint_reference_rmse']<.19
  np.savez_compressed(p/(name+'.npz'),frames=frames,correction=correction,baseline_sha256=hashlib.sha256(trajectory.tobytes()).hexdigest(),source_sha256=hashlib.sha256(source.tobytes()).hexdigest())
  (p/(name+'.json')).write_text(json.dumps(report,indent=2)+'\n');print(name,json.dumps(report),flush=True)
