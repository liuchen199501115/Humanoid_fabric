from pathlib import Path
import hashlib,json
import numpy as np,torch
from scipy.spatial.transform import Rotation
from fabrics_sim.reference.interpolation import MotionInterpolator
from numpy_leg_kinematics import LegKinematics
from blended_spline import BlendedSplineReference
from joint_correction_overlay import JointCorrectionOverlay
p=Path(__file__).parent;trajectory=np.load(p/'balance_y004.npz')['control_q'];source=np.load(p/'eight_smooth6_tube2.control_reference.npz')['source_q'];names=np.load(p/'eight_smooth6_tube2.failure.npz')['joint_names'];model=LegKinematics('src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf',names);tensor=torch.tensor(trajectory);base=BlendedSplineReference(MotionInterpolator(tensor),tensor,stride=6,start_frame=564,ramp_frames=16);parent=p/'tenth_tail4_y004.npz';base=JointCorrectionOverlay(base,tensor,torch.tensor(source),parent)
def smooth(x):x=np.clip(x,0,1);return x**3*(10-15*x+6*x*x)
for lead,roll in [(.02,-.035),(.03,-.05)]:
 name='late_lateral'+str(int(100*lead));frames=np.arange(786,871,dtype=float);correction=[];records=[]
 for f in frames:
  q=base.kinematics(f,200)[0].numpy()[0].astype(float);initial=q.copy();pos,R,_=model.evaluate(q,1);w=smooth((f-786)/12)*(1-smooth((f-829)/41));target=pos+np.array([0.,-lead*w,.002*w]);targetR=Rotation.from_rotvec([roll*w,0,0]).as_matrix()@R
  for iteration in range(150):
   actual,actualR,J=model.evaluate(q,1);pe=target-actual;re=Rotation.from_matrix(targetR@actualR.T).as_rotvec()
   if np.linalg.norm(pe)<1e-7 and np.linalg.norm(re)<1e-6:break
   error=np.r_[pe,re];delta=J.T@np.linalg.solve(J@J.T+1e-7*np.eye(6),error);delta*=min(1,.05/max(abs(delta).max(),1e-12));q[13:19]=np.clip(q[13:19]+delta,model.lower[6:12],model.upper[6:12])
  correction.append(q[7:]-initial[7:]);records.append([f,np.linalg.norm(pe),np.linalg.norm(re),np.mean((q[7:]-source[int(f),7:])**2)**.5])
 correction=np.array(correction);correction[[0,-1]]=0;records=np.array(records);report=dict(peak_foot_position_residual=float(records[:,1].max()),peak_foot_rotation_residual=float(records[:,2].max()),peak_source_joint_reference_rmse=float(records[:,3].max()),start_frame=786,end_frame=870,lateral_reference_lead_m=-lead,world_roll_correction_rad=roll,physical_tracking_certified=False)
 report['preflight_passed']=report['peak_foot_position_residual']<.002 and report['peak_foot_rotation_residual']<.01 and report['peak_source_joint_reference_rmse']<.19
 np.savez_compressed(p/(name+'.npz'),frames=frames,correction=correction,baseline_sha256=hashlib.sha256(trajectory.tobytes()).hexdigest(),source_sha256=hashlib.sha256(source.tobytes()).hexdigest(),parent_overlay_sha256=hashlib.sha256(parent.read_bytes()).hexdigest());(p/(name+'.json')).write_text(json.dumps(report,indent=2)+'\n');print(name,report,flush=True)
