"""Prepare a coherent, finite-support left landing orientation correction."""
from pathlib import Path
import json, hashlib
import numpy as np, torch
from scipy.spatial.transform import Rotation
from fabrics_sim.reference.interpolation import MotionInterpolator
from numpy_leg_kinematics import LegKinematics
from blended_spline import BlendedSplineReference
from joint_correction_overlay import JointCorrectionOverlay
p=Path(__file__).parent
z=np.load(p/'fixed_smooth_early.control_reference.npz'); trajectory=z['control_q'];source=z['source_q'];tensor=torch.tensor(trajectory)
model=LegKinematics('src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf',np.load(p/'wrench_fixed_ten.npz')['final_joint_order'])
base=BlendedSplineReference(MotionInterpolator(tensor),tensor,stride=6,start_frame=130,ramp_frames=20)
plan=json.load(open(p/'wrench_fixed_ten.steps_plan.json'));points=np.array(plan['collision_points'][0]);radii=np.array(plan['collision_radii'][0])
def smooth(x):x=np.clip(x,0,1);return x**3*(10-15*x+6*x*x)
for gain in [.6,1.]:
 name='early_flat'+str(int(gain*100));frames=np.arange(135,290,dtype=float);corrections=[];errors=[]
 for f in frames:
  q=base.kinematics(f,200.)[0].numpy()[0].astype(float);initial=q.copy();pos,R,_=model.evaluate(q,0)
  w=gain*smooth((f-135)/25)*(1-smooth((f-260)/29));yaw=np.arctan2(R[1,0],R[0,0]);flat=Rotation.from_euler('z',yaw).as_matrix();delta=Rotation.from_matrix(flat@R.T).as_rotvec();targetR=Rotation.from_rotvec(w*delta).as_matrix()@R
  target=pos.copy();target[2]+=np.min((points@R.T)[:,2]-radii)-np.min((points@targetR.T)[:,2]-radii)
  for iteration in range(150):
   actual,actualR,J=model.evaluate(q,0);pe=target-actual;re=Rotation.from_matrix(targetR@actualR.T).as_rotvec()
   if np.linalg.norm(pe)<1e-7 and np.linalg.norm(re)<1e-6:break
   change=J.T@np.linalg.solve(J@J.T+1e-7*np.eye(6),np.r_[pe,re]);change*=min(1,.05/max(abs(change).max(),1e-12));q[7:13]=np.clip(q[7:13]+change,model.lower[:6],model.upper[:6])
  corrections.append(q[7:]-initial[7:]);errors.append([np.linalg.norm(pe),np.linalg.norm(re),np.sqrt(np.mean((q[7:]-source[int(f),7:])**2))])
 corrections=np.array(corrections);corrections[[0,-1]]=0;errors=np.array(errors)
 target=p/(name+'.npz');np.savez_compressed(target,frames=frames,correction=corrections,baseline_sha256=hashlib.sha256(trajectory.tobytes()).hexdigest(),source_sha256=hashlib.sha256(source.tobytes()).hexdigest())
 overlay=JointCorrectionOverlay(base,tensor,torch.tensor(source),target);dense=np.array([overlay.kinematics(f,200)[0].numpy()[0] for f in np.arange(135,289.0001,.1)])
 margin=float(np.minimum(dense[:,7:19]-model.lower,model.upper-dense[:,7:19]).min())
 report=dict(gain=gain,peak_position_residual=float(errors[:,0].max()),peak_orientation_residual=float(errors[:,1].max()),peak_source_joint_rmse=float(errors[:,2].max()),minimum_dense_joint_limit_margin=margin,initial_qva_preserved=True,physical_tracking_certified=False)
 report['preflight_passed']=bool(errors[:,0].max()<.002 and errors[:,1].max()<.01 and errors[:,2].max()<.19 and margin>=0)
 (p/(name+'.json')).write_text(json.dumps(report,indent=2)+'\n');print(name,report,flush=True)
