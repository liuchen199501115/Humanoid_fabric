"""Flatten landing while lowering pelvis enough to preserve leg reachability."""
from pathlib import Path
import json,hashlib
import numpy as np,torch
from scipy.spatial.transform import Rotation
from fabrics_sim.reference.interpolation import MotionInterpolator
from numpy_leg_kinematics import LegKinematics
from blended_spline import BlendedSplineReference
p=Path(__file__).parent;z=np.load(p/'fixed_smooth_early.control_reference.npz');trajectory=z['control_q'];source=z['source_q'];model=LegKinematics('src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf',np.load(p/'wrench_fixed_ten.npz')['final_joint_order']);plan=json.load(open(p/'wrench_fixed_ten.steps_plan.json'));points=np.array(plan['collision_points'][0]);radii=np.array(plan['collision_radii'][0])
def smooth(x):x=np.clip(x,0,1);return x**3*(10-15*x+6*x*x)
for height in [.015,.025]:
 gain=.6
 name='early_flat_tail'+str(int(1000*height));out=trajectory.copy();records=[]
 for f in range(135,290):
  original=trajectory[f].astype(float);q=original.copy();w=smooth((f-135)/25)*(1-smooth((f-260)/29));q[2]-=.025*w
  for side in range(2):
   pos,R,_=model.evaluate(original,side);target=pos.copy();targetR=R
   if side==0:
    flat=Rotation.from_euler('z',np.arctan2(R[1,0],R[0,0])).as_matrix();delta=Rotation.from_matrix(flat@R.T).as_rotvec();targetR=Rotation.from_rotvec(gain*w*delta).as_matrix()@R
    target[2]+=np.min((points@R.T)[:,2]-radii)-np.min((points@targetR.T)[:,2]-radii)
    target[2]+=height*smooth((f-140)/15)*(1-smooth((f-164)/14))
   rows=slice(7+6*side,13+6*side);bounds=slice(6*side,6*side+6)
   for iteration in range(150):
    actual,actualR,J=model.evaluate(q,side);pe=target-actual;re=Rotation.from_matrix(targetR@actualR.T).as_rotvec()
    if np.linalg.norm(pe)<1e-7 and np.linalg.norm(re)<1e-6:break
    change=J.T@np.linalg.solve(J@J.T+1e-7*np.eye(6),np.r_[pe,re]);change*=min(1,.05/max(abs(change).max(),1e-12));q[rows]=np.clip(q[rows]+change,model.lower[bounds],model.upper[bounds])
   records.append([f,side,np.linalg.norm(pe),np.linalg.norm(re)])
  out[f]=q
 records=np.array(records);tensor=torch.tensor(out);sampler=BlendedSplineReference(MotionInterpolator(tensor),tensor,stride=6,start_frame=130,ramp_frames=20)
 dense=np.array([sampler.kinematics(f,200)[0].numpy()[0] for f in np.arange(130,300.01,.1)]);margin=float(np.minimum(dense[:,7:19]-model.lower,model.upper-dense[:,7:19]).min());rmse=float(np.sqrt(np.mean((out[:,7:]-source[:,7:])**2,axis=1))[130:300].max())
 report=dict(gain=gain,peak_position_residual=float(records[:,2].max()),peak_orientation_residual=float(records[:,3].max()),peak_source_joint_rmse=rmse,minimum_dense_joint_limit_margin=margin,initial_qva_preserved=True,physical_tracking_certified=False)
 report['preflight_passed']=bool(records[:,2].max()<.002 and records[:,3].max()<.01 and rmse<.19 and margin>=0)
 np.savez_compressed(p/(name+'.npz'),control_q=out,fps=200.,baseline_sha256=hashlib.sha256(trajectory.tobytes()).hexdigest(),source_sha256=hashlib.sha256(source.tobytes()).hexdigest());(p/(name+'.json')).write_text(json.dumps(report,indent=2)+'\n');print(name,report,flush=True)
