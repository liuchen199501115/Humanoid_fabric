"""Shift pelvis before the right support transition, preserving both foot poses."""
from pathlib import Path
import hashlib,json
import numpy as np,torch
from scipy.spatial.transform import Rotation
from fabrics_sim.reference.interpolation import MotionInterpolator
from numpy_leg_kinematics import LegKinematics
from blended_spline import BlendedSplineReference
from bounded_leg_ik import solve_leg_pose
p=Path(__file__).parent;baseline=np.load(p/'fixed_smooth_early.control_reference.npz');source=baseline['source_q'];initial=np.load(p/'early_flat_lower15.npz')['control_q'];model=LegKinematics('src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf',np.load(p/'wrench_fixed_ten.npz')['final_joint_order'])
def smooth(x):x=np.clip(x,0,1);return x**3*(10-15*x+6*x*x)
plan=json.load(open(p/'wrench_fixed_ten.steps_plan.json'))
for roll in [.06,.10]:
 shift=.03
 name='early_balance30_roll'+str(int(1000*roll));out=initial.copy();errors=[]
 for f in range(190,481):
  original=initial[f].astype(float);q=original.copy();w=smooth((f-190)/45)*(1-smooth((f-430)/50));q[1]-=shift*w;q[2]-=.2*shift*w
  for side in range(2):
   pos,R,_=model.evaluate(original,side)
   if side==0:
    targetR=Rotation.from_rotvec([roll*w,0,0]).as_matrix()@R
    points=np.array(plan['collision_points'][side]);radii=np.array(plan['collision_radii'][side])
    pos[2]+=np.min((points@R.T)[:,2]-radii)-np.min((points@targetR.T)[:,2]-radii)
    R=targetR
   if w:
    q,pe,re=solve_leg_pose(model,q,side,pos,R);errors.append([f,side,pe,re])
  out[f]=q
 errors=np.array(errors);tensor=torch.tensor(out);curve=BlendedSplineReference(MotionInterpolator(tensor),tensor,stride=6,start_frame=130,ramp_frames=20)
 dense=np.array([curve.kinematics(f,200)[0].numpy()[0] for f in np.arange(185,485.01,.1)]);margin=float(np.minimum(dense[:,7:19]-model.lower,model.upper-dense[:,7:19]).min());rmse=float(np.sqrt(np.mean((out[:,7:]-source[:,7:])**2,axis=1)).max())
 report=dict(pelvis_y_shift=-shift,pelvis_z_shift=-.2*shift,peak_position_residual=float(errors[:,2].max()),peak_orientation_residual=float(errors[:,3].max()),peak_source_joint_reference_rmse=rmse,minimum_dense_joint_limit_margin=margin,initial_qva_preserved=True,physical_tracking_certified=False)
 report['preflight_passed']=bool(errors[:,2].max()<.002 and errors[:,3].max()<.01 and rmse<.19 and margin>=0)
 np.savez_compressed(p/(name+'.npz'),control_q=out,fps=200.,baseline_sha256=hashlib.sha256(baseline['control_q'].tobytes()).hexdigest(),source_sha256=hashlib.sha256(source.tobytes()).hexdigest());(p/(name+'.json')).write_text(json.dumps(report,indent=2)+'\n');print(name,report,flush=True)
