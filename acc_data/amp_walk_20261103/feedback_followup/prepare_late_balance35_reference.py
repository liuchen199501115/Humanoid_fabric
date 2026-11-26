"""Project bounded COM paths into whole-body poses with native foot clearance retained."""
from pathlib import Path
import hashlib,json
import numpy as np,torch
from scipy.interpolate import CubicHermiteSpline
from scipy.spatial.transform import Rotation
from fabrics_sim.reference.interpolation import MotionInterpolator
from numpy_g1_com import CenterOfMassModel
from numpy_leg_kinematics import LegKinematics
from bounded_leg_ik import solve_leg_pose
from blended_spline import BlendedSplineReference
p=Path(__file__).parent;baseline=np.load(p/'fixed_smooth_early.control_reference.npz');source=baseline['source_q'];base=np.load(p/'seventh_clearance10.npz')['control_q'];names=np.load(p/'wrench_fixed_ten.npz')['final_joint_order'];urdf='src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf';com=CenterOfMassModel(urdf,names);model=LegKinematics(urdf,names);plan=json.load(open(p/'wrench_fixed_ten.steps_plan.json'));geometry=[(np.array(plan['collision_points'][s]),np.array(plan['collision_radii'][s])) for s in range(2)]
def smooth(x):x=np.clip(x,0,1);return x**3*(10-15*x+6*x*x)
for shift in [.035]:
 damping=.5
 name='late_balance'+str(int(shift*1000));out=base.copy();records=[]
 for f in range(450,len(base)):
  template=base[f].astype(float);q=template.copy();w=smooth((f-450)/65);original_com=com.evaluate(template);target_com=original_com[:2]+np.array([0.,-shift*w]);foot_targets=[]
  for side in range(2):
   pos,R,_=model.evaluate(template,side);points,radii=geometry[side];clear=pos[2]+np.min((points@R.T)[:,2]-radii);foot_targets.append((pos,R,clear))
  q[:2]+=1.25*(target_com-original_com[:2]);q[2]-=damping*np.linalg.norm(q[:2]-template[:2])
  for outer in range(4):
   foot_errors=[]
   for side in range(2):
    pos,R,clear=foot_targets[side];target=pos.copy();points,radii=geometry[side]
    for correction in range(4):
     q,pe,re=solve_leg_pose(model,q,side,target,R)
     actual,actualR,_=model.evaluate(q,side);actual_clear=actual[2]+np.min((points@actualR.T)[:,2]-radii)
     if abs(actual_clear-clear)<1e-6:break
     target[2]+=clear-actual_clear
    foot_errors.append((np.linalg.norm(actual[:2]-pos[:2]),abs(actual_clear-clear),re,np.linalg.norm(actual-pos)))
   actual_com=com.evaluate(q);ce=np.linalg.norm(target_com-actual_com[:2])
   if ce<1e-6 or outer==3:break
   q[:2]+=1.25*(target_com-actual_com[:2]);q[2]=template[2]-damping*np.linalg.norm(q[:2]-template[:2])
  limits=np.load(p/'wrench_fixed_ten.npz')['joint_position_limits'][0]
  q[7:]=np.clip(q[7:],limits[:,0]+1e-5,limits[:,1]-1e-5)
  out[f]=q;records.append([f,ce,*np.max(foot_errors,axis=0)])
 records=np.array(records);rmse=np.sqrt(np.mean((out[:,7:]-source[:,7:])**2,axis=1));root_delta=np.linalg.norm(out[:,:3]-source[:,:3],axis=1)
 tensor=torch.tensor(out);sampler=BlendedSplineReference(MotionInterpolator(tensor),tensor,stride=6,start_frame=130,ramp_frames=20);dense=np.array([sampler.kinematics(f,200)[0].numpy()[0] for f in np.arange(450,1999.01,.5)]);limits=np.load(p/'wrench_fixed_ten.npz')['joint_position_limits'][0];margin=float(np.minimum(dense[:,7:]-limits[:,0],limits[:,1]-dense[:,7:]).min())
 report=dict(root_lowering_ratio=damping,lateral_com_shift=shift,peak_com_projection_error=float(records[:,1].max()),peak_foot_horizontal_error=float(records[:,2].max()),peak_native_clearance_error=float(records[:,3].max()),peak_foot_orientation_adjustment=float(records[:,4].max()),peak_foot_link_displacement=float(records[:,5].max()),peak_source_joint_reference_rmse=float(rmse.max()),peak_root_position_delta=float(root_delta.max()),minimum_dense_joint_limit_margin=margin,initial_qva_preserved=True,physical_tracking_certified=False)
 report['preflight_passed']=bool(records[:,1].max()<.003 and records[:,2].max()<.002 and records[:,3].max()<.002 and records[:,4].max()<.1 and records[:,5].max()<.025 and rmse.max()<.19 and root_delta.max()<.19 and margin>=0)
 report['worst_projection_rows']=records[np.argsort(records[:,1])[-5:]].tolist()
 np.savez_compressed(p/(name+'.npz'),control_q=out,fps=200.,baseline_sha256=hashlib.sha256(baseline['control_q'].tobytes()).hexdigest(),source_sha256=hashlib.sha256(source.tobytes()).hexdigest());(p/(name+'.json')).write_text(json.dumps(report,indent=2)+'\n');print(name,report,flush=True)
