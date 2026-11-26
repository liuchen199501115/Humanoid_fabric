"""Bounded source-relative foot placement with unchanged original landing audits."""
from pathlib import Path
import json,hashlib
import numpy as np,torch
from scipy.spatial.transform import Rotation
from fabrics_sim.reference.interpolation import MotionInterpolator
from numpy_leg_kinematics import LegKinematics
from bounded_leg_ik import solve_leg_pose
from blended_spline import BlendedSplineReference
p=Path(__file__).parent;baseline=np.load(p/'fixed_smooth_early.control_reference.npz');source=baseline['source_q'];base=np.load(p/'sixth_plant_roll_y30.npz')['control_q'];d=np.load(p/'wrench_fixed_ten.npz');limits=d['joint_position_limits'][0];model=LegKinematics('src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf',d['final_joint_order'])
plan=json.load(open(p/'wrench_fixed_ten.steps_plan.json'));points=np.array(plan['collision_points'][1]);radii=np.array(plan['collision_radii'][1])
def smooth(x):x=np.clip(x,0,1);return x**3*(10-15*x+6*x*x)
for yshift in [-.02,-.03]:
 name='eighth_plant_roll_y'+str(int(yshift*1000));out=base.copy();records=[]
 for f in range(572,781):
  q=base[f].astype(float);pos,R,_=model.evaluate(q,1);w=smooth((f-572)/35)*(1-smooth((f-730)/50));target=pos+np.array([-.01,yshift,0.])*w
  targetR=Rotation.from_rotvec(np.array([2.8*yshift*w,0.,0.])).as_matrix()@R
  target[2]+=np.min((points@R.T)[:,2]-radii)-np.min((points@targetR.T)[:,2]-radii)
  q,pe,re=solve_leg_pose(model,q,1,target,targetR);out[f]=q;records.append([f,pe,re])
 out[572:781,7:]=np.clip(out[572:781,7:],limits[:,0]+1e-5,limits[:,1]-1e-5);records=np.array(records);rmse=np.sqrt(np.mean((out[:,7:]-source[:,7:])**2,axis=1));tensor=torch.tensor(out);curve=BlendedSplineReference(MotionInterpolator(tensor),tensor,stride=6,start_frame=130,ramp_frames=20);dense=np.array([curve.kinematics(f,200)[0].numpy()[0] for f in np.arange(572,780.01,.1)]);margin=float(np.minimum(dense[:,7:]-limits[:,0],limits[:,1]-dense[:,7:]).min())
 report=dict(landing_offset_xy=[-.01,yshift],peak_position_residual=float(records[:,1].max()),peak_orientation_residual=float(records[:,2].max()),peak_source_joint_reference_rmse=float(rmse.max()),minimum_dense_joint_limit_margin=margin,physical_tracking_certified=False)
 report['preflight_passed']=bool(records[:,1].max()<.002 and records[:,2].max()<.01 and rmse.max()<.19 and margin>=0);report['worst_rows']=records[np.argsort(records[:,1])[-5:]].tolist()
 np.savez_compressed(p/(name+'.npz'),control_q=out,fps=200.,baseline_sha256=hashlib.sha256(baseline['control_q'].tobytes()).hexdigest(),source_sha256=hashlib.sha256(source.tobytes()).hexdigest());(p/(name+'.json')).write_text(json.dumps(report,indent=2)+'\n');print(name,report,flush=True)
