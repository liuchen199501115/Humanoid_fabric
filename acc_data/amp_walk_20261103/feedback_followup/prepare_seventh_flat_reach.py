"""Preserve native foot clearance while flattening measured landing/stance poses."""
from pathlib import Path
import hashlib,json
import numpy as np,torch
from scipy.spatial.transform import Rotation
from fabrics_sim.reference.interpolation import MotionInterpolator
from numpy_leg_kinematics import LegKinematics
from bounded_leg_ik import solve_leg_pose
from blended_spline import BlendedSplineReference
p=Path(__file__).parent;z=np.load(p/'fixed_smooth_early.control_reference.npz');trajectory=z['control_q'];source=z['source_q'];base=np.load(p/'seventh_plant_roll_y25.npz')['control_q'];d=np.load(p/'wrench_fixed_ten.npz');limits=d['joint_position_limits'][0]
model=LegKinematics('src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf',d['final_joint_order']);plan=json.load(open(p/'wrench_fixed_ten.steps_plan.json'));events=plan['events'];geometry=[(np.array(plan['collision_points'][s]),np.array(plan['collision_radii'][s])) for s in range(2)]
def smooth(x):x=np.clip(x,0,1);return x**3*(10-15*x+6*x*x)
for gain in [.6,1.]:
 name='seventh_flat_reach'+str(int(gain*100));out=base.copy();records=[]
 for f in range(520,708):
  original=base[f].astype(float);q=original.copy();q[2]-=.025*smooth((f-520)/25)*(1-smooth((f-677)/30))
  for side in range(2):
   pos,R,_=model.evaluate(original,side);target=pos.copy();targetR=R;w=0.;height=0.
   for ei,event in enumerate(events[6:7],6):
    if 1-event['support']!=side:continue
    end=event['end_frame'];next_takeoff=events[ei+2]['takeoff_frame'] if ei+2<len(events) else len(out)+29
    w=max(w,smooth((f-(end-29))/25)*(1-smooth((f-next_takeoff)/25)))
    height+=0.*smooth((f-(end-24))/15)*(1-smooth((f-end)/14))
   points,radii=geometry[side]
   if w:
    flat=Rotation.from_euler('z',np.arctan2(R[1,0],R[0,0])).as_matrix();targetR=Rotation.from_rotvec(gain*w*Rotation.from_matrix(flat@R.T).as_rotvec()).as_matrix()@R
    target[2]+=np.min((points@R.T)[:,2]-radii)-np.min((points@targetR.T)[:,2]-radii)
   target[2]+=height
   q,pe,re=solve_leg_pose(model,q,side,target,targetR);records.append([f,side,pe,re])
  q[7:]=np.clip(q[7:],limits[:,0]+1e-5,limits[:,1]-1e-5);out[f]=q
 out[520:,7:]=np.clip(out[520:,7:],limits[:,0]+1e-5,limits[:,1]-1e-5)
 records=np.array(records);rmse=np.sqrt(np.mean((out[:,7:]-source[:,7:])**2,axis=1));tensor=torch.tensor(out);sampler=BlendedSplineReference(MotionInterpolator(tensor),tensor,stride=6,start_frame=130,ramp_frames=20);dense=np.array([sampler.kinematics(f,200)[0].numpy()[0] for f in np.arange(520,1999.01,.5)]);margin=float(np.minimum(dense[:,7:]-limits[:,0],limits[:,1]-dense[:,7:]).min())
 report=dict(gain=gain,peak_position_residual=float(records[:,2].max()),peak_orientation_residual=float(records[:,3].max()),peak_source_joint_rmse=float(rmse.max()),minimum_dense_joint_limit_margin=margin,initial_qva_preserved=True,physical_tracking_certified=False)
 report['preflight_passed']=bool(records[:,2].max()<.002 and records[:,3].max()<.01 and rmse.max()<.19 and margin>=0);report['worst_ik_rows']=records[np.argsort(records[:,2])[-5:]].tolist()
 np.savez_compressed(p/(name+'.npz'),control_q=out,fps=200.,baseline_sha256=hashlib.sha256(trajectory.tobytes()).hexdigest(),source_sha256=hashlib.sha256(source.tobytes()).hexdigest());(p/(name+'.json')).write_text(json.dumps(report,indent=2)+'\n');print(name,report,flush=True)
