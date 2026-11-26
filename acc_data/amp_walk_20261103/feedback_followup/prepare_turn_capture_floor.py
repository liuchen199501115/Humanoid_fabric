from pathlib import Path
import numpy as np,json,torch,hashlib
from fabrics_sim.reference.interpolation import MotionInterpolator
from numpy_leg_kinematics import LegKinematics
from bounded_leg_ik import solve_leg_pose
from blended_spline import BlendedSplineReference
p=Path(__file__).parent;parent='turn_capture_sole_y60_drop15';base=np.load(p/(parent+'.npz'));out=base['control_q'].copy();d=np.load(p/'wrench_fixed_ten.npz');limits=d['joint_position_limits'][0];m=LegKinematics('src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf',d['final_joint_order']);plan=json.load(open(p/'wrench_fixed_ten.steps_plan.json'));geometry=[(np.array(plan['collision_points'][j]),np.array(plan['collision_radii'][j])) for j in [0,1]]
def smooth(x):x=np.clip(x,0,1);return x**3*(10-15*x+6*x*x)
errors=[]
for f in range(1008,1061):
 q=out[f].astype(float);pos,R,_=m.evaluate(q,1);pos[2]+=.003*smooth((f-1008)/14)*(1-smooth((f-1044)/16));q,pe,re=solve_leg_pose(m,q,1,pos,R);out[f]=q;errors.append([pe,re])
t=torch.tensor(out);curve=BlendedSplineReference(MotionInterpolator(t),t,stride=6,start_frame=130,ramp_frames=20);samples=[curve.kinematics(f,240) for f in np.arange(674,1067,.2)];dense=np.array([x[0].numpy()[0] for x in samples]);margin=float(np.minimum(dense[:,7:]-limits[:,0],limits[:,1]-dense[:,7:]).min());minimum=1.
for q in dense:
 for side in [0,1]:
  pos,R,_=m.evaluate(q,side);pts,rs=geometry[side];minimum=min(minimum,float(pos[2]+min((pts@R.T)[:,2]-rs)))
source=np.load(p/'fixed_smooth_early.control_reference.npz')['source_q'];rmse=float(np.sqrt(np.mean((out[:,7:]-source[:,7:])**2,axis=1)).max());report=json.load(open(p/(parent+'.json')));report.update(parent=parent,tail_clearance_correction=.003,minimum_dense_native_clearance=minimum,minimum_dense_joint_limit_margin=margin,peak_source_joint_reference_rmse=rmse,tail_ik_position_residual=float(np.max(errors,axis=0)[0]),tail_ik_orientation_residual=float(np.max(errors,axis=0)[1]));report['pre_tail_native_clearance_error']=report.pop('peak_native_clearance_error');report['peak_foot_position_adjustment']=max(report['peak_foot_position_adjustment'],.003+report['tail_ik_position_residual']);report['maximum_dense_joint_velocity']=float(np.max([abs(x[1].numpy()[0,6:]).max() for x in samples]));report['audited_source_frame_interval']=[674.,1066.8];report['preflight_passed']=bool(report['peak_foot_xy_error']<.002 and report['pre_tail_native_clearance_error']<.001 and report['peak_foot_orientation_adjustment']<.1 and rmse<.19 and margin>.005 and minimum>-.0015 and np.max(errors,axis=0)[0]<.002 and np.max(errors,axis=0)[1]<.01);name=parent+'_floor';np.savez_compressed(p/(name+'.npz'),control_q=out,fps=200.,baseline_sha256=base['baseline_sha256'],source_sha256=base['source_sha256']);(p/(name+'.json')).write_text(json.dumps(report,indent=2)+'\n');print(name,report)
