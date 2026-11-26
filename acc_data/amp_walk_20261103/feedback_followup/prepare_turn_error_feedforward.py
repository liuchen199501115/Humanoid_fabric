"""Bounded reference learning from a previous physical rollout, with feet/COM retained.
This changes only the next run's smooth reference; no simulated state is replayed.
"""
from pathlib import Path
import json,numpy as np,torch
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation
from numpy_leg_kinematics import LegKinematics
from numpy_g1_com import CenterOfMassModel
from bounded_leg_ik import solve_leg_pose
from blended_spline import BlendedSplineReference
from fabrics_sim.reference.interpolation import MotionInterpolator
p=Path(__file__).parent;data=np.load(p/'turn_recovery_y40.npz');base=data['control_q'];d=np.load(p/'fixed_clock120_body10_overdue.npz');saved=np.load(p/'wrench_fixed_ten.npz');urdf='src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf';leg=LegKinematics(urdf,saved['final_joint_order']);com=CenterOfMassModel(urdf,saved['final_joint_order']);limits=saved['joint_position_limits'][0];source=np.load(p/'fixed_smooth_early.control_reference.npz')['source_q'];plan=json.load(open(p/'wrench_fixed_ten.steps_plan.json'));geometry=[(np.array(plan['collision_points'][j]),np.array(plan['collision_radii'][j])) for j in [0,1]]
frame=d['history_frame'].ravel();actual=d['history_q'].reshape(-1,36);ref=d['history_reference_q'].reshape(-1,36);rotation_error=(Rotation.from_quat(actual[:,3:7])*Rotation.from_quat(ref[:,3:7]).inv()).as_rotvec();error=np.c_[rotation_error,actual[:,19:]-ref[:,19:]];grid=np.arange(2000);errors=np.stack([np.interp(grid+12,frame,error[:,i]) for i in range(20)],1);errors=gaussian_filter1d(errors,8,axis=0)
def smooth(x):
 x=np.clip(x,0,1);return x**3*(10-15*x+6*x*x)
for start,gain in [(842,.5),(810,.5)]:
 weight=smooth((grid-start)/40)*(1-smooth((grid-1000)/100));offset=-gain*errors*weight[:,None];offset[:,:3]=.12*np.tanh(offset[:,:3]/.12);offset[:,3:]=.15*np.tanh(offset[:,3:]/.15);out=base.copy();res=[]
 for f in range(start+1,1100):
  q=base[f].astype(float);target=com.evaluate(q);feet=[leg.evaluate(q,j)[:2] for j in [0,1]];q[3:7]=(Rotation.from_rotvec(offset[f,:3])*Rotation.from_quat(q[3:7])).as_quat();q[19:]+=offset[f,3:]
  for iteration in range(16):
   q[:3]+=1.2*(target-com.evaluate(q));peaks=[]
   for j,(pos,R) in enumerate(feet):q,pe,re=solve_leg_pose(leg,q,j,pos,R);peaks.append([pe,re])
   ce=np.linalg.norm(com.evaluate(q)-target)
   if ce<1e-7:break
  out[f]=q;res.append([ce,*np.max(peaks,axis=0)])
 t=torch.tensor(out);curve=BlendedSplineReference(MotionInterpolator(t),t,stride=6,start_frame=130,ramp_frames=20);frames=np.arange(start-6,1107,.25);values=[curve.kinematics(float(f),240.) for f in frames];qs=np.array([x[0].numpy()[0] for x in values]);clear=[]
 for q in qs:
  for side in [0,1]:
   pos,R,_=leg.evaluate(q,side);pts,r=geometry[side];clear.append(pos[2]+min((pts@R.T)[:,2]-r))
 err=np.max(res,axis=0);margin=float(np.minimum(qs[:,7:]-limits[:,0],limits[:,1]-qs[:,7:]).min());rmse=float(np.sqrt(np.mean((out[:,7:]-source[:,7:])**2,axis=1)).max());vel=float(max(abs(x[1].numpy()[0,6:]).max() for x in values));report=dict(parent='turn_recovery_y40',learning_rollout='fixed_clock120_body10_overdue',source_interval=[start,1100],gain=gain,preview_frames=12,peak_com_projection_error=float(err[0]),peak_position_residual=float(err[1]),peak_orientation_residual=float(err[2]),minimum_dense_joint_limit_margin=margin,minimum_dense_native_clearance=float(min(clear)),peak_source_joint_reference_rmse=rmse,maximum_dense_joint_velocity=vel,physical_tracking_certified=False);report['preflight_passed']=bool(err[0]<.002 and err[1]<.002 and err[2]<.01 and margin>.005 and rmse<.19 and vel<20 and min(clear)>-.0015)
 name=f'turn_error_feedforward_{start}';np.savez_compressed(p/(name+'.npz'),control_q=out,fps=200.,baseline_sha256=data['baseline_sha256'],source_sha256=data['source_sha256']);(p/(name+'.json')).write_text(json.dumps(report,indent=2)+'\n');print(name,report,flush=True)
