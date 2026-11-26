"""Offline touchdown-height shaping with bounded right-leg IK and unchanged source audit."""
import hashlib,json,sys
from pathlib import Path
import numpy as np,torch
from scipy.optimize import lsq_linear
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'acc_data/amp_walk_20261103/feedback_followup'))
from numpy_leg_kinematics import LegKinematics
from blended_spline import BlendedSplineReference
from fabrics_sim.reference.interpolation import MotionInterpolator
start,end=map(int,sys.argv[1:3]);assert 918<=start<end<=950
parent=OUT/'prepared_return_source_transfer60.npz';target=OUT/f'prepared_step12_ground_{start}_{end}.npz';assert not target.exists()
base=dict(np.load(parent));source=np.load(OUT/'ten_verify.control_reference.npz')['source_q'];assert hashlib.sha256(source.tobytes()).hexdigest()==base['source_sha256'].item()
record=np.load(OUT/'ten_verify.npz');legs=LegKinematics(str(ROOT/'src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf'),record['final_joint_order'].tolist());geometry=json.loads((OUT/'ten_verify.steps_plan.json').read_text());pts=np.asarray(geometry['collision_points'][1]);radii=np.asarray(geometry['collision_radii'][1])
def smooth(x):
 x=np.clip(x,0,1);return x**3*(10-15*x+6*x*x)
out=base['control_q'].copy();peakp=peakr=shift=0.;minimum=parent_minimum=1.;peak_clearance=0.
for k in range(start+1,961):
 q=out[k].astype(float);p,R,_=legs.evaluate(q,1);clearance=float(p[2]+((pts@R.T)[:,2]-radii).min());parent_minimum=min(parent_minimum,clearance)
 delta=smooth((k-start)/(end-start))*(1-smooth((k-952)/8))*max(clearance,0.);p[2]-=delta;shift=max(shift,delta)
 for _ in range(180):
  at,rot,J=legs.evaluate(q,1);err=np.r_[p-at,Rotation.from_matrix(R@rot.T).as_rotvec()]
  if np.max(abs(err))<5e-7:break
  lo=np.maximum(-.05,legs.lower[6:]-q[13:19]);hi=np.minimum(.05,legs.upper[6:]-q[13:19]);q[13:19]+=lsq_linear(np.vstack([J,.0001*np.eye(6)]),np.r_[err,np.zeros(6)],bounds=(lo,hi),tol=1e-10).x
 out[k]=q;at,rot,_=legs.evaluate(out[k],1);peakp=max(peakp,float(np.linalg.norm(p-at)));peakr=max(peakr,float(Rotation.from_matrix(R@rot.T).magnitude()));minimum=min(minimum,float(at[2]+((pts@rot.T)[:,2]-radii).min()))
for k in range(908,952):
 p,R,_=legs.evaluate(out[k],1);peak_clearance=max(peak_clearance,float(p[2]+((pts@R.T)[:,2]-radii).min()))
np.testing.assert_array_equal(out[:start+1],base['control_q'][:start+1]);np.testing.assert_array_equal(out[961:],base['control_q'][961:]);np.testing.assert_array_equal(out[:,:13],base['control_q'][:,:13]);np.testing.assert_array_equal(out[:,19:],base['control_q'][:,19:])
def sampler(q):
 q=torch.tensor(q);return BlendedSplineReference(MotionInterpolator(q),q,stride=6,start_frame=130,ramp_frames=20)
a,b=sampler(base['control_q']),sampler(out);prefix=6*(start//6)-6
for f in np.arange(0,prefix+.01,.5):
 for x,y in zip(a.kinematics(f,280.,0.),b.kinematics(f,280.,0.)):np.testing.assert_array_equal(x.numpy(),y.numpy())
limits=record['joint_position_limits'][0];violation=float(np.maximum(limits[:,0]-out[:,7:],out[:,7:]-limits[:,1]).clip(0).max());rmse=float(np.sqrt(np.mean((out[:,7:]-source[:,7:])**2,axis=-1)).max())
r=dict(parent_file=parent.name,parent_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),source_sha256=base['source_sha256'].item(),start_frame=start,end_frame=end,maximum_vertical_shift_m=shift,maximum_ik_position_residual_m=peakp,maximum_ik_rotation_residual_rad=peakr,minimum_native_clearance_m=minimum,parent_minimum_native_clearance_m=parent_minimum,peak_reference_flight_clearance_m=peak_clearance,maximum_joint_limit_violation_rad=violation,maximum_source_joint_rmse_rad=rmse,prefix_q_v_a_exact_through_frame=prefix,root_left_upper_unchanged=True,preflight_passed=peakp<2e-5 and peakr<2e-4 and violation<1e-6 and rmse<.19 and minimum>=parent_minimum-2e-6 and peak_clearance>=.02,full_tracking_passed=False)
target.with_suffix('.json').write_text(json.dumps(r,indent=2)+'\n');print(r,flush=True)
if r['preflight_passed']:
 with target.open('xb') as file:np.savez_compressed(file,**dict(base,control_q=out))
