"""Offline bounded left-leg IK for a small step11 foot displacement; source audit stays original."""
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
label=sys.argv[1];xy=np.array([float(sys.argv[2]),float(sys.argv[3])])/1000;assert np.linalg.norm(xy)<=.030001
parent=OUT/'prepared_return_source_transfer60.npz';target=OUT/f'prepared_step11_foot_{label}.npz';assert not target.exists()
base=dict(np.load(parent));source=np.load(OUT/'ten_verify.control_reference.npz')['source_q'];assert hashlib.sha256(source.tobytes()).hexdigest()==base['source_sha256'].item()
record=np.load(OUT/'ten_verify.npz');legs=LegKinematics(str(ROOT/'src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf'),record['final_joint_order'].tolist())
def smooth(x):
 x=np.clip(x,0,1);return x**3*(10-15*x+6*x*x)
out=base['control_q'].copy();peakp=peakr=0.
for k in range(841,990):
 q=out[k].astype(float);p,R,_=legs.evaluate(q,0);p[:2]+=xy*smooth((k-840)/40)*(1-smooth((k-960)/30))
 for _ in range(180):
  at,rot,J=legs.evaluate(q,0);err=np.r_[p-at,Rotation.from_matrix(R@rot.T).as_rotvec()]
  if np.max(abs(err))<5e-7:break
  lo=np.maximum(-.05,legs.lower[:6]-q[7:13]);hi=np.minimum(.05,legs.upper[:6]-q[7:13]);q[7:13]+=lsq_linear(np.vstack([J,.0001*np.eye(6)]),np.r_[err,np.zeros(6)],bounds=(lo,hi),tol=1e-10).x
 out[k]=q;at,rot,_=legs.evaluate(out[k],0);peakp=max(peakp,float(np.linalg.norm(p-at)));peakr=max(peakr,float(Rotation.from_matrix(R@rot.T).magnitude()))
np.testing.assert_array_equal(out[:841],base['control_q'][:841]);np.testing.assert_array_equal(out[990:],base['control_q'][990:]);np.testing.assert_array_equal(out[:,:7],base['control_q'][:,:7]);np.testing.assert_array_equal(out[:,13:],base['control_q'][:,13:])
def sampler(q):
 q=torch.tensor(q);return BlendedSplineReference(MotionInterpolator(q),q,stride=6,start_frame=130,ramp_frames=20)
a,b=sampler(base['control_q']),sampler(out)
for f in np.arange(0,834.01,.5):
 for x,y in zip(a.kinematics(f,280.,0.),b.kinematics(f,280.,0.)):np.testing.assert_array_equal(x.numpy(),y.numpy())
limits=record['joint_position_limits'][0];violation=float(np.maximum(limits[:,0]-out[:,7:],out[:,7:]-limits[:,1]).clip(0).max());rmse=float(np.sqrt(np.mean((out[:,7:]-source[:,7:])**2,axis=-1)).max())
r=dict(parent_file=parent.name,parent_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),source_sha256=base['source_sha256'].item(),offset_xy_m=xy.tolist(),maximum_foot_position_ik_residual_m=peakp,maximum_foot_rotation_ik_residual_rad=peakr,maximum_joint_limit_violation_rad=violation,maximum_source_joint_rmse_rad=rmse,prefix_q_v_a_exact_through_frame=834,root_right_upper_unchanged=True,preflight_passed=peakp<2e-5 and peakr<2e-4 and violation<1e-6 and rmse<.19,full_tracking_passed=False)
target.with_suffix('.json').write_text(json.dumps(r,indent=2)+'\n');print(r,flush=True)
if r['preflight_passed']:
 with target.open('xb') as file:np.savez_compressed(file,**dict(base,control_q=out))
