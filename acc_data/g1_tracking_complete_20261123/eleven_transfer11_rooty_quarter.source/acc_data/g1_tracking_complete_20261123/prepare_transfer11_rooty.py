"""Remove prepared lateral root offset during step-11 transfer, preserving both world foot poses."""
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
label=sys.argv[1];fraction=float(sys.argv[2]);assert 0<fraction<=1
parent=OUT/'prepared_rooty_clearance10_lift10.npz';target=OUT/('prepared_transfer11_rooty_'+label+'.npz');assert not target.exists()
base=dict(np.load(parent));source=np.load(OUT/'ten_verify.control_reference.npz')['source_q'];assert hashlib.sha256(source.tobytes()).hexdigest()==base['source_sha256'].item()
names=np.load(OUT/'ten_verify.npz')['final_joint_order'].tolist();legs=LegKinematics(str(ROOT/'src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf'),names)
def smooth(u):
 u=np.clip(u,0,1);return u**3*(10-15*u+6*u*u)
out=base['control_q'].copy();peakp=peakr=shift=0.;worst=None
for k in range(841,990):
 original=out[k].copy();q=original.astype(float);q[1]+=fraction*smooth((k-840)/60)*(1-smooth((k-960)/30))*(source[k,1]-q[1]);shift=max(shift,abs(q[1]-original[1]))
 for side in (0,1):
  goal,R,_=legs.evaluate(original,side);s=slice(7+6*side,13+6*side);l=slice(6*side,6*side+6)
  for _ in range(180):
   pos,rot,J=legs.evaluate(q,side);err=np.r_[goal-pos,Rotation.from_matrix(R@rot.T).as_rotvec()]
   if np.max(abs(err))<5e-7:break
   lo=np.maximum(-.05,legs.lower[l]-q[s]);hi=np.minimum(.05,legs.upper[l]-q[s])
   q[s]+=lsq_linear(np.vstack([J,.0001*np.eye(6)]),np.r_[err,np.zeros(6)],bounds=(lo,hi),tol=1e-10).x
  pos,rot,_=legs.evaluate(q,side);res=float(np.linalg.norm(goal-pos))
  if res>peakp:peakp=res;worst=(k,side)
  peakr=max(peakr,float(Rotation.from_matrix(R@rot.T).magnitude()))
 out[k]=q
np.testing.assert_array_equal(out[:841],base['control_q'][:841]);np.testing.assert_array_equal(out[990:],base['control_q'][990:]);np.testing.assert_array_equal(out[:,19:],base['control_q'][:,19:]);np.testing.assert_array_equal(out[:,2:7],base['control_q'][:,2:7]);np.testing.assert_array_equal(out[:,0],base['control_q'][:,0])
def sampler(q):
 t=torch.tensor(q);return BlendedSplineReference(MotionInterpolator(t),t,stride=6,start_frame=130,ramp_frames=20)
a,b=sampler(base['control_q']),sampler(out)
for k in np.arange(0,834.01,.5):
 for x,y in zip(a.kinematics(k,280.,0.),b.kinematics(k,280.,0.)):np.testing.assert_array_equal(x.numpy(),y.numpy())
rmse=float(np.sqrt(np.mean((out[:,7:]-source[:,7:])**2,axis=-1)).max());limits=np.load(OUT/'ten_verify.npz')['joint_position_limits'][0];violation=float(np.maximum(limits[:,0]-out[:,7:],out[:,7:]-limits[:,1]).clip(0).max())
r=dict(method=__doc__,fraction=fraction,source_sha256=base['source_sha256'].item(),parent_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),maximum_root_y_shift_m=shift,peak_position_residual_m=peakp,peak_rotation_residual_rad=peakr,worst_frame_and_side=worst,maximum_joint_limit_violation_rad=violation,peak_source_joint_rmse_rad=rmse,prefix_q_v_a_equal_through_frame=834,upper_root_xz_rotation_unchanged=True,preflight_passed=peakp<2e-5 and peakr<2e-4 and rmse<.19 and violation<1e-6,full_tracking_passed=False)
target.with_suffix('.json').write_text(json.dumps(r,indent=2)+'\n')
if r['preflight_passed']:
 with target.open('xb') as f:np.savez_compressed(f,**dict(base,control_q=out))
print(target.name,r,flush=True)
