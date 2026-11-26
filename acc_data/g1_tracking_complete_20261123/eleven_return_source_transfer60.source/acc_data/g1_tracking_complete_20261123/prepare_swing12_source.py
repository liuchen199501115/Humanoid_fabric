"""Offline bounded right-leg IK toward the original world foot pose after step 11."""
import hashlib,json,sys
from pathlib import Path
import numpy as np, torch
from scipy.optimize import lsq_linear
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'acc_data/amp_walk_20261103/feedback_followup'))
from numpy_leg_kinematics import LegKinematics
from blended_spline import BlendedSplineReference
from fabrics_sim.reference.interpolation import MotionInterpolator
label=sys.argv[1];end=int(sys.argv[2]);assert 930<=end<=990
parent=OUT/'prepared_rooty_clearance10_lift10.npz';target=OUT/('prepared_swing12_source_'+label+'.npz');assert not target.exists()
base=dict(np.load(parent));source=np.load(OUT/'ten_verify.control_reference.npz')['source_q'];assert hashlib.sha256(source.tobytes()).hexdigest()==base['source_sha256'].item()
names=np.load(OUT/'ten_verify.npz')['final_joint_order'].tolist();legs=LegKinematics(str(ROOT/'src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf'),names)
g=json.loads((OUT/'ten_verify.steps_plan.json').read_text());points=np.asarray(g['collision_points'][1]);radii=np.asarray(g['collision_radii'][1])
def smooth(u):
 u=np.clip(u,0,1);return u**3*(10-15*u+6*u*u)
out=base['control_q'].copy();peakp=peakr=0.;minimum=parentminimum=1.;worst=None
for k in range(907,1040):
 old=out[k].copy();p,R,_=legs.evaluate(old,1);sp,sR,_=legs.evaluate(source[k],1)
 blend=smooth((k-906)/(end-906))*(1-smooth((k-1010)/30))
 goal=p+blend*(sp-p);goalR=Rotation.from_rotvec(blend*Rotation.from_matrix(sR@R.T).as_rotvec()).as_matrix()@R
 q=old.astype(float)
 for _ in range(150):
  pos,rot,J=legs.evaluate(q,1);err=np.r_[goal-pos,Rotation.from_matrix(goalR@rot.T).as_rotvec()]
  if np.max(abs(err))<5e-7:break
  lo=np.maximum(-.05,legs.lower[6:]-q[13:19]);hi=np.minimum(.05,legs.upper[6:]-q[13:19])
  q[13:19]+=lsq_linear(np.vstack([J,.0001*np.eye(6)]),np.r_[err,np.zeros(6)],bounds=(lo,hi),tol=1e-10).x
 out[k]=q;pos,rot,_=legs.evaluate(out[k],1);res=float(np.linalg.norm(goal-pos))
 if res>peakp:peakp=res;worst=k
 peakr=max(peakr,float(Rotation.from_matrix(goalR@rot.T).magnitude()))
 minimum=min(minimum,float(pos[2]+np.min((points@rot.T)[:,2]-radii)));parentminimum=min(parentminimum,float(p[2]+np.min((points@R.T)[:,2]-radii)))
np.testing.assert_array_equal(out[:907],base['control_q'][:907]);np.testing.assert_array_equal(out[1040:],base['control_q'][1040:]);np.testing.assert_array_equal(out[:,:13],base['control_q'][:,:13]);np.testing.assert_array_equal(out[:,19:],base['control_q'][:,19:])
def sampler(q):
 t=torch.tensor(q);return BlendedSplineReference(MotionInterpolator(t),t,stride=6,start_frame=130,ramp_frames=20)
a,b=sampler(base['control_q']),sampler(out)
for k in np.arange(0,900.01,.5):
 for x,y in zip(a.kinematics(k,280.,0.),b.kinematics(k,280.,0.)):np.testing.assert_array_equal(x.numpy(),y.numpy())
rmse=float(np.sqrt(np.mean((out[:,7:]-source[:,7:])**2,axis=-1)).max());limits=np.load(OUT/'ten_verify.npz')['joint_position_limits'][0];violation=float(np.maximum(limits[:,0]-out[:,7:],out[:,7:]-limits[:,1]).clip(0).max())
r=dict(method=__doc__,end_frame=end,source_sha256=base['source_sha256'].item(),parent_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),peak_position_residual_m=peakp,peak_rotation_residual_rad=peakr,worst_frame=worst,minimum_native_clearance_m=minimum,parent_minimum_native_clearance_m=parentminimum,maximum_joint_limit_violation_rad=violation,peak_source_joint_rmse_rad=rmse,prefix_q_v_a_equal_through_frame=900,root_left_upper_unchanged=True,preflight_passed=peakp<2e-5 and peakr<2e-4 and rmse<.19 and violation<1e-6 and minimum>=parentminimum-2e-6,full_tracking_passed=False)
target.with_suffix('.json').write_text(json.dumps(r,indent=2)+'\n')
if r['preflight_passed']:
 with target.open('xb') as f:np.savez_compressed(f,**dict(base,control_q=out))
print(target.name,r,flush=True)
