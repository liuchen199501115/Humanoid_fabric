"""Bounded offline right-foot offset after frame906, preserving root and other joints."""
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
label,y_mm,z_mm=sys.argv[1],float(sys.argv[2]),float(sys.argv[3]);assert -40<=y_mm<=0 and -30<=z_mm<=0
profile=sys.argv[4] if len(sys.argv)>4 else 'plateau'
assert profile in ('plateau','compact','compact918')
compact=profile.startswith('compact');start=918 if profile=='compact918' else 906
parent=OUT/'prepared_rooty_clearance10_lift10.npz';target=OUT/('prepared_swing12_'+label+'.npz');assert not target.exists()
base=dict(np.load(parent));source=np.load(OUT/'ten_verify.control_reference.npz')['source_q']
assert hashlib.sha256(source.tobytes()).hexdigest()==base['source_sha256'].item()
names=np.load(OUT/'ten_verify.npz')['final_joint_order'].tolist();legs=LegKinematics(str(ROOT/'src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf'),names)
g=json.loads((OUT/'ten_verify.steps_plan.json').read_text());points=np.asarray(g['collision_points'][1]);radii=np.asarray(g['collision_radii'][1])
def smooth(x):
 x=np.clip(x,0,1);return x**3*(10-15*x+6*x*x)
out=base['control_q'].copy();peakp=peakr=shift=0.;minimum=1.;original_minimum=1.;worst_frame=None
end=start+48 if compact else 990
for k in range(start+1,end+1):
 original=base['control_q'][k];p,R,_=legs.evaluate(original,1);u=(k-start)/(end-start)
 phase=64*u**3*(1-u)**3 if compact else smooth((k-906)/24)*(1-smooth((k-966)/24))
 floor=.003-float(np.min((points@R.T)[:,2]-radii));goal=p.copy();goal[1]+=phase*y_mm/1000;goal[2]=min(p[2],max(p[2]+phase*z_mm/1000,floor));q=original.astype(float).copy();shift=max(shift,float(np.linalg.norm(goal-p)))
 for _ in range(200):
  pos,rot,J=legs.evaluate(q,1);error=np.r_[goal-pos,Rotation.from_matrix(R@rot.T).as_rotvec()]
  if np.max(abs(error))<5e-7:break
  lo=np.maximum(-.05,legs.lower[6:]-q[13:19]);hi=np.minimum(.05,legs.upper[6:]-q[13:19])
  q[13:19]+=lsq_linear(np.vstack((J,.0001*np.eye(6))),np.r_[error,np.zeros(6)],bounds=(lo,hi),tol=1e-10).x
 out[k]=q;pos,rot,_=legs.evaluate(out[k],1)
 residual=float(np.linalg.norm(goal-pos))
 if residual>peakp:peakp=residual;worst_frame=k
 peakr=max(peakr,float(Rotation.from_matrix(R@rot.T).magnitude()))
 minimum=min(minimum,float(pos[2]+np.min((points@rot.T)[:,2]-radii)));original_minimum=min(original_minimum,float(p[2]+np.min((points@R.T)[:,2]-radii)))
np.testing.assert_array_equal(out[:907],base['control_q'][:907]);np.testing.assert_array_equal(out[991:],base['control_q'][991:]);np.testing.assert_array_equal(out[:,:13],base['control_q'][:,:13]);np.testing.assert_array_equal(out[:,19:],base['control_q'][:,19:])
def sampler(q):
 t=torch.tensor(q);return BlendedSplineReference(MotionInterpolator(t),t,stride=6,start_frame=130,ramp_frames=20)
old,new=sampler(base['control_q']),sampler(out)
for k in np.arange(0,900.01,.5):
 for a,b in zip(old.kinematics(k,280.,0.),new.kinematics(k,280.,0.)):np.testing.assert_array_equal(a.numpy(),b.numpy())
rmse=float(np.sqrt(np.mean((out[:,7:]-source[:,7:])**2,axis=1)).max())
report=dict(method=__doc__,parent_file=parent.name,parent_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),source_sha256=base['source_sha256'].item(),requested_y_mm=y_mm,requested_z_mm=z_mm,compact=compact,start_frame=start,end_frame=end,worst_foot_residual_frame=worst_frame,maximum_foot_shift_m=shift,maximum_foot_position_residual_m=peakp,maximum_foot_rotation_residual_rad=peakr,minimum_native_clearance_m=minimum,parent_minimum_native_clearance_m=original_minimum,peak_source_joint_rmse_rad=rmse,prefix_q_v_a_equal_through_frame=900,root_left_leg_upper_body_unchanged=True,preflight_passed=peakp<2e-5 and peakr<2e-4 and rmse<.19 and minimum>=min(.003,original_minimum)-2e-6,full_tracking_passed=False)
target.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
if report['preflight_passed']:
 with target.open('xb') as stream:np.savez_compressed(stream,**dict(base,control_q=out))
print(target.name,report,flush=True)
