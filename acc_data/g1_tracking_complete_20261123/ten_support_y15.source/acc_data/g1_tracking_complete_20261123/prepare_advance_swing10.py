"""Advance only the tenth swing foot through bounded, fixed-root leg IK."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
from scipy.optimize import lsq_linear
from scipy.spatial.transform import Rotation,Slerp
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'acc_data/amp_walk_20261103/feedback_followup'))
from numpy_leg_kinematics import LegKinematics
parent=OUT/'prepared_combined_root_y50.npz'
with np.load(parent) as a:base={k:a[k] for k in a.files}
with np.load(ROOT/'acc_data/g1_tracking_contacts_20261123/eight_verify.control_reference.npz') as a:source=a['source_q']
assert hashlib.sha256(source.tobytes()).hexdigest()==base['source_sha256'].item()
with np.load(ROOT/'acc_data/g1_tracking_contacts_20261123/eight_verify.npz') as a:names=a['final_joint_order'].tolist()
model=LegKinematics(str(ROOT/'src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf'),names)
start,end=762,802
poses=[model.evaluate(q,1)[:2] for q in base['control_q'][start:end+1]]
positions=np.array([v[0] for v in poses]);rotations=Slerp(np.arange(start,end+1),Rotation.from_matrix([v[1] for v in poses]))
for lead in [8,12]:
 target=OUT/f'prepared_advance_swing10_f{lead}.npz'
 if target.exists():raise FileExistsError(target)
 out=base['control_q'].copy();D=end-start-lead
 coeff=np.linalg.solve([[1,1,1],[3,4,5],[6,12,20]],[lead,-D,0.])
 peakp=peakr=0.;frames=[]
 for k in range(start+1,end):
  smooth=lambda x: np.clip(x,0,1)**3*(10-15*np.clip(x,0,1)+6*np.clip(x,0,1)**2)
  f=float(k+lead*smooth((k-start)/8)*(1-smooth((k-(end-24))/24)));frames.append(f)
  j=min(end-start-1,int(f)-start);alpha=f-(start+j)
  p=(1-alpha)*positions[j]+alpha*positions[j+1];R=rotations(f).as_matrix()
  q=out[k].astype(float).copy()
  for iteration in range(80):
   pos,rot,J=model.evaluate(q,1);e=np.r_[p-pos,Rotation.from_matrix(R@rot.T).as_rotvec()]
   if np.linalg.norm(e[:3])<5e-7 and np.linalg.norm(e[3:])<5e-6:break
   lo=np.maximum(-.06,model.lower[6:]-q[13:19]);hi=np.minimum(.06,model.upper[6:]-q[13:19])
   delta=lsq_linear(np.vstack([J,.001*np.eye(6)]),np.r_[e,np.zeros(6)],bounds=(lo,hi),tol=1e-10).x
   q[13:19]+=delta
  out[k]=q
  pos,rot,_=model.evaluate(out[k],1);peakp=max(peakp,float(np.linalg.norm(p-pos)));peakr=max(peakr,float(Rotation.from_matrix(R@rot.T).magnitude()))
 np.testing.assert_array_equal(out[:start+1],base['control_q'][:start+1]);np.testing.assert_array_equal(out[end:],base['control_q'][end:])
 np.testing.assert_array_equal(out[:,:13],base['control_q'][:,:13]);np.testing.assert_array_equal(out[:,19:],base['control_q'][:,19:])
 assert np.min(np.diff(frames))>=-1e-10
 report=dict(advance_frames=lead,advance_source_seconds=lead/200.,peak_foot_position_error_m=peakp,peak_foot_rotation_error_rad=peakr,
   peak_source_joint_rmse_rad=float(np.sqrt(np.mean((out[:,7:]-source[:,7:])**2,axis=1)).max()),
   peak_control_joint_change_rmse_rad=float(np.sqrt(np.mean((out[:,7:]-base['control_q'][:,7:])**2,axis=1)).max()),
   root_other_leg_and_upper_body_exactly_unchanged=True,source_sha256=base['source_sha256'].item(),parent_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),full_tracking_passed=False)
 report['preflight_passed']=peakp<2e-5 and peakr<2e-4 and report['peak_source_joint_rmse_rad']<.19
 target.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
 if report['preflight_passed']:
  with target.open('xb') as stream:np.savez_compressed(stream,**dict(base,control_q=out))
 print(report,flush=True)
