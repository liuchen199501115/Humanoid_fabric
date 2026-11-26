"""Offline bounded pelvis-roll correction with fixed world foot poses.

The original source is unchanged. Acceptance uses original source fidelity.
"""
import hashlib,json,sys
from pathlib import Path
import numpy as np
from scipy.optimize import lsq_linear
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'acc_data/amp_walk_20261103/feedback_followup'))
from numpy_leg_kinematics import LegKinematics
parent=OUT/'prepared_combined_root_y50.npz'
with np.load(parent) as a:base={k:a[k] for k in a.files}
source=np.load(ROOT/'acc_data/g1_tracking_contacts_20261123/eight_verify.control_reference.npz')['source_q']
assert hashlib.sha256(source.tobytes()).hexdigest()==base['source_sha256'].item()
names=np.load(OUT/'ten_verify.npz')['final_joint_order'].tolist()
model=LegKinematics(str(ROOT/'src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf'),names)
def smooth(u):
 u=np.clip(u,0,1);return u**3*(10-15*u+6*u*u)
root_dip=float(sys.argv[1]) if len(sys.argv)>1 else .002
for angle in [-.06,-.12]:
 target=OUT/f'prepared_root_roll_minus{round(-angle*100):02}_z{round(root_dip*1000)}.npz'
 if target.exists():raise FileExistsError(target)
 out=base['control_q'].copy();maxp=maxr=0.;worst=None
 for k in range(681,1000):
  w=smooth((k-680)/90)*(1-smooth((k-900)/100));original=base['control_q'][k];q=original.astype(float).copy()
  q[3:7]=(Rotation.from_rotvec([angle*w,0,0])*Rotation.from_quat(q[3:7])).as_quat()
  q[2]-=root_dip*w
  for side in (0,1):
   p0,R0,_=model.evaluate(original,side);sl=slice(7+6*side,13+6*side);loq=model.lower[6*side:6*side+6];hiq=model.upper[6*side:6*side+6]
   for iteration in range(160):
    p,R,J=model.evaluate(q,side);error=np.r_[p0-p,Rotation.from_matrix(R0@R.T).as_rotvec()]
    if np.max(abs(error))<5e-7:break
    lo=np.maximum(-.05,loq-q[sl]);hi=np.minimum(.05,hiq-q[sl])
    if np.any(lo>=hi):q[sl]=np.clip(q[sl],loq+1e-8,hiq-1e-8);continue
    dq=lsq_linear(np.vstack([J,.0001*np.eye(6)]),np.r_[error,np.zeros(6)],bounds=(lo,hi),tol=1e-10).x;q[sl]+=dq
   out[k]=q;p,R,_=model.evaluate(out[k],side);pe=np.linalg.norm(p-p0);re=Rotation.from_matrix(R0@R.T).magnitude()
   if pe>maxp:worst=int(k)
   maxp=max(maxp,float(pe));maxr=max(maxr,float(re))
 rmse=float(np.sqrt(np.mean((out[:,7:]-source[:,7:])**2,axis=1)).max())
 report=dict(method='C2 negative world pelvis roll with bounded leg IK restoring both feet',roll_offset_rad=angle,root_z_offset_m=-root_dip,maximum_foot_position_residual_m=maxp,maximum_foot_rotation_residual_rad=maxr,worst_frame=worst,peak_source_joint_rmse_rad=rmse,parent_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),source_sha256=base['source_sha256'].item(),full_tracking_passed=False,preflight_passed=maxp<2e-5 and maxr<2e-4 and rmse<.19)
 np.testing.assert_array_equal(out[:681],base['control_q'][:681]);np.testing.assert_array_equal(out[1000:],base['control_q'][1000:])
 target.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
 if report['preflight_passed']:
  with target.open('xb') as f:np.savez_compressed(f,**dict(base,control_q=out))
 print(report,flush=True)
