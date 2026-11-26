"""Root support correction with bounded root rotation and exact foot-pose checks."""
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
with np.load(ROOT/'acc_data/g1_tracking_contacts_20261123/eight_verify.control_reference.npz') as a:source=a['source_q']
assert hashlib.sha256(source.tobytes()).hexdigest()==base['source_sha256'].item()
with np.load(ROOT/'acc_data/g1_tracking_contacts_20261123/eight_verify.npz') as a:names=a['final_joint_order'].tolist()
model=LegKinematics(str(ROOT/'src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf'),names)
def smooth(t):
 t=np.clip(t,0,1);return t**3*(10-15*t+6*t*t)
def skew(v):
 x,y,z=v;return np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
for hold in [4.1]:
 target=OUT/f'prepared_root_continue{round(hold*10)}.npz'
 if target.exists():raise FileExistsError(target)
 out=base['control_q'].copy();peakp=peakr=peakroot=0.;worst=None
 for k,original in enumerate(base['control_q']):
  weight=smooth((k/200.-3.2)/.5)*(1-smooth((k/200.-hold)/.5))
  if weight<1e-12:continue
  q=original.astype(float).copy();q[:3]+=weight*np.array([0,.05,-.005]);root0=Rotation.from_quat(q[3:7])
  goals=[model.evaluate(original,side)[:2] for side in [0,1]]
  for iteration in range(80):
   J=np.zeros((12,15));error=np.zeros(12);ps=[];rs=[]
   for side in [0,1]:
    position,rotation,leg_J=model.evaluate(q,side);p,R=goals[side]
    pe=p-position;re=Rotation.from_matrix(R@rotation.T).as_rotvec();rows=slice(side*6,(side+1)*6)
    error[rows]=np.r_[pe,re];J[rows,:3]=np.vstack((-skew(position-q[:3]),np.eye(3)));J[rows,3+side*6:9+side*6]=leg_J
    ps.append(np.linalg.norm(pe));rs.append(np.linalg.norm(re))
   if max(ps)<5e-7 and max(rs)<5e-6:break
   root_error=(Rotation.from_quat(q[3:7])*root0.inv()).as_rotvec()
   lo=np.r_[np.maximum(-.01,-.08-root_error),np.maximum(-.06,model.lower-q[7:19])]
   hi=np.r_[np.minimum(.01,.08-root_error),np.minimum(.06,model.upper-q[7:19])]
   regularizer=np.diag(np.r_[np.full(3,.03),np.full(12,.001)])
   change=lsq_linear(np.vstack((J,regularizer)),np.r_[error,np.zeros(15)],bounds=(lo,hi),tol=1e-10,max_iter=100).x
   q[3:7]=(Rotation.from_rotvec(change[:3])*Rotation.from_quat(q[3:7])).as_quat();q[7:19]+=change[3:]
  out[k]=q
  for side in [0,1]:
   position,rotation,_=model.evaluate(out[k],side);p,R=goals[side]
   pe=float(np.linalg.norm(position-p));re=float(Rotation.from_matrix(rotation@R.T).magnitude())
   if max(pe,re)>max(peakp,peakr):worst=k
   peakp=max(peakp,pe);peakr=max(peakr,re)
  peakroot=max(peakroot,float((Rotation.from_quat(out[k,3:7])*root0.inv()).magnitude()))
 report=dict(hold_end_source_s=hold,peak_foot_position_error_m=peakp,peak_foot_rotation_error_rad=peakr,
   peak_root_rotation_change_rad=peakroot,worst_frame=worst,
   peak_source_joint_rmse_rad=float(np.sqrt(np.mean((out[:,7:]-source[:,7:])**2,axis=1)).max()),
   peak_source_root_displacement_m=float(np.linalg.norm(out[:,:3]-source[:,:3],axis=1).max()),
   source_sha256=base['source_sha256'].item(),parent_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),full_tracking_passed=False)
 passed=peakp<2e-5 and peakr<2e-4 and peakroot<.1 and report['peak_source_joint_rmse_rad']<.19 and report['peak_source_root_displacement_m']<.1
 report['preflight_passed']=passed;target.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
 if passed:
  with target.open('xb') as stream:np.savez_compressed(stream,**dict(base,control_q=out))
 print(report,flush=True)
