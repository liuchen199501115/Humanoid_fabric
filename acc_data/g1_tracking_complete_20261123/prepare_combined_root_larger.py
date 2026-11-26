"""Offline root translation with both world foot poses preserved by bounded IK."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'acc_data/amp_walk_20261103/feedback_followup'))
from root_balance_offset import root_balance_offset
from numpy_leg_kinematics import LegKinematics
parent=OUT/'prepared_support7_y30.npz'
with np.load(parent) as a:base={k:a[k] for k in a.files}
with np.load(ROOT/'acc_data/g1_tracking_contacts_20261123/eight_verify.control_reference.npz') as a:source=a['source_q']
assert hashlib.sha256(source.tobytes()).hexdigest()==base['source_sha256'].item()
with np.load(ROOT/'acc_data/g1_tracking_contacts_20261123/eight_verify.npz') as a:names=a['final_joint_order'].tolist()
urdf=ROOT/'src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf';model=LegKinematics(str(urdf),names)
for mm in [70,90]:
 target=OUT/f'prepared_combined_root_y{mm}.npz'
 if target.exists():raise FileExistsError(target)
 arr,report=root_balance_offset(base['control_q'],names,str(urdf),offset=(0,mm/1000.,-.005),start=2.5,rise=.3,hold_end=3.2,fall=.5)
 # Verify the actual final samples, including the last IK iteration.
 pos=rot=0.
 for frame in range(501,740):
  for side in [0,1]:
   p0,R0,_=model.evaluate(base['control_q'][frame],side);p1,R1,_=model.evaluate(arr[frame],side)
   pos=max(pos,float(np.linalg.norm(p1-p0)));rot=max(rot,float(Rotation.from_matrix(R1@R0.T).magnitude()))
 report.update(verified_foot_position_error_m=pos,verified_foot_rotation_error_rad=rot,
   peak_source_joint_rmse_rad=float(np.sqrt(np.mean((arr[:,7:]-source[:,7:])**2,axis=1)).max()),
   peak_source_root_displacement_m=float(np.linalg.norm(arr[:,:3]-source[:,:3],axis=1).max()),
   parent_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),source_sha256=base['source_sha256'].item())
 np.testing.assert_array_equal(arr[:501],base['control_q'][:501]);np.testing.assert_array_equal(arr[740:],base['control_q'][740:])
 passed=pos<2e-5 and rot<2e-4 and report['peak_source_joint_rmse_rad']<.19 and report['peak_source_root_displacement_m']<.1
 report['preflight_passed']=passed
 target.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
 if passed:
  with target.open('xb') as stream:np.savez_compressed(stream,**dict(base,control_q=arr))
 print(mm,report,flush=True)
