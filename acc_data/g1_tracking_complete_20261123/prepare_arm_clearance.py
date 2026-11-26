"""Smooth arm-only reference corrections; retain source audit and foot/root poses."""
import hashlib
import json
from pathlib import Path
import numpy as np

OUT=Path(__file__).resolve().parent
parent=OUT.parent/'g1_tracking_contacts_20261123/prepared_landing8_y15.npz'
with np.load(parent) as a: original={k:a[k] for k in a.files}
with np.load(OUT.parent/'g1_tracking_contacts_20261123/eight_verify.control_reference.npz') as a: source=a['source_q']
assert hashlib.sha256(source.tobytes()).hexdigest()==original['source_sha256'].item()
with np.load(OUT.parent/'g1_tracking_contacts_20261123/eight_verify.npz') as a:
 names=a['final_joint_order'].tolist();limits=a['joint_position_limits'].reshape(29,2)
left=names.index('left_shoulder_roll_joint')+7
right=names.index('right_shoulder_roll_joint')+7
phase=np.clip((np.arange(len(source))-500)/80.,0,1)
blend=phase**3*(10-15*phase+6*phase**2)
for amount in [.2,.3]:
 q=original['control_q'].copy();q[:,left]+=amount*blend;q[:,right]-=amount*blend
 other=[j for j in range(q.shape[1]) if j not in [left,right]]
 np.testing.assert_array_equal(q[:,other],original['control_q'][:,other])
 np.testing.assert_array_equal(q[:501],original['control_q'][:501])
 peak=float(np.sqrt(np.mean((q[:,7:]-source[:,7:])**2,axis=1)).max())
 assert peak<.19
 for col in [left,right]:
  lo,hi=limits[col-7]; assert (q[:,col]>=lo).all() and (q[:,col]<=hi).all()
 target=OUT/f'prepared_arm{int(amount*100):02d}.npz'
 with target.open('xb') as stream:np.savez_compressed(stream,**dict(original,control_q=q))
 report=dict(parent=str(parent),parent_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),
   source_sha256=original['source_sha256'].item(),arm_roll_offset_rad=amount,
   start_frame=500,end_frame=580,peak_source_joint_rmse_rad=peak,
   other_configuration_coordinates_unchanged=True,original_audit_retained=True)
 target.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
