"""Audit measured root drift and knee state; samples do not establish causality."""
import json
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parent
baseline=root.parent/'continue_20261114_0338/baseline_traced.failure.npz'
rows=[]
for path in [baseline,*sorted(root.glob('*.npz'))]:
 with np.load(path) as a:
  if 'history_qdd_qp_reference' not in a:continue
  t=a['history_time'];q=a['history_q'][:,0];v=a['history_qd'][:,0]
  raw=a['history_reference_q'][:,0];reference=a['history_whole_body_reference_q'][:,0]
  error=q[:,:3]-raw[:,:3]
  mask=t>=.9
  if not np.any(mask):continue
  commanded=a['history_commanded_mode'].reshape(-1)
  support=np.c_[(commanded&1)!=0,(commanded&2)!=0]
  knees=q[:,[10,16]]
  source_knees=raw[:,[10,16]]
  rows.append(dict(record=str(path.relative_to(root.parent)),last_history_time_s=float(t[-1]),
    peak_raw_root_error_m=float(np.linalg.norm(error,axis=1).max()),
    peak_joint_speed_after_09_rad_s=float(abs(v[mask,6:]).max()),
    minimum_commanded_support_knee_after_09_rad=float(knees[support & mask[:,None]].min()),
    samples=[dict(time_s=float(t[i]),source_root_error_xyz_m=error[i].tolist(),
      control_root_error_xyz_m=(q[i,:3]-reference[i,:3]).tolist(),
      projected_base_acceleration=a['history_qdd_projected'][i,0,:6].tolist(),
      qp_base_acceleration=a['history_qdd_qp'][i,0,:6].tolist(),
      knee_rad=knees[i].tolist(),source_knee_rad=source_knees[i].tolist(),
      mode=int(commanded[i])) for target in (.9,1.,1.1,1.2,1.25,1.28,1.3,1.34)
      if target<=t[-1] for i in [int(np.argmin(abs(t-target)))]]))
(root/'drift_audit.json').write_text(json.dumps(dict(reference='Unchanged original-source pose in history_reference_q; commanded contact masks, not a no-slip certificate.',trials=rows),indent=2)+'\n')
for row in rows:print(row['record'],row['last_history_time_s'],row['peak_raw_root_error_m'],row['minimum_commanded_support_knee_after_09_rad'])
