"""Read-only checks of saved QP torque and the immediately following physics step."""
import json
from pathlib import Path
import numpy as np
root=Path(__file__).parent
rows=[]
for p in sorted(root.glob('*.response_*.npz')):
 qp=p.with_name(p.name.replace('.response_', '.qp_'))
 if not qp.exists():continue
 d=np.load(qp);r=np.load(p)
 actual=(r['qd'][0]-d['qd'][0])/float(r['dt']);pred=d['predicted_acc'][0]
 rows.append(dict(run=p.name.split('.response_')[0],time_s=float(d['time']),frame=float(d['motion_frame_position']),
  applied_torque_max_error_Nm=float(abs(r['applied_torque']-d['total_torque']).max()),
  predicted_root_acceleration=pred[:6].tolist(),observed_root_acceleration=actual[:6].tolist(),
  root_linear_error_norm=float(np.linalg.norm(actual[:3]-pred[:3])),root_angular_error_norm=float(np.linalg.norm(actual[3:6]-pred[3:6])),
  joint_acceleration_error_rmse=float(np.sqrt(np.mean((actual[6:]-pred[6:])**2)))))
(root/'applied_response_audit.json').write_text(json.dumps(dict(scope='Immediate saved response only; finite differences include contact impacts and discretization.',samples=rows),indent=2)+'\n')
for r in rows:print(r['run'],r['time_s'],'torque error',r['applied_torque_max_error_Nm'],'root angular error',r['root_angular_error_norm'])
