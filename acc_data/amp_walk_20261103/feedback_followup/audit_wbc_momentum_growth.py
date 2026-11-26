"""Compare observed whole-body angular momentum against the same q/v reference."""
from pathlib import Path
import json
import numpy as np
from scipy.spatial.transform import Rotation
from numpy_g1_com import CenterOfMassModel
p=Path(__file__).parent
model_data=np.load(p/'wrench_fixed_ten.npz')
model=CenterOfMassModel('src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf',model_data['final_joint_order'])
d=np.load(p/'full_motion_no_mpc.npz')
times=d['history_time'].reshape(-1)
rows=[]
for time in [3.6,3.65,3.7,3.8,3.9,3.95,4.,4.04,4.08]:
 i=int(abs(times-time).argmin());q=d['history_q'][i].reshape(36);v=d['history_qd'][i].reshape(35)
 rq=d['history_whole_body_reference_q'][i].reshape(36);rv=d['history_whole_body_reference_v'][i].reshape(35)
 observed=model.momentum(q,v);reference=model.momentum(rq,rv)
 root_error=(Rotation.from_quat(q[3:7])*Rotation.from_quat(rq[3:7]).inv()).as_rotvec()
 rows.append(dict(time=float(times[i]),source_frame=float(d['history_frame'][i]),angular_momentum_error=(observed-reference).tolist(),root_orientation_error_world=root_error.tolist(),angular_velocity_error_world=(v[3:6]-rv[3:6]).tolist()))
(p/'wbc_momentum_growth_audit.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps(rows[-4:],indent=2))
