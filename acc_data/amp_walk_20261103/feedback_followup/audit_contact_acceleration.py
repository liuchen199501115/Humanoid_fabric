from pathlib import Path
import json,numpy as np
from scipy.spatial.transform import Rotation
from mujoco_reference_dynamics import ReferenceDynamics
from generalized_dynamics_linearization import perturb
from centroidal_pose_linearization import skew
p=Path(__file__).parent;saved=np.load(p/'wrench_fixed_ten.npz');m=ReferenceDynamics('src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf',saved['final_joint_order']);reports=[]
for t in [3.5,3.7,3.8]:
 d=np.load(p/f'fixed_clock120_body10_preview_snapshots.snapshot_{t:.3f}.npz');q=d['q'][0].astype(float);v=d['v'][0].astype(float);s=m.evaluate(q,v,kinematic_bias=True);eps=1e-5;plus=m.evaluate(perturb(q,eps*v),v);minus=m.evaluate(perturb(q,-eps*v),v);origin_fd=((plus['jacobian']-minus['jacobian'])/(2*eps))@v
 errors=[]
 for side in [0,1]:
  local=np.array([.10,-.03,-.04]);R=Rotation.from_quat(s['feet'][side,3:]).as_matrix();offset=R@local;omega=s['jacobian'][side,3:]@v
  predicted=s['foot_bias'][side,:3]-skew(offset)@s['foot_bias'][side,3:]+np.cross(omega,np.cross(omega,offset))
  J=[]
  for sample in [plus,minus]:
   rot=Rotation.from_quat(sample['feet'][side,3:]).as_matrix();j=sample['jacobian'][side];J.append(j[:3]-skew(rot@local)@j[3:])
  point_fd=(J[0]-J[1])@v/(2*eps);errors.append(float(abs(predicted-point_fd).max()))
 row=dict(time=t,origin_bias_max_error=float(abs(s['foot_bias']-origin_fd).max()),point_bias_max_error=max(errors));reports.append(row);print(row)
assert max(max(r['origin_bias_max_error'],r['point_bias_max_error']) for r in reports)<1e-5
(p/'contact_acceleration_bias_audit.json').write_text(json.dumps(reports,indent=2)+'\n')
