from pathlib import Path
import numpy as np,json
from generalized_dynamics_linearization import perturb,residual
from preview_manifold import difference
from mujoco_reference_dynamics import ReferenceDynamics
p=Path(__file__).parent;d=np.load(p/'fixed_clock120_body10_full_body_preview42_semi_process24.full_body_preview.failure_previous_plan.npz');m=ReferenceDynamics('src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf',np.load(p/'wrench_fixed_ten.npz')['final_joint_order']);q=d['physical_q'][1].astype(np.float32);v=d['physical_v'][1];a=d['controls'][1,:35];w=d['controls'][1,35:47];results=[]
for dtype in [np.float32,np.float64]:
 columns=[];steps=[];position=q.astype(dtype)
 for j in range(35):
  delta=np.eye(35)[j]*1e-5;plus=perturb(position,delta);minus=perturb(position,-delta);steps.append(difference(plus,minus)/2e-5)
  columns.append((residual(m.evaluate(plus,v),a,w)-residual(m.evaluate(minus,v),a,w))/2e-5)
 results.append((np.array(columns).T,np.array(steps).T))
report=dict(float32_max_tangent_step_error=float(abs(results[0][1]-np.eye(35)).max()),float64_max_tangent_step_error=float(abs(results[1][1]-np.eye(35)).max()),inverse_dynamics_gradient_max_error=float(abs(results[0][0]-results[1][0]).max()),inverse_dynamics_gradient_relative_norm=float(np.linalg.norm(results[0][0]-results[1][0])/np.linalg.norm(results[1][0])))
print(report);(p/'preview_precision_audit.json').write_text(json.dumps(report,indent=2)+'\n')
