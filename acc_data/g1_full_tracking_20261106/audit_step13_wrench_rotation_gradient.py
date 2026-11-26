"""Compare the offline wrench-cone rotation derivative with central differences."""
from pathlib import Path
import ast,json,hashlib
import numpy as np,torch
from scipy.spatial.transform import Rotation
from mujoco_reference_dynamics import ReferenceDynamics
from full_effort_linearization import perturb
from fabrics_sim.solver.AccelerationQP import build_single_support_wrench_constraints
p=Path(__file__).parent;script=p/'fit_step13_whole_body_geometric.py'
model=ReferenceDynamics('src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf',np.load(p/'rolling_contact.npz')['final_joint_order'])
ns=dict(model=model,np=np,torch=torch,Rotation=Rotation,build_single_support_wrench_constraints=build_single_support_wrench_constraints)
for node in ast.parse(script.read_text()).body:
 if isinstance(node,ast.FunctionDef) and node.name in ['wrench_rows','skew']:
  exec(compile(ast.Module(body=[node],type_ignores=[]),str(script),'exec'),ns)
trajectory=np.load(p/'step13_whole_body_hard_seed.npz')['rejected_or_unexecuted_control_q'];rng=np.random.default_rng(13);errors=[]
for frame in [960,973,1008,1020]:
 q=trajectory[frame];state=model.evaluate(q,np.zeros(35))
 for foot in (0,1):
  G,b=ns['wrench_rows'](q,foot);w=np.r_[rng.normal(size=2)*20,350.,rng.normal(size=3)*5]
  direction=rng.normal(size=35);direction[3:6]=0.;direction/=np.linalg.norm(direction)
  omega=state['jacobian'][foot][3:]@direction
  expected=(G[:,:3]@ns['skew'](w[:3])+G[:,3:]@ns['skew'](w[3:]))@omega
  plus,_=ns['wrench_rows'](perturb(q,1e-6*direction),foot);minus,_=ns['wrench_rows'](perturb(q,-1e-6*direction),foot)
  observed=(plus-minus)@w/2e-6;errors.append(dict(frame=frame,foot=foot,maximum_absolute_error=float(abs(observed-expected).max())))
report=dict(script_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),cases=errors,passed=all(e['maximum_absolute_error']<1e-7 for e in errors),physical_tracking_certified=False)
(p/'step13_wrench_rotation_gradient.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
assert report['passed']
