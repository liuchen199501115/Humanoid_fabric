from pathlib import Path
import json,numpy as np
from mujoco_reference_dynamics import ReferenceDynamics
from generalized_dynamics_linearization import perturb
p=Path(__file__).parent;saved=np.load(p/'wrench_fixed_ten.npz');model=ReferenceDynamics('src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf',saved['final_joint_order']);rng=np.random.default_rng(812);reports=[]
for t in [3.5,3.7,3.8]:
 d=np.load(p/f'fixed_clock120_body10_preview_snapshots.snapshot_{t:.3f}.npz');q=d['q'][0].astype(float);v=d['v'][0].astype(float)
 for sides in [[0],[1],[0,1]]:
  impulse=np.array([.5,-.2,6.,.01,.02,-.03,.8,.1,5.,.01,-.02,.01]);delta_impulse=rng.normal(size=12)*1e-5
  def jump(q,impulse):
   state=model.evaluate(q,v);rhs=sum((state['jacobian'][j].T@impulse[6*j:6*j+6] for j in sides),start=np.zeros(35));return np.linalg.solve(state['mass'],rhs)
  derivative=np.zeros((35,35));epsilon=1e-5
  for coordinate in range(35):
   delta=np.zeros(35);delta[coordinate]=epsilon
   derivative[:,coordinate]=(jump(perturb(q,delta),impulse)-jump(perturb(q,-delta),impulse))/(2*epsilon)
  dq=rng.normal(size=35)*1e-5;actual=jump(perturb(q,dq),impulse+delta_impulse)-jump(q,impulse);predicted=derivative@dq+jump(q,delta_impulse)
  row=dict(time=t,sides=sides,relative_error=float(np.linalg.norm(actual-predicted)/np.linalg.norm(actual)));reports.append(row)
print(reports);assert max(r['relative_error'] for r in reports)<1e-3
(p/'impact_map_linearization_audit.json').write_text(json.dumps(reports,indent=2)+'\n')
