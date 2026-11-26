from pathlib import Path
import numpy as np,json
from preview_collision import CollisionDynamics
from preview_manifold import integrate
p=Path(__file__).parent;name='fixed_clock120_body10_full_body_preview42_semi_process24';d=np.load(p/(name+'.full_body_preview.failure_plan.npz'),allow_pickle=True);names=np.load(p/'wrench_fixed_ten.npz')['final_joint_order'];m=CollisionDynamics('src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf',names)
s=m.evaluate(d['q'],d['v'],kinematic_bias=True)['collision'];active=s['distance']<.4;C=-s['jacobian'][active];b=s['bound'][active];G=d['problem_G'];gb=d['problem_b'];matches=np.argmin(np.linalg.norm(C[:,None,:]-G[None,:,:35],axis=2),axis=1);errors=abs(C-G[matches,:35]);berror=abs(b-gb[matches]);eps=1e-5
plus=m.evaluate(integrate(d['q'],eps*d['v']),d['v'],kinematic_bias=True)['collision'];minus=m.evaluate(integrate(d['q'],-eps*d['v']),d['v'],kinematic_bias=True)['collision'];fd=((plus['jacobian']-minus['jacobian'])/(2*eps))@d['v'];prior=np.load(p/(name+'.full_body_preview.failure_previous_plan.npz'),allow_pickle=True)
r=dict(total_pairs=len(s['distance']),active_pairs=int(active.sum()),matched_actual_rows=matches.tolist(),jacobian_max_error=float(errors.max()),bound_max_error=float(berror.max()),analytic_bias_fd_error=float(abs(fd-s['bias']).max()),previous_next_control_barrier_violation=float((C@prior['controls'][1,:35]-b).max()))
print(r);(p/'preview_collision_equivalence.json').write_text(json.dumps(r,indent=2)+'\n')
assert errors.max()<1e-5
assert berror.max()<.1
assert abs(fd-s['bias']).max()<1e-4
