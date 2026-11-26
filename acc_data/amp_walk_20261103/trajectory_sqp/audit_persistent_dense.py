from pathlib import Path
import json,numpy as np,torch
from scipy.interpolate import BSpline
p=Path(__file__).parent
ns={'__file__':str(p/'prepare_spline_persistent.py'),'__name__':'dense_audit'}
s=(p/'prepare_spline_persistent.py').read_text().split('for iteration in range(args.iterations):')[0]
exec(compile(s,str(p/'prepare_spline_persistent.py'),'exec'),ns)
d=np.load(p/'persistent_control.npz');frames=np.arange(0,float(d['frames'][-1])+.0001,.4);N=len(frames)
f=ns['G1PoseFabric'](N,'cpu',robot_name='g1_29dof_rev_1_0',task_profile='dynamics-aware')
coeff=d['coefficients'];basis=BSpline(d['knots'],np.eye(len(coeff)),int(d['degree']))
ns.update(N=N,model=f.dynamics_tasks.model,B=basis(frames),D=basis(frames,1)*200,E=basis(frames,2)*200**2,collision_cut_sets=[set() for _ in range(N)])
q,v,a=ns['sample'](coeff);state=ns['snapshot'](q);audit=ns['collision_linearization'](state,q,v,a)
r=np.where(audit['active'],audit['residual'],np.inf);i,j=np.unravel_index(np.argmin(r),r.shape)
report=dict(samples=N,rate_hz=500,minimum_collision_barrier_residual=float(r[i,j]),worst_source_frame=float(frames[i]),worst_pair=int(j),minimum_collision_cover_distance=float(audit['distance'][:,ns['first_pair']:].min()),peak_joint_speed=float(abs(v[:,6:]).max()),peak_joint_acceleration=float(abs(a[:,6:]).max()),full_tracking_passed=False)
(p/'persistent_dense_audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
