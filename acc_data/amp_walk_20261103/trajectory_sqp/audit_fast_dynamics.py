from pathlib import Path
import numpy as np,json,time
p=Path(__file__).parent
ns={'__file__':str(p/'prepare_spline_persistent.py'),'__name__':'fast_audit'}
s=(p/'prepare_spline_persistent.py').read_text().split('for iteration in range(args.iterations):')[0]
exec(compile(s,str(p/'prepare_spline_persistent.py'),'exec'),ns)
exec((p/'fast_generalized_dynamics.inc').read_text(),ns)
from fast_snapshot import snapshot as fast_snapshot
candidate=np.load(p/'persistent_control.npz');q,v,a=[candidate[k] for k in ('q','v','a')]
t0=time.perf_counter();normal=ns['snapshot'](q);normal_time=time.perf_counter()-t0
t0=time.perf_counter();fast=fast_snapshot(ns['model'],q);fast_time=time.perf_counter()-t0
errors={k:float(abs(fast[k]-normal[k]).max()) for k in fast}
ns['update_support'](normal)
t0=time.perf_counter();M,h,J,_=ns['dynamics'](normal,q,v,a);old_time=time.perf_counter()-t0
t0=time.perf_counter();g,fastJ=ns['generalized_dynamics'](fast,q,v,a);new_time=time.perf_counter()-t0
report=dict(geometry_error=errors,force_error=float(abs(g-((M@a[...,None])[...,0]+h)).max()),contact_jacobian_error=float(abs(fastJ-J).max()),snapshot_times=[normal_time,fast_time],generalized_force_times=[old_time,new_time])
assert max(errors.values())<1e-12 and report['force_error']<1e-8 and report['contact_jacobian_error']<1e-12,report
(p/'fast_dynamics_audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
