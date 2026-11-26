from pathlib import Path
import numpy as np,json
from scipy.optimize import linprog
p=Path(__file__).parent;d=np.load(p/'fixed_clock120_body10_turn_recovery40_comfirst.failure.npz');out={}
for prefix in ['priority_level_0_','priority_level_1_','problem_']:
 A=d[prefix+'A_eq'][0].astype(float);b=d[prefix+'b_eq'][0].astype(float);G=d[prefix+'G'][0].astype(float);h=d[prefix+'b'][0].astype(float);lo=d[prefix+'lb'][0];hi=d[prefix+'ub'][0];bounds=list(zip(lo,hi));n=len(lo);C=d['task_acceleration_C'][0].astype(float);padded=np.pad(C[-3:],((0,0),(0,n-35)))
 matching=np.array([min(np.linalg.norm(row-padded,axis=1))<1e-4 for row in G])
 for scope in ['all','without_auxiliary_tubes']:
  chosen=np.ones(len(G),bool) if scope=='all' else ~matching;gg=G[chosen];hh=h[chosen];gs=np.maximum(np.linalg.norm(gg,axis=1),1e-10);as_=np.maximum(np.linalg.norm(A,axis=1),1e-10)
  r=linprog(np.zeros(n),A_ub=gg/gs[:,None],b_ub=hh/gs,A_eq=A/as_[:,None],b_eq=b/as_,bounds=bounds,method='highs');result=dict(status=r.message,success=bool(r.success),removed_rows=int((~chosen).sum()))
  if r.success:result['maximum_original_units_violation']=float(max(np.max(gg@r.x-hh),np.max(abs(A@r.x-b)),0.))
  out[prefix+scope]=result
(p/'turn_priority_feasibility_audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
