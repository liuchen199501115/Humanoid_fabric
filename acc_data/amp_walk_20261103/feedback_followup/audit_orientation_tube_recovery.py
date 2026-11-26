from pathlib import Path
from types import SimpleNamespace
import json,math,torch
from tracking_tube import tracking_tube_constraints
from tube_recovery import solve_with_orientation_tube_recovery
q=torch.zeros(1,36,dtype=torch.float64);q[:,6]=1.;ref=q.clone();q[:,3]=math.sin(.1);q[:,6]=math.cos(.1);v=torch.zeros(1,35,dtype=q.dtype);v[:,3]=.3;zero=torch.zeros_like(v)
C,d,diag=tracking_tube_constraints(q,v,ref,zero,zero)
C30,d30,_=tracking_tube_constraints(q,v,ref,zero,zero,orientation_bandwidth=30.)
expected=d.clone();expected[:,-1]+=40*diag[:,2,1]+800*diag[:,2,0]
assert torch.equal(C,C30) and torch.equal(d[:,:2],d30[:,:2])
assert torch.allclose(expected,d30,rtol=0.,atol=1e-12)
physical_C=torch.randn(1,2,35,dtype=q.dtype);physical_d=torch.randn(1,2,dtype=q.dtype)
class FakeSolver:
 def __init__(self,outcomes):self.outcomes=iter(outcomes);self.calls=[]
 def solve(self,**kw):self.calls.append(kw);return SimpleNamespace(validated=torch.tensor([next(self.outcomes)]))
 def reset(self):pass
kwargs=dict(acceleration_constraints=(torch.cat((physical_C,C),1),torch.cat((physical_d,d),1)),task_requests=[],priority_tasks=())
for outcomes in [[True],[False,False,True],[False,False,False,True]]:
 solver=FakeSolver(outcomes);result=solve_with_orientation_tube_recovery(solver,tube_C=C,tube_d=d,tube_diagnostics=diag,**kwargs)
 assert bool(result.validated.all()) and len(solver.calls)==len(outcomes)
 for call in solver.calls:
  cc,dd=call['acceleration_constraints'];assert torch.equal(cc[:,:2],physical_C) and torch.equal(dd[:,:2],physical_d)
 if len(outcomes)==1:assert solver.calls[0]['acceleration_constraints'] is kwargs['acceleration_constraints']
 else:assert solver.orientation_tube_recovery_bandwidth==30.
report=dict(algebra_matches_direct_recomputation=True,physical_constraints_unchanged=True,successful_first_solve_unchanged=True,recovery_requires_validated_solution=True,checks_passed=True,physical_tracking_certified=False)
Path(__file__).with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
