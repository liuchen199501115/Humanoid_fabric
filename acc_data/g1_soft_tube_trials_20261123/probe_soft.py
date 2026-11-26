"""Frozen failed baseline QP: retain every physical row and soften only three tubes."""
import json, math
from pathlib import Path
import numpy as np
import torch
from fabrics_sim.solver.admm_qp import QPData
from g1_soft_tracking_tube import SoftTrackingTubeBackend
from fabrics_sim.solver.qp_refinement import _violation
OUT=Path(__file__).resolve().parent
with np.load(OUT.parent/'g1_both_paths_20261123/baseline.failure.npz') as a:
    p={k:torch.tensor(a['problem_'+k],dtype=torch.float64) for k in ('H','h','G','b','lb','ub','A_eq','b_eq')}
C=p['G'][:,-3:,:35].clone();d=p['b'][:,-3:].clone()
assert not p['G'][:,-3:,35:].any()
p['G']=p['G'][:,:-3];p['b']=p['b'][:,:-3]
rows=[]
for weight in (.1,1.,10.,100.):
    solver=SoftTrackingTubeBackend(weight)
    solver.tube=C,d,torch.tensor([.195**2,.195**2,math.sin(.17)**2],dtype=torch.float64)
    result=solver.solve(QPData(**p));z=result.z[0].numpy()
    row=dict(weight=weight,certified=bool(result.status_ok.all()),status=solver.last_status,
        physical_violation=_violation({k:v[0].numpy() for k,v in p.items()},z),
        tube_residual=((C@result.z[:,:35,None]).squeeze(-1)-d)[0].tolist(),
        normalized_slack=solver.last_slack[0].tolist(),root_angular_acceleration=z[3:6].tolist(),
        native_solve_count=solver.last_solve_counts)
    rows.append(row)
(OUT/'frozen_soft.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps(rows,indent=2))
