"""Same-state acceleration-cost probe; no physics or new landings."""
import json
from pathlib import Path
import numpy as np
import torch
from fabrics_sim.solver.admm_qp import QPData
from fabrics_sim.solver.daqp_backend import DAQPBackend
root=Path(__file__).resolve().parent
source=root.parent/'continue_20261114_0338'
rows=[]
for file in sorted(source.glob('baseline_snapshots.qp_*.npz')):
 with np.load(file) as a:
  p={k:torch.from_numpy(a['problem_'+k].copy()) for k in ('H','h','G','b','lb','ub','A_eq','b_eq')}
  target=-p['h'][0,:3].double()/100
  np.testing.assert_allclose(p['H'][0,:35,:35].numpy(), np.diag([100.]*6+[1.+1e-6]*29), atol=2e-5, rtol=0)
  for weight in (100,300,1000,3000):
   trial={k:v.clone() for k,v in p.items()}
   change=weight-100
   trial['H'][0,:3,:3]+=change*torch.eye(3)
   trial['h'][0,:3]-=(change*target).float()
   solver=DAQPBackend(single_solve=True); solution=solver.solve(QPData(**trial))
   assert solver.last_solve_counts==[1]
   z=solution.x if hasattr(solution,'x') else solution.z
   residual=max(0.,float((trial['G'].double()@z[...,None]).squeeze(-1).sub(trial['b']).max()), float((trial['A_eq'].double()@z[...,None]).squeeze(-1).sub(trial['b_eq']).abs().max()), float((trial['lb']-z).max()), float((z-trial['ub']).max()))
   if bool(solution.status_ok.item()): assert residual <= 1e-6, residual
   rows.append(dict(time_s=float(a['time']),linear_weight=weight,max_constraint_violation=residual,accepted=bool(solution.status_ok.item()),target=target.tolist(),acceleration=z[0,:6].tolist()))
(root/'translation_probe.json').write_text(json.dumps(rows,indent=2)+'\n')
for row in rows:print(row)
