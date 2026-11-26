"""Re-solve saved original QPs under alternate numerical settings, offline only."""
import json,sys,numpy as np,torch
from pathlib import Path
from fabrics_sim.solver.admm_qp import QPData
from fabrics_sim.solver.daqp_backend import DAQPBackend
p=Path(__file__).resolve().parent;name=sys.argv[1];d=np.load(p/(name+'.failure.npz'));fields=['H','h','G','b','lb','ub','A_eq','b_eq'];prefix='soft_problem_' if 'soft_problem_H' in d else 'problem_';qp=QPData(**{k:torch.tensor(d[prefix+k],dtype=torch.float64) for k in fields});rows=[]
for cap in [1.,3.,10.,30.]:
 for prox in [0.,1e-4,1e-3]:
  solver=DAQPBackend(single_solve=True,single_proximal=prox,equilibration_cap=cap);r=solver.solve(qp);rows.append(dict(cap=cap,proximal=prox,valid=bool(r.status_ok.all()),status=solver.last_status,iterations=int(r.iters),native_attempts=solver.last_solve_counts))
(p/(name+'.frozen_numerics.json')).write_text(json.dumps(rows,indent=2)+'\n');print(rows)
