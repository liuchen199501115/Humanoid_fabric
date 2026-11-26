"""Offline single-attempt numerical replay, keeping the original QP and certificates."""
import json
from pathlib import Path
import numpy as np,torch
from fabrics_sim.solver.admm_qp import QPData
from fabrics_sim.solver.daqp_backend import DAQPBackend
OUT=Path(__file__).resolve().parent;rows=[]
for name in ['eleven_tube1000_ori32','eleven_tube1000_pelvis4','eleven_speed24_tube10000']:
 a=np.load(OUT/(name+'.failure.npz'));fields=['H','h','G','b','lb','ub','A_eq','b_eq'];qp=QPData(**{k:torch.tensor(a['soft_problem_'+k]) for k in fields})
 for cap in [1.,2.,5.,10.,100.]:
  solver=DAQPBackend(50000,single_solve=True,single_proximal=.0001,equilibration_cap=cap);s=solver.solve(qp);r=dict(name=name,cap=cap,certified=bool(s.status_ok.all()),status=solver.last_status,attempts=solver.last_solve_counts);rows.append(r);print(r,flush=True)
(OUT/'step12_scaling_replay.json').write_text(json.dumps(rows,indent=2)+'\n')
