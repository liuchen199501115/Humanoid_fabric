"""Offline objective sensitivity of the saved single QP."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from fabrics_sim.solver.daqp_backend import DAQPBackend
OUT=Path(__file__).resolve().parent
h=np.load(OUT/'eight_snapshot.npz');records=[]
for path in sorted(OUT.glob('eight_snapshot.snapshot_*.npz')):
 d=np.load(path);arrays={k:d['soft_problem_'+k].copy() for k in ['H','h','G','b','lb','ub','A_eq','b_eq']}
 J=arrays['A_eq'][0,:3,:35]/arrays['A_eq'][0,0,0]
 i=abs(h['history_time']-float(d['time'])).argmin();bias=h['history_com_curvature'][i,0]
 np.testing.assert_allclose(J@d['soft_problem_solution'][0,:35]+bias,h['history_com_acc_qp_predicted'][i,0],atol=1e-4)
 target=h['history_com_acc_original_desired'][i,0]-bias
 record=dict(time_s=float(d['time']),cases={})
 for scale in [0,1,3,15,63]:
  a={k:v.copy() for k,v in arrays.items()}
  W=np.diag([1000.,1000.,80.])*scale
  a['H'][0,:35,:35]+=J.T@W@J;a['h'][0,:35]-=J.T@W@target
  solver=DAQPBackend(single_solve=True,single_proximal=.0001)
  r=solver.solve(SimpleNamespace(**{k:torch.tensor(v,dtype=torch.float64) for k,v in a.items()}));z=r.z.numpy()[0]
  record['cases'][str(scale)]=dict(valid=bool(r.converged.item()),com_acceleration=(J@z[:35]+bias).tolist(),
    max_joint_qdd=float(abs(z[6:35]).max()),joint_tube_slack=float(z[76]))
 records.append(record)
(OUT/'com_weight.json').write_text(json.dumps(records,indent=2)+'\n');print(json.dumps(records,indent=2))
