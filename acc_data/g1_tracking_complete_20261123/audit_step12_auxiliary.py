"""Offline frozen-state sensitivity; never changes the running controller."""
import json
from pathlib import Path
import numpy as np, torch
from fabrics_sim.solver.admm_qp import QPData
from fabrics_sim.solver.daqp_backend import DAQPBackend
OUT=Path(__file__).resolve().parent
a=np.load(OUT/'eleven_upper64_hip20.snapshot_4.520.npz');fields=['H','h','G','b','lb','ub','A_eq','b_eq'];rows=[]
for mode in ['original','position','orientation','both']:
 p={k:a['soft_problem_'+k].astype(float).copy() for k in fields}
 for idx in ([-2] if mode=='position' else [-1] if mode=='orientation' else [-2,-1] if mode=='both' else []):p['G'][:,idx]=0;p['b'][:,idx]=0
 solver=DAQPBackend(50000,single_solve=True,single_proximal=.0001)
 sol=solver.solve(QPData(**{k:torch.tensor(v) for k,v in p.items()}));ok=bool(sol.status_ok.all());r=dict(mode=mode,certified=ok)
 if ok:
  x=sol.z.numpy()[0];r['root_acceleration']=x[:6].tolist();r['wrench']=x[35:47].tolist()
  for n in ['center_of_mass','joint_posture','feet_pose']:
   prefix='composed_task_'+n+'_';res=a[prefix+'A'][0]@x[:35]+a[prefix+'bias'][0]-a[prefix+'desired'][0];r[n+'_residual']=res.tolist()
 rows.append(r);print(mode,ok,r.get('root_acceleration'),flush=True)
(OUT/'step12_auxiliary_sensitivity.json').write_text(json.dumps(rows,indent=2)+'\n')
