import json
from pathlib import Path
import numpy as np,torch
from fabrics_sim.solver.daqp_backend import DAQPBackend
from fabrics_sim.solver.admm_qp import QPData
p=Path('acc_data/amp_walk_20261102/continuation');reports=[]
for path in sorted(p.glob('baseline.qp_*.npz')):
 d=np.load(path);fields={k:torch.tensor(d['priority_level_0_'+k],dtype=torch.float64) for k in ['H','h','G','b','lb','ub','A_eq','b_eq']}
 C=torch.zeros((1,1,76),dtype=torch.float64);C[:,:,:35]=torch.tensor(d['task_center_of_mass_A'][:,1:2])
 y=torch.tensor(d['task_center_of_mass_desired'][:,1:2]-d['task_center_of_mass_bias'][:,1:2],dtype=torch.float64)
 fields['A_eq']=torch.cat((fields['A_eq'],C),dim=1);fields['b_eq']=torch.cat((fields['b_eq'],y),dim=1)
 solver=DAQPBackend();sol=solver.solve(QPData(**fields));x=sol.z[0].numpy();orig=d['problem_solution'][0]
 row=dict(time=float(d['time']),solved=bool(sol.converged.all()),status=solver.last_status,
   max_abs_acceleration=float(abs(x[:35]).max()),original_max_abs_acceleration=float(abs(orig[:35]).max()),
   task_residuals={})
 for key in d.files:
  if not key.startswith('task_') or not key.endswith('_A'):continue
  name=key[5:-2];A=d[key][0];b=d['task_'+name+'_bias'][0]-d['task_'+name+'_desired'][0];active=d['task_'+name+'_active'][0]
  row['task_residuals'][name]={'original':((A@orig[:35]+b)*active).round(4).tolist(),'exact_lateral_com':((A@x[:35]+b)*active).round(4).tolist()}
 print(json.dumps({k:v for k,v in row.items() if k!='task_residuals'}));reports.append(row)
(p/'com_tradeoff.json').write_text(json.dumps(reports,indent=2)+'\n')
