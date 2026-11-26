"""Instantaneous feasibility of the COM target with full saved constraints."""
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
OUT=Path(__file__).resolve().parent
h=np.load(OUT/'eight_snapshot.npz');results=[]
for path in sorted(OUT.glob('eight_snapshot.snapshot_*.npz')):
 d=np.load(path);a={k:d['soft_problem_'+k][0] for k in ['G','b','A_eq','b_eq','lb','ub']}
 i=abs(h['history_time']-float(d['time'])).argmin();n=len(a['lb'])
 J=a['A_eq'][:3,:35]/a['A_eq'][0,0]
 target=h['history_com_acc_original_desired'][i,0]-h['history_com_curvature'][i,0]
 extra=np.zeros((3,n));extra[:,:35]=J
 record=dict(time_s=float(d['time']),desired_com_acceleration=h['history_com_acc_original_desired'][i,0].tolist(),cases={})
 for name,rows in [('xyz',[0,1,2]),('y',[1]),('xy',[0,1])]:
  E=np.vstack((a['A_eq'],extra[rows]));e=np.r_[a['b_eq'],target[rows]]
  es=np.maximum(abs(E).max(axis=1),1e-12);gs=np.maximum(abs(a['G']).max(axis=1),1e-12)
  r=linprog(np.zeros(n),A_ub=a['G']/gs[:,None],b_ub=a['b']/gs,A_eq=E/es[:,None],b_eq=e/es,bounds=list(zip(a['lb'],a['ub'])),method='highs')
  record['cases'][name]=dict(feasible=r.success,status=r.status)
 results.append(record)
(OUT/'com_feasibility.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(results,indent=2))
