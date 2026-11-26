"""Independent saved-problem solve; no counterfactual solution is executed."""
from pathlib import Path
import json,numpy as np
from scipy import sparse
from scipy.optimize import linprog
from solve_upper_fit_qp import solve
p=Path(__file__).parent;d=np.load(p/'crouch_descent_swing_priority.failure.npz');reports=[]
for prefix in ['priority_problem_','problem_']:
 data={k:d[prefix+k][0].astype(float) for k in ['H','h','G','b','lb','ub','A_eq','b_eq']};H=data['H'];n=len(H);G=data['G'];b=data['b'];E=data['A_eq'];e=data['b_eq'];lb=data['lb'];ub=data['ub']
 C=sparse.csc_matrix(np.vstack((E,G,np.eye(n))));lo=np.r_[e,np.full(len(G),-np.inf),lb];hi=np.r_[e,b,ub]
 x,report=solve(sparse.csc_matrix((H+H.T)/2),data['h'],C,lo,hi)
 feasibility=linprog(np.zeros(n),A_ub=G,b_ub=b,A_eq=E,b_eq=e,bounds=list(zip(lb,ub)),method='highs')
 row=dict(problem=prefix,clarabel=report,linear_feasibility=dict(success=bool(feasibility.success),status=int(feasibility.status),message=feasibility.message),last_rows=[dict(nonzero=np.flatnonzero(abs(r)>1e-10).tolist(),bound=float(z)) for r,z in zip(G[-3:],b[-3:])])
 if feasibility.success:
  xx=feasibility.x;row['linear_feasibility']['constraint_residual']=float(max(abs(E@xx-e).max(),(G@xx-b).max(),(lb-xx).max(),(xx-ub).max(),0.))
 if prefix=='priority_problem_' and all(np.all(np.flatnonzero(abs(r)>1e-10)<6) for r in G[-2:]):
  test=linprog(np.zeros(n),A_ub=G[:-2],b_ub=b[:-2],A_eq=E,b_eq=e,bounds=list(zip(lb,ub)),method='highs')
  row['diagnostic_without_last_two_root_tube_rows']=dict(success=bool(test.success),status=int(test.status),scope='Counterfactual diagnostic only; never executed or used to relax original acceptance')
 reports.append(row);print(json.dumps(row),flush=True)
(p/'swing_priority_failure_audit.json').write_text(json.dumps(reports,indent=2)+'\n')
