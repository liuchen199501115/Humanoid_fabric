from pathlib import Path
import json,numpy as np
from scipy.optimize import linprog
import daqp
root=Path(__file__).parent
rows=[]
for path in sorted(root.glob('four_strict_diagnostic.qp_*.npz')):
 d=np.load(path);p={k:d['priority_problem_'+k][0].astype(float) for k in ['H','h','G','b','lb','ub','A_eq','b_eq']};n=len(p['h'])
 A=d['task_center_of_mass_A'][0].astype(float);M=d['task_center_of_mass_metric'][0].astype(float);target=d['task_center_of_mass_desired'][0].astype(float)-d['task_center_of_mass_bias'][0];bias=d['task_center_of_mass_bias'][0]
 Af=np.c_[A,np.zeros((3,n-35))];x=d['problem_solution'][0].astype(float)
 row=dict(time=float(d['time']),desired=(target+bias).tolist(),baseline=(Af@x+bias).tolist(),single_axis_ranges=[],reweighted=[])
 for axis in range(3):
  limits=[]
  for sign in [1.,-1.]:
   lp=linprog(sign*Af[axis],A_ub=p['G'],b_ub=p['b'],A_eq=p['A_eq'],b_eq=p['b_eq'],bounds=list(zip(p['lb'],p['ub'])),method='highs')
   limits.append(float(Af[axis]@lp.x+bias[axis]) if lp.success else lp.message)
  row['single_axis_ranges'].append(limits)
 for scale in [1.,2.,4.,10.,100.]:
  H=p['H']+2*(scale-1)*Af.T@M@Af;h=p['h']-2*(scale-1)*Af.T@M@target
  H=(H+H.T)/2;s=max(abs(H).max(),abs(h).max(),1.)
  E,e=p['A_eq'],p['b_eq'];matrix=np.ascontiguousarray(np.vstack((E,p['G'])));lo=np.r_[p['lb'],e,np.full(len(p['b']),-np.inf)];hi=np.r_[p['ub'],e,p['b']]
  sense=np.zeros(len(hi),dtype=np.int32);sense[:n][p['lb']==p['ub']]=5;sense[n:n+len(e)]=5
  z,obj,flag,info=daqp.solve(np.ascontiguousarray(H/s),np.ascontiguousarray(h/s),matrix,hi,lo,sense,primal_tol=1e-8,dual_tol=1e-10,eps_prox=1e-8,eta_prox=1e-10,iter_limit=10000)
  violation=max(np.max(p['G']@z-p['b']),np.max(abs(E@z-e)),np.max(p['lb']-z),np.max(z-p['ub']),0.)
  body=d['task_body_orientation_A'][0]@z[:35]+d['task_body_orientation_bias'][0]-d['task_body_orientation_desired'][0]
  row['reweighted'].append(dict(scale=scale,status=int(flag),violation=float(violation),com_acceleration=(Af@z+bias).tolist(),body_acceleration_error=body.tolist()))
 rows.append(row);print(json.dumps(row),flush=True)
(root/'qp_authority.json').write_text(json.dumps(rows,indent=2)+'\n')
