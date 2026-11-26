"""Offline feasibility/certificate test of the saved full-metric failure."""
import json
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import linprog
from fabrics_sim.solver.admm_qp import QPData
from fabrics_sim.solver.daqp_backend import DAQPBackend
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
source=OUT/'fabric_metric.failure.npz'
if not source.exists():source=ROOT/'acc_data/g1_com_feedback_20261123/fabric_metric.failure.npz'
with np.load(source) as a:
 p={k:a['problem_'+k].astype(float).copy() for k in ('H','h','G','b','lb','ub','A_eq','b_eq')}
 saved=a['problem_solution'][0].astype(float)
 hist=a['history_time'].copy()
def residual(z):
 return max(0.,float(np.max(p['G'][0]@z-p['b'][0])),float(np.max(abs(p['A_eq'][0]@z-p['b_eq'][0]))),float(np.max(p['lb'][0]-z)),float(np.max(z-p['ub'][0])))
report=dict(source=str(source.relative_to(ROOT)),scope='offline frozen-state only; no controller constraints or solver tolerances changed',last_applied_time_s=float(hist[-1]),saved_primal_residual=residual(saved),cases=[])
lp=linprog(np.zeros(76),A_ub=p['G'][0],b_ub=p['b'][0],A_eq=p['A_eq'][0],b_eq=p['b_eq'][0],bounds=list(zip(p['lb'][0],p['ub'][0])),method='highs')
report['linear_feasibility']=dict(status=int(lp.status),message=lp.message,residual=residual(lp.x) if lp.success else None)
for single in [True,False]:
 solver=DAQPBackend(single_solve=single)
 sol=solver.solve(QPData(**{k:torch.from_numpy(v) for k,v in p.items()}))
 report['cases'].append(dict(single_solve=single,accepted=bool(sol.status_ok.item()),status=solver.last_status,native_solve_counts=solver.last_solve_counts,primal_residual=residual(sol.z[0].numpy())))
# Reproduce the backend's native, scaled KKT certificate without changing it.
import daqp
H=.5*(p['H'][0]+p['H'][0].T);h=p['h'][0];n=len(h)
E=p['A_eq'][0];e=p['b_eq'][0];G=p['G'][0];b=p['b'][0]
scale=max(1.,abs(H).max(),abs(h).max());matrix=np.ascontiguousarray(np.vstack((E,G)))
lower=np.r_[p['lb'][0],e,np.full(len(b),-np.inf)];upper=np.r_[p['ub'][0],e,b]
sense=np.zeros(len(upper),np.int32);sense[:n][p['lb'][0]==p['ub'][0]]=5;sense[n:n+len(e)]=5
x,_,flag,info=daqp.solve(np.ascontiguousarray(H/scale),np.ascontiguousarray(h/scale),matrix,upper,lower,sense,primal_tol=1e-8,dual_tol=1e-10,eps_prox=-1e-8,eta_prox=1e-8,iter_limit=5000)
full=np.vstack((np.eye(n),matrix));lam=info['lam'];y=full@x;ineq=sense!=5;high=ineq&(lam>0);low=ineq&(lam<0)
report['native_certificate']=dict(flag=int(flag),stationarity=float(abs(H@x/scale+h/scale+full.T@lam).max()),stationarity_limit=1e-8,complementarity=float(max(abs(lam[high]*(upper[high]-y[high])).max(initial=0.),abs(lam[low]*(lower[low]-y[low])).max(initial=0.))),complementarity_limit=1e-7,primal_residual=residual(x),primal_limit=1e-6)
(OUT/'fabric_failure_check.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
