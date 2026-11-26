"""Equivalent variable/row scaling of rejected QPs; original KKT certification."""
import json
from pathlib import Path
import numpy as np
import daqp
from fabrics_sim.solver.qp_refinement import _violation

OUT=Path(__file__).resolve().parent
results=[]
for name in ['com_early16','support7_15','support7_30','eight_scaled','support30_scaled','com16_scaled']:
 d=np.load(OUT/f'{name}.failure.npz')
 p={k:d['soft_problem_'+k][0].astype(float) for k in ['H','h','G','b','lb','ub','A_eq','b_eq']}
 H=.5*(p['H']+p['H'].T);n=len(p['h']);E=p['A_eq'];e=p['b_eq'];matrix=np.vstack((E,p['G']))
 low=np.r_[p['lb'],e,np.full(len(p['b']),-np.inf)];high=np.r_[p['ub'],e,p['b']]
 sense=np.zeros(len(high),dtype=np.int32);sense[:n][p['lb']==p['ub']]=5;sense[n:n+len(e)]=5
 full=np.vstack((np.eye(n),matrix));scale=max(1.,abs(H).max(),abs(p['h']).max())
 record=dict(name=name,cases=[])
 for cap in [100.]:
  D=np.minimum(cap,np.sqrt(max(np.diag(H))/np.maximum(np.diag(H),1e-12)))
  for tolerance in [1e-8,1e-9,1e-10]:
   norm=False
   mat=matrix*D[None,:];row=np.maximum(np.linalg.norm(mat,axis=1),1.) if norm else np.ones(len(mat))
   lo=np.r_[p['lb']/D,e/row[:len(e)],np.full(len(p['b']),-np.inf)];hi=np.r_[p['ub']/D,np.r_[e,p['b']]/row]
   y,_,flag,info=daqp.solve(np.ascontiguousarray(H*D[:,None]*D[None,:]/scale),np.ascontiguousarray(p['h']*D/scale),np.ascontiguousarray(mat/row[:,None]),hi,lo,sense,
        primal_tol=tolerance,dual_tol=1e-10,eps_prox=1e-4,eta_prox=1e-12,iter_limit=5000)
   x=D*y;lam=info['lam']/np.r_[D,row]
   stationarity=float(abs(H@x/scale+p['h']/scale+full.T@lam).max())
   residual=float(_violation(p,x));z=full@x;upper=(sense!=5)&(lam>0);lower=(sense!=5)&(lam<0)
   comp=float(max(abs(lam[upper]*(high[upper]-z[upper])).max(initial=0.),abs(lam[lower]*(low[lower]-z[lower])).max(initial=0.)))
   record['cases'].append(dict(cap=cap,primal_tolerance=tolerance,row_normalization=norm,flag=int(flag),iterations=int(info['iterations']),physical_violation=residual,stationarity=stationarity,complementarity=comp,
      accepted=bool(flag in [1,-2,-4,-7] and residual<=1e-6 and stationarity<=1e-8 and comp<=1e-7)))
 results.append(record);print(record,flush=True)
(OUT/'scaling_tolerance.json').write_text(json.dumps(results,indent=2)+'\n')
