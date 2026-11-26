"""Counterfactual saved first-stage costs; solutions are never applied."""
from pathlib import Path
import json,numpy as np
from scipy import sparse
from solve_upper_fit_qp import solve
p=Path(__file__).parent;reports=[]
for sample in ['thirteenth_waist_metric4.qp_2100','rolling_contact.qp_2150']:
 path=p/(sample+'.npz')
 if not path.exists():continue
 d=np.load(path);prefix='priority_problem_';get=lambda k:d[prefix+k][0].astype(float)
 H,h,E,e,G,b,lo,hi=[get(k) for k in ['H','h','A_eq','b_eq','G','b','lb','ub']]
 A=d['task_joint_posture_A'][0].astype(float);M=d['task_joint_posture_metric'][0].astype(float);active=d['task_joint_posture_active'][0];M*=active[:,None]*active[None,:];r=d['task_joint_posture_bias'][0]-d['task_joint_posture_desired'][0];N=len(h);J=np.c_[A,np.zeros((29,N-35))]
 C=sparse.csc_matrix(np.vstack([E,G,np.eye(N)]));lower=np.r_[e,np.full(len(G),-np.inf),lo];upper=np.r_[e,b,hi]
 for case,rows in [('all',range(29)),('upper',range(12,29)),('upper_and_swing',list(range(6))+list(range(12,29)))]:
  mask=np.zeros(29);mask[list(rows)]=1;MM=M*mask[:,None]*mask[None,:];HH=H+J.T@(MM-M)@J;hh=h+J.T@(MM-M)@r
  x,report=solve(HH,hh,C,lower,upper);row=dict(sample=sample,primary_posture=case,solver=report)
  if report['accepted']:
   row.update(root_acceleration=x[:6].tolist(),waist_acceleration=x[18:21].tolist(),leg_acceleration_norm=float(np.linalg.norm(x[6:18])),posture_residual_norm=float(np.linalg.norm(J@x+r)))
  reports.append(row);print(json.dumps(row),flush=True)
(p/'step13_posture_partition.json').write_text(json.dumps(dict(scope='Offline diagnostics only, no QP solution applied',cases=reports),indent=2)+'\n')
