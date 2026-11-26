"""Check added knee acceleration barriers in saved QPs, without execution."""
from pathlib import Path
import json,numpy as np
from scipy import sparse
from solve_upper_fit_qp import solve
p=Path(__file__).parent;rows=[]
for name in ['thirteenth_waist_metric4.qp_2100','thirteenth_waist_metric4.qp_2150']:
 d=np.load(p/(name+'.npz'));get=lambda k:d['priority_problem_'+k][0].astype(float)
 H,h,E,e,G,b,lo,hi=[get(k) for k in ['H','h','A_eq','b_eq','G','b','lb','ub']];q=d['q'][0];v=d['qd'][0];N=len(h)
 lower=-40*v[15]-400*(q[16]-.2);guard=np.zeros((1,N));guard[0,15]=-1
 C=sparse.csc_matrix(np.vstack((E,G,guard,np.eye(N))));low=np.r_[e,np.full(len(G)+1,-np.inf),lo];high=np.r_[e,b,-lower,hi]
 x,status=solve(H,h,C,low,high);row=dict(sample=name,knee_position=float(q[16]),knee_velocity=float(v[15]),minimum_acceleration=float(lower),solver=status)
 if status['accepted']:row.update(predicted_knee_acceleration=float(x[15]),root_acceleration=x[:6].tolist())
 rows.append(row);print(json.dumps(row),flush=True)
(p/'step13_knee_guard_feasibility.json').write_text(json.dumps(dict(scope='Added constraint, offline solves only',cases=rows),indent=2)+'\n')
