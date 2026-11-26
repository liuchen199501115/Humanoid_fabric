"""Counterfactual effort budgets; never exceed 80% of verified model limits."""
from pathlib import Path
import sys,json,numpy as np
from scipy import sparse
from scipy.optimize import linprog
root=Path(__file__).resolve().parents[3];out=Path(__file__).resolve().parent
sys.path.insert(0,str(root/'acc_data/amp_walk_20261103/feedback_followup'))
from solve_upper_fit_qp import solve
model=json.loads((out/'baseline_probe_v2.effort_model.json').read_text())
limits=.8*np.asarray(model['model_limits']);rows=[]
for p in sorted(out.glob('baseline_probe_v2.qp_*.npz')):
 with np.load(p,allow_pickle=False) as d:
  get=lambda k:d['priority_level_0_'+k][0].astype(float)
  H,h,E,e,G,b,original_lo,original_hi=[get(k) for k in ['H','h','A_eq','b_eq','G','b','lb','ub']];n=len(h)
  assert np.max(abs(d['correction']))==0
  for scope in ['ankles','all']:
   lo=original_lo.copy();hi=original_hi.copy()
   ii=np.array([4,5,10,11] if scope=='ankles' else list(range(29)))
   lo[47+ii]=-limits[ii];hi[47+ii]=limits[ii]
   A=np.c_[d['task_center_of_mass_A'][0],np.zeros((3,n-35))]
   target=d['task_center_of_mass_desired'][0]-d['task_center_of_mass_bias'][0]
   exact=linprog(np.zeros(n),A_ub=G,b_ub=b,A_eq=np.vstack((E,A)),b_eq=np.r_[e,target],bounds=np.c_[lo,hi],method='highs')
   C=sparse.csc_matrix(np.vstack((E,G,np.eye(n))))
   x,status=solve(sparse.csc_matrix(H),h,C,np.r_[e,np.full(len(G),-np.inf),lo],np.r_[e,b,hi])
   row=dict(time_s=float(d['time']),scope=scope,exact_com_feasibility_status=int(exact.status),solver=status)
   if status['accepted']:
    row.update(com_acceleration=(A@x+d['task_center_of_mass_bias'][0]).tolist(),com_requested=d['task_center_of_mass_desired'][0].tolist(),right_ankle_pitch_torque=float(x[57]))
   rows.append(row)
(out/'effort_audit.json').write_text(json.dumps(dict(scope='Offline budget counterfactual; use 80% of min(URDF, simulator) with unchanged physical constraints.',cases=rows),indent=2)+'\n')
print('Audited',len(rows),'effort counterfactuals')
