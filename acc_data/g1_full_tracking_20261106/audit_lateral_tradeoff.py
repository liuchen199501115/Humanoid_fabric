"""Saved-state task tradeoff when enforcing one feasible COM acceleration row."""
import json
from pathlib import Path
import numpy as np
from scipy import sparse
from solve_upper_fit_qp import solve
p=Path(__file__).parent;rows=[]
for name in ['pelvis_replace_before_turn.qp_1900.npz','root_margin150_turn.qp_2000.npz','rolling_contact_damping8.qp_2150.npz']:
 d=np.load(p/name);prefix='priority_level_0_';get=lambda k:d[prefix+k][0].astype(float)
 H,h,E,e,G,b,lb,ub=[get(k) for k in ['H','h','A_eq','b_eq','G','b','lb','ub']];n=len(h)
 c=np.zeros(n);c[:35]=d['task_center_of_mass_A'][0,1]
 target=float(d['task_center_of_mass_desired'][0,1]-d['task_center_of_mass_bias'][0,1])
 C=sparse.csc_matrix(np.vstack((E,G,np.eye(n),c)));lo=np.r_[e,np.full(len(b),-np.inf),lb,target];hi=np.r_[e,b,ub,target]
 x,report=solve(sparse.csc_matrix((H+H.T)/2),h,C,lo,hi)
 row=dict(snapshot=name,solve=report)
 if x is not None:
  row['root_acc_before']=d['predicted_acc'][0,:6].tolist();row['root_acc_after']=x[:6].tolist();row['tasks']={}
  for task in ['center_of_mass','body_orientation','source_pelvis_orientation','joint_posture','feet_pose']:
   A,bias,desired,metric,active=[d['task_'+task+'_'+k][0] for k in ['A','bias','desired','metric','active']]
   before=(A@d['predicted_acc'][0]+bias-desired)*active;after=(A@x[:35]+bias-desired)*active
   row['tasks'][task]=dict(residual_before=before.tolist(),residual_after=after.tolist(),cost_before=float(before@metric@before),cost_after=float(after@metric@after))
 rows.append(row)
 print(name,report, 'root',row.get('root_acc_after'), flush=True)
(p/'lateral_tradeoff_audit.json').write_text(json.dumps(rows,indent=2)+'\n')
