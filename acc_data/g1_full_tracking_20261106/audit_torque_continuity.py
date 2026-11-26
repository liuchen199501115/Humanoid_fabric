"""Independent saved-state QP sweep for torque continuity; never executed."""
from pathlib import Path
import json,numpy as np
from scipy import sparse
from solve_upper_fit_qp import solve
p=Path(__file__).parent;rows=[]
for name in ['rolling_contact_damping8.qp_2150.npz','pelvis_replace_before_turn.qp_1900.npz','root_margin150_turn.qp_2000.npz']:
 d=np.load(p/name);record=np.load(p/(name.split('.qp_')[0]+'.npz'));times=record['history_time'].ravel();index=int(abs(times-float(d['time'])).argmin())-1;previous=record['history_torque_qp_total'][index].reshape(29)
 prefix='priority_level_0_';get=lambda k:d[prefix+k][0].astype(float)
 H,h,E,e,G,b,lb,ub=[get(k) for k in ['H','h','A_eq','b_eq','G','b','lb','ub']];n=len(h);C=sparse.csc_matrix(np.vstack((E,G,np.eye(n))));lo=np.r_[e,np.full(len(b),-np.inf),lb];hi=np.r_[e,b,ub]
 for w in [0.,1.,10.,100.,1000.]:
  HH=H.copy();hh=h.copy();HH[47:76,47:76]+=w*np.eye(29);hh[47:76]-=w*previous
  x,report=solve(sparse.csc_matrix((HH+HH.T)/2),hh,C,lo,hi);row=dict(snapshot=name,weight=w,solve=report)
  if x is not None:
   row['root_acc']=x[:6].tolist();row['torque_delta_norm']=float(np.linalg.norm(x[47:76]-previous));row['torque_delta_peak']=float(abs(x[47:76]-previous).max());row['tasks']={}
   for task in ['center_of_mass','body_orientation','source_pelvis_orientation','joint_posture','feet_pose']:
    A,bias,desired,metric,active=[d['task_'+task+'_'+k][0] for k in ['A','bias','desired','metric','active']];r=(A@x[:35]+bias-desired)*active;row['tasks'][task]=float(r@metric@r)
  rows.append(row);print(name,w,row.get('torque_delta_peak'),row.get('root_acc'),flush=True)
(p/'torque_continuity_sweep.json').write_text(json.dumps(rows,indent=2)+'\n')
