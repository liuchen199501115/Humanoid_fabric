"""Check that late experiments really preserve the recorded successful prefix."""
from pathlib import Path
import json,numpy as np
p=Path(__file__).parent;base=np.load(p/'rolling_contact.npz');rows=[]
for name,cutoff in [('torque_continuity100_after12',4.22),('stance_heading_turn_v3',4.09),('balance_pose_after12',4.22)]:
 path=p/(name+'.npz')
 if not path.exists():path=p/(name+'.failure.npz')
 if not path.exists():continue
 d=np.load(path);a=base['history_time'].ravel();b=d['history_time'].ravel();n=min(np.count_nonzero(a<=cutoff+1e-9),np.count_nonzero(b<=cutoff+1e-9));result=dict(run=name,cutoff_s=cutoff,samples=n,fields={})
 for key in ['history_time','history_frame','history_q','history_qd','history_torque_qp_total']:
  x,y=base[key][:n],d[key][:n];result['fields'][key]=dict(bitwise_equal=bool(np.array_equal(x,y)),maximum_absolute_difference=float(abs(x-y).max()))
 result['identical']=all(v['bitwise_equal'] for v in result['fields'].values());rows.append(result);print(name,n,result['identical'])
(p/'late_prefix_audit.json').write_text(json.dumps(rows,indent=2)+'\n')
