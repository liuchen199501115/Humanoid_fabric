import numpy as np,json
from pathlib import Path
p=Path('acc_data/amp_walk_20261102/continuation/measured_load.failure.npz');d=np.load(p)
g=lambda k:d['history_'+k].reshape(len(d['history_time']),-1)
t=g('time')[:,0];v=g('com_velocity');pred=g('com_acc_qp_predicted');obs=g('com_acc_observed');cmd=g('commanded_mode')[:,0];meas=g('measured_mode')[:,0];q=g('q');ref=g('reference_q');forces=g('ground_foot_forces').reshape(-1,2,3)
rows=[]
for a,b in [(0,.1),(.1,.2),(.2,.32),(.32,.44),(.44,.52),(.52,.8),(.8,.94),(.94,1.1),(1.1,1.26),(1.26,1.4),(1.4,1.5),(1.5,1.7),(1.7,1.9)]:
 mask=(t>=a)&(t<b);ids=np.flatnonzero(mask);i=ids[-1]
 row=dict(interval=[a,b],pred_dv=(pred[mask]*.002).sum(0).round(4).tolist(),obs_dv=(obs[mask]*.002).sum(0).round(4).tolist(),root_error=(q[i,:3]-ref[i,:3]).round(4).tolist(),end_com_velocity=v[i].round(4).tolist(),max_force=forces[mask,:,2].max(0).round(1).tolist(),max_mismatch=float(abs(obs[mask]-pred[mask]).max()))
 rows.append(row);print(row)
print('largest errors')
ids=np.argsort(np.linalg.norm(obs[:950]-pred[:950],axis=1))[-15:]
for i in ids:print(round(t[i],3), 'mode',cmd[i],meas[i], 'f',forces[i,:,2].round(1),'pred',pred[i].round(2),'obs',obs[i].round(2),'dv',(0.002*(obs[i]-pred[i])).round(3))
Path('acc_data/amp_walk_20261102/impact_followup/response_intervals.json').write_text(json.dumps(rows,indent=2)+'\n')
