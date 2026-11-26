from pathlib import Path
import numpy as np
import json
root=Path(__file__).resolve().parents[3]
p=root/'acc_data/g1_2_tracking/formula_wbc_verification/control_wbc.npz'
with np.load(p) as d:
 print('response fields',[k for k in d.files if any(s in k for s in ('observed','predicted','response','history_time'))])
 t=d['history_time'].ravel(); v=d['history_qd'].reshape(-1,35); predicted=d['history_qdd_predicted'].reshape(-1,35)
 modes=d['history_measured_mode'].ravel(); forces=d['history_foot_forces'].reshape(-1,2,3)
 dt=np.diff(t); actual=np.diff(v,axis=0)/dt[:,None]; residual=actual-predicted[:-1]
 stable=(modes[1:]==modes[:-1]) & (np.max(np.abs(np.diff(forces[:,:,2],axis=0)),axis=1)<100)
 rows=[]
 for a,b in [(0,1),(1,3),(3,3.8),(3.8,4.1),(4.1,4.3)]:
  ix=(t[1:]>=a)&(t[1:]<b)&stable
  r=residual[ix]
  row=dict(interval=[a,b],count=int(ix.sum()),mean_base=r[:,:6].mean(0).tolist(),rms_base=np.sqrt((r[:,:6]**2).mean(0)).tolist(),rms_joint=float(np.sqrt((r[:,6:]**2).mean())))
  rows.append(row);print(row)
 print('period',dt.min(),dt.max())
 (Path(__file__).parent/'response_audit.json').write_text(json.dumps(dict(source=str(p),intervals=rows),indent=2)+'\n')
