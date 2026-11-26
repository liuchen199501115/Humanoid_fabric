"""Supplement the original-source landing audit with velocity/slack evidence."""
import json
from pathlib import Path
import numpy as np

OUT=Path(__file__).resolve().parent/'com_velocity'
results=json.loads((OUT/'results.json').read_text())
rows=[]
for trial in results['trials']:
    if 'recording' not in trial:
        continue
    row=dict(run=trial['run'],steps=trial['raw_source_audit']['prefix_landings'],
             stop_time_s=trial['last_time_s'],guard=trial.get('com_velocity_soft'))
    with np.load(trial['recording']) as d:
        t=d['history_time'].ravel()
        velocity_error=(d['history_com_velocity']-d['history_com_target_velocity']).reshape(-1,3)
        row['intervals']=[]
        for lo,hi in ((3.3,3.58),(3.58,3.8),(3.8,4.2)):
            m=(t>=lo)&(t<=hi)
            if t[-1]<hi or not m.any():continue
            row['intervals'].append(dict(start=lo,end=hi,
                com_velocity_rms_norm_mps=float(np.sqrt(np.mean(np.sum(velocity_error[m]**2,axis=1))))))
        if 'history_com_soft_activation' in d:
            options=json.loads(d['controller_options'].item())
            horizon=options['com_velocity_soft_horizon']
            ids=np.flatnonzero((d['history_com_soft_activation'].ravel()>0)&(t+horizon<=t[-1]))
            future=ids+round(horizon/np.median(np.diff(t)))
            forecast=d['history_com_soft_forecast_error'].reshape(-1,3)
            row['forecast_comparison']=dict(samples=len(ids),
                future_error_prediction_rms_norm_mps=float(np.sqrt(np.mean(np.sum(
                    (velocity_error[future]-forecast[ids])**2,axis=1)))) if len(ids) else None,
                actual_future_error_smaller_fraction=float((np.linalg.norm(velocity_error[future],axis=1)
                    <np.linalg.norm(velocity_error[ids],axis=1)).mean()) if len(ids) else None)
    row['augmented_snapshots']=[]
    for path in sorted(OUT.glob(row['run']+'.com_soft_*.npz')):
        with np.load(path) as s:
            x=s['solution']
            inequality=float(np.maximum((s['G']@x[...,None])[...,0]-s['b'],0).max())
            equality=float(abs((s['A_eq']@x[...,None])[...,0]-s['b_eq']).max()) if 'A_eq' in s else 0.
            bounds=float(np.maximum(np.maximum(s['lb']-x,x-s['ub']),0).max())
            row['augmented_snapshots'].append(dict(file=path.name,inequality=inequality,
                equality=equality,bounds=bounds,slack_mps=s['slack'].tolist()))
    rows.append(row)
(OUT/'velocity_audit.json').write_text(json.dumps(rows,indent=2)+'\n')
for row in rows:
    print(json.dumps({k:v for k,v in row.items() if k!='augmented_snapshots'}))
