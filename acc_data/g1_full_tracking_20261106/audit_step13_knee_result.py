"""Verify the support-knee experiment's added row and unchanged prefix."""
from pathlib import Path
import json, numpy as np
p=Path(__file__).parent;name='thirteenth_support_knee20'
path=p/(name+'.npz')
if not path.exists():path=p/(name+'.failure.npz')
with np.load(path) as d,np.load(p/'rolling_contact.npz') as base:
 t=d['history_time'].ravel();a=d['history_qdd_predicted'].reshape(-1,35)
 active=d['history_support_knee_active'].reshape(-1,2)
 lower=d['history_support_knee_minimum_acceleration'].reshape(-1,2)
 error=np.where(active,lower-a[:,[9,15]],0.).clip(min=0)
 count=np.count_nonzero(t<=4.22+1e-9);prefix={}
 for key in ['history_time','history_frame','history_q','history_qd','history_torque_qp_total']:
  x,y=d[key][:count],base[key][:count]
  prefix[key]=dict(bitwise_equal=bool(np.array_equal(x,y)),maximum_absolute_difference=float(np.max(abs(x-y))))
 report=dict(run=name,valid_history_samples=len(t),active_constraint_samples=int(active.sum()),
             peak_added_constraint_violation=float(error.max()),
             prefix_cutoff_s=4.22,prefix_samples=count,prefix=prefix,
             prefix_identical=all(v['bitwise_equal'] for v in prefix.values()),
             scope='Constraint and prefix checks only. The separate raw-source/landing audit determines tracking success.')
(p/'step13_knee_result.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
