"""Compare closed-loop trials without counting failed initialization as tracking."""
import json
from pathlib import Path
import numpy as np
OUT=Path(__file__).resolve().parent
rows=[];histories={}
for p in sorted(OUT.glob('*.result.json')):
 r=json.loads(p.read_text());name=r['name']
 record=OUT/(name+'.failure.npz')
 if not record.exists():record=OUT/(name+'.npz')
 if not record.exists() or 'tracking' not in r:continue
 assert r['tracking']['tolerances']==dict(joint_rmse_rad=.2,root_position_m=.2,root_orientation_rad=.35),name
 next_step=r['steps'].get('next_step_diagnostics')
 if next_step is not None:
  assert next_step['required_clearance_m']==.02
  assert next_step['required_landing_hold_s']==.05
  assert next_step['landing_position_tolerance_m']==.04
 with np.load(record) as a:
  t=a['history_time'];histories[name]={k:a[k].copy() for k in ['history_time','history_com','history_com_target','history_com_acc_original_desired','history_com_acc_qp_predicted','history_q','history_qd','history_qdd_qp','history_qdd_qp_reference'] if k in a}
  for k in ['history_priority_active','history_fabric_task_requests_cached','history_joint_tube_recovery_used']:assert not np.any(a[k]),(name,k)
  assert np.all(a['history_qp_native_solve_count']==1),name
  assert not np.any(a['history_torque_pd_estimate']),name
  names=a['joint_names'] if 'joint_names' in a else a['final_joint_order']
  order=[a['servo_joint_order'].tolist().index(n) for n in names]
  np.testing.assert_array_equal(a['servo_torque'][:,:,order],a['history_torque_qp_total'][a['servo_qp_update_index'].astype(int)])
  if 'history_qp_fabric_metric' in a and np.all(a['history_qp_fabric_metric']) and r['scale']==0:
   np.testing.assert_array_equal(a['history_qdd_qp_reference'],a['history_qdd_fabric'])
  y=a['history_com'][:,0,1]-a['history_com_target'][:,0,1]
  rows.append(dict(name=name,scale=r['scale'],measured_steps=r['steps']['measured_steps'],stop_time_s=r['tracking'].get('failed_time_s',r['steps'].get('last_time_s')),stop_reason=r['tracking'].get('stop_reason',r['steps'].get('stop_reason')),last_time=float(t[-1]),final_com_y_error_m=float(y[-1]),peak_com_y_error_m=float(abs(y).max()),single_qp=True,task_cache=False,applied_torque_equals_qp=True))
  if name=='baseline_v2':
   with np.load(OUT.parent/'g1_projected_acc_qp_20261114/continue_knee_20261114/baseline_final.failure.npz') as old:
    for k in ['history_time','history_q','history_qd','history_qdd_qp','servo_torque']:np.testing.assert_array_equal(a[k],old[k])
   rows[-1]['bitwise_matches_historical_baseline']=True
if 'baseline_v2' in histories:
 baseline=histories['baseline_v2']
 for row in rows:
  h=histories[row['name']];end=min(h['history_time'][-1],baseline['history_time'][-1])
  pair={}
  for label,data in [('trial',h),('baseline',baseline)]:
   error=(data['history_com']-data['history_com_target'])[:,0,1][data['history_time']<=end+1e-8]
   pair[label]=dict(rmse_m=float(np.sqrt(np.mean(error**2))),final_error_m=float(error[-1]))
  row['comparison_to_baseline']=dict(end_time_s=float(end),**pair)
  if row['name'].endswith('_late'):
   mask=h['history_time']<=.7
   for key in ('history_q','history_qd','history_qdd_qp','history_qdd_qp_reference'):
    np.testing.assert_array_equal(h[key][mask],baseline[key][baseline['history_time']<=.7])
   row['prefix_through_07_bitwise_equal']=True
if histories:
 end=min(h['history_time'][-1] for h in histories.values())
 for row in rows:
  h=histories[row['name']];idx=h['history_time']<=end+1e-8
  error=(h['history_com']-h['history_com_target'])[:,0,1][idx]
  row.update(common_end_time_s=float(end),common_com_y_rmse_m=float(np.sqrt(np.mean(error**2))),common_com_y_final_error_m=float(error[-1]))
(OUT/'results.json').write_text(json.dumps(dict(note='World Y; common horizon across completed trials. Startup failures are excluded. No acceptance thresholds changed.',trials=rows),indent=2)+'\n')
for row in rows:print(json.dumps(row))
