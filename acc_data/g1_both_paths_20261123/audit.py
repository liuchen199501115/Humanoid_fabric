"""Audit both paths and compare each fresh run to its previous independent run."""
import json
from pathlib import Path
import numpy as np
OUT=Path(__file__).resolve().parent;OLD=OUT.parent/'g1_com_feedback_20261123'
prior={'baseline':'baseline_v2','com_original':'feedback1_v2','com_delayed':'feedback4_late','fabric_metric':'fabric_metric'}
def record(folder,name):
 for suffix in ('.failure.npz','.npz'):
  p=folder/(name+suffix)
  if p.exists():return p
 return None
rows=[];histories={};hashes=[]
for name,previous in prior.items():
 rp=OUT/(name+'.result.json');path=record(OUT,name)
 if not rp.exists():continue
 result=json.loads(rp.read_text())
 if not result['tracking_evaluated'] or path is None:
  rows.append(dict(name=name,tracking_evaluated=False,returncode=result['returncode'],timed_out=result['timed_out']));continue
 tracking=result['tracking'];steps=result['steps']
 assert tracking['tolerances']==dict(joint_rmse_rad=.2,root_position_m=.2,root_orientation_rad=.35)
 step=steps.get('next_step_diagnostics')
 if step:
  assert (step['required_clearance_m'],step['required_landing_hold_s'],step['landing_position_tolerance_m'])==(.02,.05,.04)
 hashes.append(json.loads((OUT/(name+'.sources.json')).read_text()))
 with np.load(path) as a,np.load(record(OLD,previous)) as old:
  for field in ['history_priority_active','history_fabric_task_requests_cached','history_joint_tube_recovery_used','history_torque_pd_estimate']:assert not np.any(a[field]),(name,field)
  assert np.all(a['history_qp_native_solve_count']==1),name
  if 'diagnostic_qp_native_solve_count' in a:assert np.all(a['diagnostic_qp_native_solve_count']==1)
  joint_names=a['joint_names'] if 'joint_names' in a else a['final_joint_order']
  order=[a['servo_joint_order'].tolist().index(n) for n in joint_names]
  np.testing.assert_array_equal(a['servo_torque'][:,:,order],a['history_torque_qp_total'][a['servo_qp_update_index'].astype(int)])
  if name=='fabric_metric':
   assert np.all(a['history_qp_fabric_metric'])
   np.testing.assert_array_equal(a['history_qdd_qp_reference'],a['history_qdd_fabric'])
  else:assert not np.any(a['history_qp_fabric_metric'])
  scale=a['history_qp_com_feedback_scale'];t=a['history_time']
  if name=='com_original':assert np.all(scale==1)
  elif name=='com_delayed':
   assert np.all(scale[t<=.7]==0)
   np.testing.assert_allclose(scale[t>=.8],4,rtol=0,atol=1e-12)
  else:assert np.all(scale==0)
  error=(a['history_com']-a['history_com_target'])[:,0,1]
  histories[name]=dict(time=t.copy(),error=error.copy())
  equality={k:bool(np.array_equal(a[k],old[k])) for k in ['history_time','history_q','history_qd','history_qdd_qp','servo_torque']}
  qp_equal={k:bool(np.array_equal(a[k],old[k])) for k in ('problem_H','problem_h','problem_G','problem_b','problem_A_eq','problem_b_eq','problem_lb','problem_ub') if k in a and k in old}
  rows.append(dict(name=name,tracking_evaluated=True,options=result['options'],measured_steps=steps['measured_steps'],stop_time_s=tracking.get('failed_time_s',steps.get('last_time_s')),stop_reason=tracking.get('stop_reason'),peak_joint_rmse_rad=tracking['peak_joint_rmse'],peak_root_position_m=tracking['peak_root_position_error'],peak_root_orientation_rad=tracking['peak_root_orientation_error'],final_com_y_error_m=float(error[-1]),previous_run=previous,bitwise_equal_to_previous=equality,failure_qp_bitwise_equal=qp_equal,single_qp=True,no_task_cache=True,applied_torque_equals_qp=True))
if hashes:assert all(h==hashes[0] for h in hashes)
if 'baseline' in histories:
 for row in rows:
  if not row['tracking_evaluated']:continue
  h=histories[row['name']];b=histories['baseline'];end=min(h['time'][-1],b['time'][-1]);pair={}
  for label,data in [('trial',h),('baseline',b)]:
   e=data['error'][data['time']<=end+1e-9];pair[label]=dict(final_error_m=float(e[-1]),rmse_m=float(np.sqrt(np.mean(e**2))))
  row['matched_time_com_y_comparison']=dict(end_time_s=float(end),**pair)
report=dict(expected_runs=list(prior),completed_runs=len(rows),controller_sources_identical_across_runs=True if hashes else None,scope='Fresh independent closed-loop runs from the start. Same source/clock/physics/acceptance thresholds. World-Y COM position error. Return code 1 may be an intentionally stopped failed tracking trial.',trials=rows)
(OUT/'results.json').write_text(json.dumps(report,indent=2)+'\n')
for r in rows:print(json.dumps(r))
